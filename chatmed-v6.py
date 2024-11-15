import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_pdf():
    """Load PDF and return debug information"""
    try:
        loader = PyPDFLoader("Cushing's.pdf")
        raw_docs = loader.load_and_split()    
        return raw_docs
    
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

def split_documents(raw_docs):
    """Split documents and return debug information"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(raw_docs)
             
        return docs
    
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

@st.cache_resource
def initialize_resources():
    # Load and split PDF with debugging
    raw_docs = load_pdf()
    docs = split_documents(raw_docs)
      
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # # Setup Pinecone
    pc = Pinecone(api_key=pinecone_api_key )
    index_name = "chatmed-index"

# Create index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Check if index is empty before adding documents
    if len(index.fetch(ids=["1"]).vectors) == 0:  # Check if index is empty
        try:
            vectorstore.add_documents(docs)
            logging.info("Documents added to empty index")
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
    else:
        logging.info("Index already contains vectors, skipping document addition")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """
    You are Dr. Steve, a veterinarian who provides advice on Cushings diseas of Dogs. Use the following guidelines:
    Do not infer or assume anything.
    You must not hallucinate or give information which is not in the context
    Provide detailed information strictly based on the context. Do not include any advice or responses not directly found within the content. 
     
    Context for veterinary-related questions:
    
    {context}
    """
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return retriever, rag_chain

# Initialize the resources
retriever, rag_chain = initialize_resources()


def process_query(prompt, chat_history):
    """Process the query with enhanced debugging"""
    try:
        # Get relevant documents first
        relevant_docs = retriever.get_relevant_documents(prompt)
        
        # Process with RAG chain
        response = rag_chain.invoke({
            "input": prompt, 
            "chat_history": chat_history
        })
        
        return response["answer"], relevant_docs
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing your question: {str(e)}", []

def main():
    st.title("Cushings Medicine Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about ..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            chat_history = [(msg["role"], msg["content"]) 
                          for msg in st.session_state.messages[:-1]]
            
            answer, relevant_docs = process_query(prompt, chat_history)
            st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
