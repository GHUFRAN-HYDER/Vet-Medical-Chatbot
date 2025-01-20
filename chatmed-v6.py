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
from langchain_core.documents import Document
import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

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
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def load_csv_data():
    """Load CSV data combining post_text and comment columns"""
    try:
        file_path = "posts-comments-time.csv"
        
        loader = CSVLoader(
            file_path=file_path,
            encoding="MacRoman"
        )
        
        documents = []
        raw_docs = loader.load()
        
        for doc in raw_docs:
            post = doc.metadata.get('post_text', '')
            comment = doc.metadata.get('comment', '')
            doc.page_content = f"Question: {post}\nAnswer: {comment}"
            documents.append(doc)
            
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise

def split_documents(raw_docs):
    """Split documents and return debug information"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(raw_docs)
        return docs
    
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

# def parse_qa_pairs(file_path):
#     """Parse raw text into structured Q&A pairs and return LangChain Documents."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         content = file.read()

#     # Split text by "Question:" delimiter
#     raw_pairs = content.split("Question:")
#     documents = []

#     for pair in raw_pairs:
#         if "Answer:" in pair:
#             question, answer = pair.split("Answer:", 1)
#             documents.append(Document(
#                 page_content=f"Question: {question.strip()}\nAnswer: {answer.strip()}",
#                 metadata={}  # Add metadata here if needed
#             ))

#     return documents
    
@st.cache_resource
def initialize_resources():
    # Load documents
    documents = load_csv_data()
    docs = split_documents(documents)

    # qa_pairs = parse_qa_pairs("questions_answers_formatted.txt")
    # docs = qa_pairs 
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Setup Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "chatmed-index"

    # Create index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)

    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    if not index.describe_index_stats().total_vector_count:
        try:
            vectorstore.add_documents(docs)
            logging.info("Documents added to empty index")
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
    else:
        logging.info("Index already contains vectors, skipping document addition")
        
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    api_key = os.getenv("OPENAI_API_KEY")
    logging.info(f"API key being used: {api_key}")
    llm = ChatOpenAI(model="gpt-4", temperature=0)  # Fixed model name typo
    
    system_prompt = """
    Act as Dr. Steve, a veterinarian specializing in Cushing's disease in dogs. Using the provided context, answer the query thoroughly without omitting any important details from retrieved chunks. 
    Guidelines:
    Response must be in paragrapgh format
    Do NOT infer or assume additional symptoms, conditions, or causes beyond what is provided in the user's query and context.
    Consolidate all relevant information from the retrieved chunks.
    Address the question comprehensively, integrating advice and recommendations, causes and reasons in the retrieved chunks.
    Explain ALL cause-and-effect relationships found in the context
    Include any hyperlinks or references mentioned in the retrieved chunks like for medicine buying
    State ALL success rates and improvement timeframes mentioned
    Provide ALL dietary specifications and restrictions
     Share ALL supplement recommendations with their specific sources/availability
     Exclusions:
     Do NOT include cautionary statements such as recommending regular check-ups, monitoring, or consulting veterinarians.
     Do NOT add general advice about professional oversight or adjustments.
    If all of the retrieved chunks donot contain the relevant information, reply with:
    "I don't have the information you're looking for . For further assistance, 
    Please reach out to the Facebook group Ask Dr. Steve DVM® at https://www.facebook.com/groups/1158575954706282.
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

def main():
    st.title("Dr. Steve Chat Assistant")
    
    # Initialize resources if not already done
    if st.session_state.retriever is None or st.session_state.rag_chain is None:
        st.session_state.retriever, st.session_state.rag_chain = initialize_resources()
    
    # Add debug toggle
    show_debug = st.sidebar.checkbox("Show Debug Information")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about Cushing's disease..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            chat_history = [(msg["role"], msg["content"]) 
                          for msg in st.session_state.messages[:-1]]
            
            try:
                # Get relevant documents first
                relevant_docs = st.session_state.retriever.get_relevant_documents(prompt)
                
                # Show debug information if enabled
                if show_debug:
                    with st.expander("Debug: Retrieved Documents"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Document {i+1}:**")
                            st.text(doc.page_content)
                            st.markdown("---")
                
                # Process with RAG chain
                response = st.session_state.rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": chat_history
                })
                
                st.markdown(response["answer"])
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
