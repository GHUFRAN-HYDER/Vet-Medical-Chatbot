import logging
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
import time
import getpass
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import re
from langchain.text_splitter import CharacterTextSplitter


# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Medical Chatbot", layout="centered")
# Load environment variables from .env file
load_dotenv()

# Get the Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the text splitter to split by cases
def split_cases(text):
    cases = re.split(r"(Case \d+)", text)  # Split by case numbers
    docs = []
    for i in range(1, len(cases), 2):
        case_text = f"{cases[i]} {cases[i + 1]}"  # Combine "Case X" with its content
        docs.append(case_text.strip())
    return docs



@st.cache_resource
def initialize_resources():
    # Load and split PDF
    loader = PyPDFLoader("I:/UPWORK/ai-chatbot/Cushing's.pdf")

    raw_docs = loader.load_and_split()
 
    # Concatenate all page contents into a single string
    # raw_text = " ".join(doc.page_content for doc in raw_docs)
    
    # Apply the case splitter to the concatenated text
    # case_texts = split_cases(raw_text)

    # # Create documents
    # docs = [Document(page_content=case) for case in case_texts]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(raw_docs)

    # Verify the content and length of each chunk
    # for i, doc in enumerate(docs):
    #     logging.info(f"Chunk {i+1} length: {len(doc.page_content)} characters")
    # Generate embeddings
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
    
    # Initialize retriever and LLM
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Create the RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Dr. Steve, a veterinarian who provides advice on pet health.You must not include information or any text on you own.

        You must not miss any information in the context.You must give the exact answer in the context with no hallucination.
        
        
        If the context does not contain the answer to the user's question, reply with:
    "I'm sorry, I couldn't find the exact information you're looking for. For further assistance, please reach out to the Facebook group Ask Dr. Steve DVM® at https://www.facebook.com/groups/1158575954706282.
        {context}
        """),
        ("human", "{input}")
    ])
    
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Initialize the RAG chain
rag_chain = initialize_resources()

def main():

    st.title("Vet Q&A Assistant")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(""):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
