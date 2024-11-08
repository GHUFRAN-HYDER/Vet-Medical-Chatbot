# Vet Medical Chatbot
This project is a Veterinarian Q&A Assistant powered by LangChain, Pinecone, and OpenAI's GPT. It uses retrieval-augmented generation (RAG) to answer questions on pet health. The assistant, referred to as "Dr. Steve," provides responses based only on the context of loaded documents, avoiding hallucination by sticking strictly to provided information.

## Features
- PDF Text Processing: Loads and processes content from PDF files.
- Document Splitting by Cases: Splits large text into individual cases for focused answers.
- Pinecone Vector Store: Stores document embeddings for efficient similarity search.
- Conversational Retrieval Chain: Enables conversational interaction, retrieving relevant information from the documents.
- Streamlit Interface: Simple, user-friendly interface for querying the assistant in a chat-like environment.
### Prerequisites
- Python 3.8 or later
- Access to Pinecone and OpenAI API keys

## Installation

- Install dependencies
```
pip install -r requirements.txt
```
- Add API keys:
Set PINECONE_API_KEY and OPENAI_API_KEY in your environment or Streamlit secrets.
## Usage
Run the Streamlit App
```
streamlit run chatmed-v6.py
```
## Interact with Dr. Steve
- Enter your query in the chat input, and Dr. Steve will provide responses based on the context from medical documents.

## Code Overview
- initialize_resources():
Loads a PDF file and splits it by cases.
Stores document embeddings in Pinecone and initializes a conversational retrieval chain.
main():
Initializes the RAG chain and sets up the Streamlit interface.
Handles user input and responses, displaying a conversational history.
- Configuration
The following settings can be customized:

- Document Processing: Adjust the case-splitting logic within split_cases().
- Pinecone Configuration: Change index_name, dimension, and metric as needed for your use case.
- Prompt Template: Modify the assistant's instructions and response handling in the prompt setup.
- Dependencies
A requirements.txt file with all required packages is included.