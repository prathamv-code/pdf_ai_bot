# Import necessary libraries
import streamlit as st  # To build the web app interface
from PyPDF2 import PdfReader  # To extract text from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For text chunking
import os  # For environment and file operations
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google Gemini embeddings for vectorization
import google.generativeai as genai  # For Gemini API configuration
from langchain.vectorstores import FAISS  # For the FAISS vector store (semantic search)
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini chat model
from langchain.chains.question_answering import load_qa_chain  # For the question-answering pipeline
from langchain.prompts import PromptTemplate  # For custom prompt templates
from dotenv import load_dotenv  # To load environment variables

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Gemini API with your key


# Extract text from one or more uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    # Loop through each PDF file uploaded
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Extract text from every page in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split the full text into chunks to make them suitable for LLM context length limits
def get_text_chunks(text):
    # Create a text splitter object with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Convert text chunks into embeddings and store them in a FAISS index for fast retrieval
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the index to disk for future use


# Set up the Gemini conversational chain for QA, with a custom prompt template
def get_conversational_chain():
    # Define custom prompt to instruct the model on how to answer
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize Gemini model with custom temperature for answer randomness
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a QA chain combining the Gemini model and custom prompt behavior
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Handle user questions, search the vector store for context, and fetch answer from LLM
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the existing FAISS index with the same embeddings as used before
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search to get relevant document chunks
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    # Run the QA chain with retrieved context and the user's question
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    print(response)  # Optionally print the result in the console for debugging
    st.write("Reply: ", response["output_text"])  # Display the answer in the app


# Main Streamlit app structure
def main():
    # Basic app setup and UI headers
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF💁")

    # Text box to receive user's question about uploaded PDFs
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)  # Handle and answer question

    with st.sidebar:
        st.title("Menu:")
        # File uploader for accepting multiple PDF files
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Extract full text from PDFs
                text_chunks = get_text_chunks(raw_text)  # Split into chunks
                get_vector_store(text_chunks)  # Store in FAISS vector DB
                st.success("Done")  # Notify user of completion


# Run the Streamlit app
if __name__ == "__main__":
    main()
