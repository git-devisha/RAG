import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated Import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))  # Use BytesIO for Streamlit UploadedFile
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""  # Handle None case
            text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store  # Now returning the vector store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response.get("output_text", "No response generated."))
    except FileNotFoundError:
        st.error("No processed data found. Please upload and process a PDF first.")

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="wide")
    st.header("ChatPDF powered by Gemini 🤖")
    
    # Apply Light Theme CSS
    st.markdown(
        """
        <style>
            body {
                background-color: #f8f9fa; /* Soft Background */
                color: black;
            }
            .stChatMessage {
                background-color: #e8eaf6; /* Soft Lavender */
                color: black;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .stChatUserMessage {
                background-color: #c8e6c9; /* Light Green */
                color: black;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                text-align: right;
            }
            .stButton > button {
                background-color: #1e88e5; /* Calming Blue */
                color: white;
                border-radius: 5px;
                border: none;
                transition: 0.3s;
            }
            .stButton > button:hover {
                background-color: #1565c0;
            }
            .stSidebar {
                color: black;
                background-color: #f1f1f1; /* Light Sidebar */
                padding: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload your PDF Files")
        pdf_docs = st.file_uploader("Choose Files from local computer", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    uploaded_files = [pdf.name for pdf in pdf_docs]
                    st.write("Uploaded files:", ", ".join(uploaded_files))
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete ✅")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
