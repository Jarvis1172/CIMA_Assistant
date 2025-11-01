import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- CONFIGURATION AND INITIAL SETUP ---

# Load environment variables (for local development)
load_dotenv()

# Configure Google Generative AI
# Uses st.secrets for Streamlit Cloud deployment or os.getenv for local execution
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure GOOGLE_API_KEY is set in st.secrets or .env.")

# Predefined PDF file path
PRESET_PDF_PATH = "2-1-2 pseudocode and flowcharts.pdf"
# Initialize file name for display
file_name = os.path.splitext(os.path.basename(PRESET_PDF_PATH))[0]

# --- CORE PROCESSING FUNCTIONS ---

def get_pdf_text(pdf_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def get_text_chunks(text):
    """Splits text into chunks for embeddings."""
    try:
        # NOTE: Chunk size is large (10000) for better context in RAG
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Use st.cache_resource to ensure this expensive operation runs only once per unique resource
@st.cache_resource
def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks and saves it locally."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

@st.cache_resource
def get_conversational_chain():
    """Initializes and caches the Gemini QA chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.
    The answer must be provided as if you are a teacher explaining the concepts to a student.

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    # Using the fast and capable Gemini 2.5 Flash model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7) 

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index from disk. Cached for quick access."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # allow_dangerous_deserialization=True is necessary for FAISS.load_local()
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Could not load vector store: {e}")
        return None

# --- STREAMLIT UI AND LOGIC FUNCTIONS ---

def process_pdf_and_create_vector_store(pdf_path, uploaded_file_name):
    """Processes a PDF, creates a vector store, and updates the session state."""
    
    # 1. Get raw text
    with st.spinner(f"Reading '{uploaded_file_name}'..."):
        raw_text = get_pdf_text(pdf_path)
    
    if not raw_text:
        return None, None

    # 2. Get text chunks
    with st.spinner("Splitting text into chunks..."):
        text_chunks = get_text_chunks(raw_text)
    
    if not text_chunks:
        return None, None

    # 3. Create vector store (Caches the new resource and saves it to 'faiss_index')
    with st.spinner("Creating embeddings and vector store..."):
        get_vector_store(text_chunks)
        
    st.success(f"Document '{uploaded_file_name}' processed successfully! Ready for questions. 👍")
    
    # Return the file name information for display
    return uploaded_file_name, os.path.splitext(uploaded_file_name)[0]


def user_input(user_question):
    """Handles the user's question, performs similarity search, and calls the QA chain."""
    try:
        new_db = load_faiss_index() # Load the cached index
        
        if new_db is None:
            st.error("Vector store failed to load. Cannot answer questions.")
            return

        # Retrieve relevant documents
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain() # Get the cached chain

        # Run the QA chain
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    st.set_page_config(page_title="Teaching Assistant", page_icon="🎓")
    st.header("Welcome to your Teaching Assistant! 👋")

    st.markdown(
        """
        I'm a bot here to help you learn! I can answer questions based on the document you provide or the preset document.
        """
    )
    
    # Initialize session state for tracking the current document
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = os.path.splitext(os.path.basename(PRESET_PDF_PATH))[0]

    
    # --- 1. File Uploader Section ---
    uploaded_file = st.file_uploader("Or, upload a **new PDF** to chat with:", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location for PyPDF2 to read
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.get_buffer())
        
        # Clear all cached resources so the new vector store can be built
        st.cache_resource.clear() 
        
        # Process the new file
        processed_name, processed_display_name = process_pdf_and_create_vector_store(temp_file_path, uploaded_file.name)
        
        if processed_name:
            st.session_state.current_file_name = processed_display_name
            # Remove the temporary file after processing
            os.remove(temp_file_path)
    
    # --- 2. Initial Preset Processing ---
    # Only run if no new file was uploaded AND the index doesn't exist
    elif not os.path.exists("faiss_index"):
        st.info(f"Automatically loading the preset document: **{PRESET_PDF_PATH}**")
        processed_name, processed_display_name = process_pdf_and_create_vector_store(PRESET_PDF_PATH, PRESET_PDF_PATH)
        if processed_name:
             st.session_state.current_file_name = processed_display_name
    
    # Display current status
    else:
        st.info(f"I'm ready to help you with the **{st.session_state.current_file_name}** document!")


    # --- 3. Q&A Section ---
    user_question = st.text_input(f"Ask a Question from the **{st.session_state.current_file_name}** document")

    if user_question and os.path.exists("faiss_index"):
        user_input(user_question)
    elif user_question:
        st.warning("Please upload a document first or wait for the preset document to load before asking a question.")

    st.markdown(
        """
        ---
        ##### Remember, I'm still learning, so let's keep it light and fun! 😁
        """
    )

if __name__ == "__main__":
    main()
