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
import shutil

# Load environment variables
load_dotenv()

# Configure Google Generative AI
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# Predefined PDF file path
PRESET_PDF_PATH = "2-1-2 pseudocode and flowcharts.pdf"
file_name = os.path.splitext(os.path.basename(PRESET_PDF_PATH))[0]

def get_pdf_text(pdf_path):
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
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer these questions as if you are a teacher explaining a student.
    Answer:
    """

    # Update to Gemini 1.5
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1) 


    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

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
        - I'm a bot specialized in the document **'2-1-2 pseudocode and flowcharts'**. 😊 
        - Feel free to ask questions about the document, and I'll do my best to serve you an accurate answer!
        ---
        """
    )

    # Initialize session state for document processing flag and reindex trigger
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False
    if "reindex_triggered" not in st.session_state:
        st.session_state.reindex_triggered = False


    is_ready = os.path.exists("faiss_index")

    # --- Document Processing/Re-indexing Logic ---
    
    # Check if the FAISS index doesn't exist OR the reindex button was clicked
    if not is_ready or st.session_state.reindex_triggered:
        st.info("Starting document processing...")
        
        # 1. Clean up old index if it exists or reindex was requested
        if os.path.exists("faiss_index"):
            try:
                # Remove the directory and its contents
                import shutil
                shutil.rmtree("faiss_index")
                st.warning("Removed old FAISS index. Starting fresh!")
                is_ready = False # Reset readiness state
            except Exception as e:
                st.error(f"Could not remove old index: {e}")
                st.stop()
        
        # 2. Process the document
        with st.spinner("Brewing the document and creating the knowledge base..."):
            raw_text = get_pdf_text(PRESET_PDF_PATH)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    # Note: get_vector_store handles saving the new index
                    if get_vector_store(text_chunks):
                        st.session_state.doc_processed = True # Mark as processed
                        st.session_state.reindex_triggered = False # Reset trigger
                        st.success("Document re-indexed and brewed successfully! Ready for your questions. 👍")
                        is_ready = True # Update readiness state
                    # Note: Errors are handled within get_vector_store
            else:
                st.error("Failed to process the preset PDF.")
    
    # --- Status and Reprocessing Button ---

    if is_ready:
        st.info(f"I'm your friendly Teaching Bot, ready to help you with **{file_name}**!")
    else:
        st.error("Document not ready. Please check the PDF path or API key.")


    # Button to force reprocessing
    if st.button("🔄 Re-process Document (Use if PDF changed)"):
        # Set the session state flag and rerun the script
        st.session_state.reindex_triggered = True
        st.rerun() # Forces the script to run from the top, hitting the processing logic


    # --- User Input ---

    user_question = st.text_input(
        f"Ask a Question from the **{file_name}** document here:",
        disabled=not is_ready # Disable input if the document isn't ready
    )

    if user_question and is_ready:
        user_input(user_question)

    st.markdown(
        """
        ---
        ##### Remember, I'm still learning, so let's keep it light and fun! 😁
        """
    )

if __name__ == "__main__":
    # Ensure this import is available for directory deletion
    import shutil 
    main()









