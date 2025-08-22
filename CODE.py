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

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined PDF file path
PRESET_PDF_PATH = "SCS_May_Aug25_Pre_seen_14d92606a1.pdf"
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)

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
        st.write(len(response["output_text"]))
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    st.set_page_config(page_title="CIMA Assistant", page_icon="💎")
    st.header("Welcome to your CIMA Assistant! 👋")

    st.markdown(
        """
        - I'm a brand-new bot, so go easy on me. 😊 
        - Feel free to ask questions about the document, and I'll do my best to serve you some accurate answers!
        """
    )

    # Check if the FAISS index already exists
    if not os.path.exists("faiss_index"):
        # Automatically process the preset PDF on startup
        with st.spinner("Brewing the document, just a moment..."):
            raw_text = get_pdf_text(PRESET_PDF_PATH)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.success("Document brewed successfully! Ready for your questions. 👍")
            else:
                st.error("Failed to process the preset PDF.")
    else:
        st.info("I'm your friendly Annual Report Bot, ready to help you!")

    user_question = st.text_input(f"Ask a Question from the {file_name}")

    if user_question:
        user_input(user_question)

    st.markdown(
        """
        ---
        ##### Remember, I'm still learning, so let's keep it light and fun! 😁
        """
    )

if __name__ == "__main__":
    main()

