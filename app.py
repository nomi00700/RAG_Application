import streamlit as st
import PyPDF2
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, Ollama
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile
import os
from typing import List
import pickle

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.bot-message {
    background-color: #f1f8e9;
    border-left: 4px solid #4caf50;
}
.sidebar-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def process_pdfs(uploaded_files, embedding_choice, openai_api_key=None, chunk_size=1000, chunk_overlap=200):
    """Process uploaded PDFs and create vector store"""
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return None
    
    # Extract text from all PDFs
    documents = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                )
                documents.append(doc)
    
    if not documents:
        st.error("No text could be extracted from the uploaded PDFs.")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    with st.spinner("Splitting documents into chunks..."):
        split_docs = text_splitter.split_documents(documents)
    
    st.success(f"Created {len(split_docs)} text chunks from {len(documents)} documents.")
    
    # Create embeddings
    try:
        if embedding_choice == "OpenAI":
            if not openai_api_key:
                st.error("Please provide OpenAI API key for OpenAI embeddings.")
                return None
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:  # HuggingFace
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Create FAISS vector store
        with st.spinner("Creating vector embeddings..."):
            vector_store = FAISS.from_documents(split_docs, embeddings)
        
        st.success("Vector store created successfully!")
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def get_answer(question: str, vector_store, model_choice: str, openai_api_key=None, temperature=0.7):
    """Get answer for user question using RAG"""
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )
        
        # Create language model
        if model_choice == "Ollama DeepSeek-R1:1.5b":
            try:
                llm = Ollama(
                    model="deepseek-r1:1.5b",
                    temperature=temperature,
                    base_url="http://localhost:11434"  # Default Ollama URL
                )
                # Test connection
                test_response = llm("Hello")
                st.success("✅ Connected to Ollama DeepSeek-R1:1.5b")
            except Exception as e:
                st.error(f"❌ Ollama connection failed: {str(e)}")
                st.error("Make sure Ollama is running: 'ollama serve'")
                return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
        elif model_choice == "OpenAI GPT-3.5":
            if not openai_api_key:
                st.error("Please provide OpenAI API key.")
                return "Error: OpenAI API key required."
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
                temperature=temperature
            )
        elif model_choice == "OpenAI GPT-4":
            if not openai_api_key:
                st.error("Please provide OpenAI API key.")
                return "Error: OpenAI API key required."
            llm = ChatOpenAI(
                model_name="gpt-4",
                openai_api_key=openai_api_key,
                temperature=temperature
            )
        else:
            # For demo purposes, we'll use a simple approach
            # In real implementation, you'd use HuggingFace models
            st.warning("HuggingFace models require additional setup. Using fallback response.")
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content[:500] for doc in docs])
            return f"Based on the documents:\n\n{context}\n\n[Note: This is a simplified response. For full LLM integration, please configure OpenAI API or set up HuggingFace models.]"
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": question})
        
        return result["result"]
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# Main UI
st.markdown('<h1 class="main-header">📚 PDF-based RAG Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for OpenAI embeddings and models"
    )
    
    # Model selection
    st.subheader("Model Settings")
    embedding_choice = st.selectbox(
        "Choose Embedding Model",
        ["HuggingFace", "OpenAI"],
        help="HuggingFace is free but slower, OpenAI is faster but requires API key"
    )
    
    model_choice = st.selectbox(
        "Choose Language Model",
        ["Ollama DeepSeek-R1:1.5b", "OpenAI GPT-3.5", "OpenAI GPT-4", "HuggingFace (Demo)"],
        help="Select the model for generating answers"
    )
    
    temperature = st.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make responses more creative but less focused"
    )
    
    # Text splitting parameters
    st.subheader("Text Processing")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
    
    # Ollama Status Check
    st.subheader("🦙 Ollama Status")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("✅ Ollama Server Running")
            st.info("🤖 DeepSeek-R1:1.5b Ready")
        else:
            st.warning("⚠️ Ollama Server Issue")
    except:
        st.error("❌ Ollama Server Not Running")
        st.error("Start with: `ollama serve`")
    
    # Info section
    st.markdown("""
    <div class="sidebar-info">
    <h4>ℹ️ How it works:</h4>
    <ol>
    <li>Upload PDF files</li>
    <li>Documents are split into chunks</li>
    <li>Embeddings are created and stored in FAISS</li>
    <li>Ask questions about your documents</li>
    <li>Get AI-powered answers with DeepSeek-R1!</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to create your knowledge base"
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"• {file.name}")
    
    # Process documents button
    if st.button("🔄 Process Documents", type="primary"):
        if uploaded_files:
            vector_store = process_pdfs(
                uploaded_files, 
                embedding_choice, 
                openai_api_key,
                chunk_size,
                chunk_overlap
            )
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.documents_processed = True
        else:
            st.warning("Please upload PDF files first.")

with col2:
    st.subheader("💬 Chat Interface")
    
    # Display processing status
    if st.session_state.documents_processed:
        st.success("✅ Documents processed! You can now ask questions.")
    else:
        st.info("ℹ️ Please upload and process documents first.")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You:</strong><br>
                {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Bot:</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic discussed in the documents?",
        disabled=not st.session_state.documents_processed
    )
    
    col_ask, col_clear = st.columns([1, 1])
    
    with col_ask:
        if st.button("🚀 Ask Question", type="primary", disabled=not st.session_state.documents_processed):
            if question and st.session_state.vector_store:
                answer = get_answer(
                    question, 
                    st.session_state.vector_store, 
                    model_choice,
                    openai_api_key,
                    temperature
                )
                st.session_state.chat_history.append((question, answer))
                st.rerun()
            elif not question:
                st.warning("Please enter a question.")
    
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit, Langchain, and FAISS</p>
    <p><em>Upload your PDFs and start chatting with your documents!</em></p>
</div>
""", unsafe_allow_html=True)