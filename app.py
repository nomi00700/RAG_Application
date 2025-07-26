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
    """Extract text from uploaded PDF file with multiple methods"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Method 1: Standard extraction
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Check if page has text
                text += page_text + "\n"
            else:
                st.warning(f"Page {page_num + 1} appears to be empty or image-based")
        
        # Check if any text was extracted
        if not text.strip():
            st.error("❌ No text could be extracted from this PDF")
            st.info("💡 This might be a scanned PDF or image-based. Try:")
            st.info("• Converting to text-based PDF")
            st.info("• Using OCR tools")
            st.info("• Uploading a different PDF")
            return ""
        
        # Validate extracted text
        if len(text.strip()) < 50:
            st.warning(f"⚠️ Very little text extracted ({len(text)} characters)")
            st.info("This PDF might have formatting issues")
        
        st.success(f"✅ Extracted {len(text)} characters from PDF")
        return text
        
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        st.info("Try uploading a different PDF file")
        return ""

def process_pdfs(uploaded_files, embedding_choice, openai_api_key=None, chunk_size=1000, chunk_overlap=200):
    """Process uploaded PDFs and create vector store with better error handling"""
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return None
    
    # Extract text from all PDFs
    documents = []
    total_text_length = 0
    
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            
            if text and len(text.strip()) > 10:  # Minimum text threshold
                # Create Document object with metadata
                doc = Document(
                    page_content=text.strip(),
                    metadata={"source": uploaded_file.name, "length": len(text)}
                )
                documents.append(doc)
                total_text_length += len(text)
                st.success(f"✅ Successfully processed {uploaded_file.name}")
            else:
                st.error(f"❌ Failed to extract text from {uploaded_file.name}")
    
    if not documents:
        st.error("❌ No text could be extracted from any of the uploaded PDFs.")
        st.info("💡 **Possible solutions:**")
        st.info("• Try different PDF files")
        st.info("• Ensure PDFs contain selectable text (not scanned images)")
        st.info("• Convert image PDFs to text using OCR tools")
        return None
    
    st.info(f"📊 **Processing Summary:** {len(documents)} documents, {total_text_length:,} total characters")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
    )
    
    with st.spinner("Splitting documents into chunks..."):
        split_docs = text_splitter.split_documents(documents)
    
    if not split_docs:
        st.error("❌ No chunks were created from the documents")
        st.info("Try reducing the chunk size or uploading different PDFs")
        return None
    
    st.success(f"✅ Created {len(split_docs)} text chunks from {len(documents)} documents.")
    
    # Show chunk statistics
    chunk_lengths = [len(chunk.page_content) for chunk in split_docs]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    st.info(f"📈 **Chunk Stats:** Avg length: {avg_length:.0f} chars, Range: {min(chunk_lengths)}-{max(chunk_lengths)} chars")
    
    # Create embeddings
    try:
        if "🔑 OpenAI" in embedding_choice:
            if not embedding_api_key:
                st.error("🔑 OpenAI API key required for OpenAI embeddings.")
                return None
            embeddings = OpenAIEmbeddings(openai_api_key=embedding_api_key)
        elif "🔑 Cohere" in embedding_choice:
            if not embedding_api_key:
                st.error("🔑 Cohere API key required for Cohere embeddings.")
                return None
            # Add Cohere embeddings implementation here
            st.error("Cohere embeddings not yet implemented. Please use HuggingFace or OpenAI.")
            return None
        elif "🔑 HuggingFace Hub" in embedding_choice:
            if not embedding_api_key:
                st.error("🔑 HuggingFace API key required for Hub models.")
                return None
            # Add HuggingFace Hub implementation here
            st.error("HuggingFace Hub embeddings not yet implemented. Please use local HuggingFace.")
            return None
        else:  # Default to local HuggingFace
            with st.spinner("Loading HuggingFace model..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        
        # Create FAISS vector store
        with st.spinner(f"Creating vector embeddings for {len(split_docs)} chunks..."):
            vector_store = FAISS.from_documents(split_docs, embeddings)
        
        st.success("✅ Vector store created successfully!")
        st.balloons()  # Celebration!
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Error creating embeddings: {str(e)}")
        st.info("🛠️ **Troubleshooting:**")
        st.info("• Check internet connection for HuggingFace models")
        st.info("• Try reducing chunk size")
        st.info("• Verify OpenAI API key if using OpenAI embeddings")
        return None

def get_answer(question: str, vector_store, model_choice: str, api_key=None, temperature=0.7):
    """Get answer for user question using RAG with dynamic model selection"""
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
        )
        
        # Create language model based on selection
        if "🦙" in model_choice:  # Ollama models
            try:
                # Extract actual model name
                model_name = model_choice.split("🦙 ")[1].split(" (")[0]
                llm = Ollama(
                    model=model_name,
                    temperature=temperature,
                    base_url="http://localhost:11434"
                )
                # Test connection
                test_response = llm("Hello")
                st.success(f"✅ Connected to Ollama {model_name}")
            except Exception as e:
                st.error(f"❌ Ollama connection failed: {str(e)}")
                st.error("Make sure Ollama is running: 'ollama serve'")
                return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
                
        elif "🔑 OpenAI GPT-3.5" in model_choice:
            if not api_key:
                st.error("🔑 OpenAI API key required.")
                return "Error: OpenAI API key required."
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=api_key,
                temperature=temperature
            )
        elif "🔑 OpenAI GPT-4" in model_choice:
            if not api_key:
                st.error("🔑 OpenAI API key required.")
                return "Error: OpenAI API key required."
            llm = ChatOpenAI(
                model_name="gpt-4",
                openai_api_key=api_key,
                temperature=temperature
            )
        elif "🔑 Anthropic Claude" in model_choice:
            if not api_key:
                st.error("🔑 Anthropic API key required.")
                return "Error: Anthropic API key required."
            st.error("Anthropic Claude not yet implemented. Please use Ollama or OpenAI.")
            return "Error: Model not implemented yet."
        elif "🔑 Cohere" in model_choice:
            if not api_key:
                st.error("🔑 Cohere API key required.")
                return "Error: Cohere API key required."
            st.error("Cohere Command not yet implemented. Please use Ollama or OpenAI.")
            return "Error: Model not implemented yet."
        else:
            # Fallback mode
            st.warning("⚠️ Using fallback mode - limited functionality.")
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content[:500] for doc in docs])
            return f"Based on the documents:\n\n{context}\n\n[Note: This is a simplified response. Please configure a proper LLM model.]"
        
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
        st.error(f"❌ Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# Main UI
st.markdown('<h1 class="main-header">📚 PDF-based RAG Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Auto-detect available models
    def detect_available_models():
        """Detect what models are available locally and via APIs"""
        available = {
            'embeddings': [],
            'llms': [],
            'ollama_models': [],
            'ollama_running': False
        }
        
        # Always available (local)
        available['embeddings'].append("🤗 HuggingFace (Local - Free)")
        
        # Check Ollama status and models
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                available['ollama_running'] = True
                ollama_data = response.json()
                if 'models' in ollama_data:
                    for model in ollama_data['models']:
                        model_name = model['name']
                        available['ollama_models'].append(model_name)
                        available['llms'].append(f"🦙 {model_name} (Local - Free)")
        except:
            pass
        
        # Cloud options
        available['embeddings'].extend([
            "🔑 OpenAI (Cloud - API Key Required)",
            "🔑 Cohere (Cloud - API Key Required)",
            "🔑 HuggingFace Hub (Cloud - API Key Required)"
        ])
        
        available['llms'].extend([
            "🔑 OpenAI GPT-3.5 (Cloud - API Key Required)",
            "🔑 OpenAI GPT-4 (Cloud - API Key Required)",
            "🔑 Anthropic Claude (Cloud - API Key Required)",
            "🔑 Cohere Command (Cloud - API Key Required)"
        ])
        
        return available
    
    # Get available models
    available_models = detect_available_models()
    
    # Ollama Status Display
    if available_models['ollama_running']:
        st.success("🦙 Ollama Server: Running")
        if available_models['ollama_models']:
            st.info(f"📋 Local Models: {len(available_models['ollama_models'])}")
            with st.expander("View Local Models"):
                for model in available_models['ollama_models']:
                    st.write(f"• {model}")
        else:
            st.warning("⚠️ No models found in Ollama")
    else:
        st.error("❌ Ollama Server: Not Running")
        st.info("💡 Start with: `ollama serve`")
    
    st.divider()
    
    # === EMBEDDING MODEL CONFIGURATION ===
    st.subheader("🧠 Embedding Model")
    st.caption("Converts text to vectors for similarity search")
    
    embedding_choice = st.selectbox(
        "Choose Embedding Model:",
        available_models['embeddings'],
        key="embedding_model"
    )
    
    # Dynamic API key input for embeddings
    embedding_api_key = None
    if "🔑" in embedding_choice:
        if "OpenAI" in embedding_choice:
            embedding_api_key = st.text_input(
                "🔑 OpenAI API Key (for embeddings):",
                type="password",
                help="Required for OpenAI embeddings - get from platform.openai.com",
                key="openai_embedding_key"
            )
        elif "Cohere" in embedding_choice:
            embedding_api_key = st.text_input(
                "🔑 Cohere API Key:",
                type="password",
                help="Required for Cohere embeddings - get from cohere.ai",
                key="cohere_embedding_key"
            )
        elif "HuggingFace Hub" in embedding_choice:
            embedding_api_key = st.text_input(
                "🔑 HuggingFace API Key:",
                type="password",
                help="Required for HuggingFace Hub models - get from huggingface.co",
                key="hf_embedding_key"
            )
    
    st.divider()
    
    # === LANGUAGE MODEL CONFIGURATION ===
    st.subheader("🤖 Language Model")
    st.caption("Generates answers from retrieved context")
    
    llm_choice = st.selectbox(
        "Choose Language Model:",
        available_models['llms'],
        key="language_model"
    )
    
    # Dynamic API key input for LLMs
    llm_api_key = None
    if "🔑" in llm_choice:
        if "OpenAI" in llm_choice:
            llm_api_key = st.text_input(
                "🔑 OpenAI API Key (for LLM):",
                type="password",
                help="Required for OpenAI models - get from platform.openai.com",
                key="openai_llm_key"
            )
        elif "Anthropic" in llm_choice:
            llm_api_key = st.text_input(
                "🔑 Anthropic API Key:",
                type="password",
                help="Required for Claude models - get from console.anthropic.com",
                key="anthropic_key"
            )
        elif "Cohere" in llm_choice:
            llm_api_key = st.text_input(
                "🔑 Cohere API Key:",
                type="password",
                help="Required for Cohere models - get from cohere.ai",
                key="cohere_llm_key"
            )
    
    st.divider()
    
    # === MODEL PARAMETERS ===
    st.subheader("⚙️ Model Parameters")
    
    temperature = st.slider(
        "🌡️ Response Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    # Text splitting parameters
    st.subheader("📄 Text Processing")
    chunk_size = st.slider("📏 Chunk Size:", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("🔗 Chunk Overlap:", 50, 500, 200, 50)
    
    st.divider()
    
    # === CONFIGURATION SUMMARY ===
    st.subheader("📋 Current Setup")
    
    # Clean model names for display
    def clean_model_name(model_name):
        return model_name.replace("🤗 ", "").replace("🦙 ", "").replace("🔑 ", "").split(" (")[0]
    
    embedding_display = clean_model_name(embedding_choice)
    llm_display = clean_model_name(llm_choice)
    
    st.info(f"""
    **🧠 Embeddings:** {embedding_display}
    **🤖 Language Model:** {llm_display}
    **🌡️ Temperature:** {temperature}
    **📏 Chunk Size:** {chunk_size}
    """)
    
    # Cost estimation
    is_free_setup = "Local - Free" in embedding_choice and "Local - Free" in llm_choice
    if is_free_setup:
        st.success("💰 **Cost:** Completely Free!")
    else:
        st.warning("💳 **Cost:** API charges may apply")
    
    st.divider()
    
    # === HELP SECTION ===
    with st.expander("ℹ️ How RAG Works"):
        st.markdown("""
        **Step 1:** 📄 Upload PDFs → Extract text
        **Step 2:** ✂️ Split into chunks → Create embeddings
        **Step 3:** 🗃️ Store in FAISS vector database
        **Step 4:** ❓ Ask question → Find similar chunks
        **Step 5:** 🤖 LLM generates answer from context
        """)
    
    # Store selections in session state
    st.session_state.embedding_choice = embedding_choice
    st.session_state.llm_choice = llm_choice
    st.session_state.embedding_api_key = embedding_api_key
    st.session_state.llm_api_key = llm_api_key
    st.session_state.temperature = temperature
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap

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
            # Get current configuration
            embedding_choice = st.session_state.get('embedding_choice', '')
            embedding_api_key = st.session_state.get('embedding_api_key', None)
            chunk_size = st.session_state.get('chunk_size', 1000)
            chunk_overlap = st.session_state.get('chunk_overlap', 200)
            
            vector_store = process_pdfs(
                uploaded_files, 
                embedding_choice, 
                embedding_api_key,
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
                # Get current configuration
                llm_choice = st.session_state.get('llm_choice', '')
                llm_api_key = st.session_state.get('llm_api_key', None)
                temperature = st.session_state.get('temperature', 0.7)
                
                answer = get_answer(
                    question, 
                    st.session_state.vector_store, 
                    llm_choice,
                    llm_api_key,
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