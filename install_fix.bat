# Fix setuptools issue first, then install requirements

# Step 1: Upgrade pip and setuptools
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

# Step 2: Install requirements individually (safer approach)
pip install streamlit==1.28.1
pip install langchain==0.1.0
pip install faiss-cpu==1.7.4
pip install openai==1.3.0
pip install PyPDF2==3.0.1
pip install sentence-transformers==2.2.2
pip install tiktoken==0.5.1
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install langchain-community==0.0.13
pip install requests==2.31.0

# Alternative: If above doesn't work
pip install --no-cache-dir -r requirements.txt