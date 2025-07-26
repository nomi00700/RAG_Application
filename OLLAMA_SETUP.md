# PDF RAG Chatbot with Ollama DeepSeek-R1 🚀

## 🎉 Successfully Updated with Your Local Model!

### ✅ Your Setup:
- **Ollama Server**: Running locally
- **Model**: DeepSeek-R1:1.5b (1.1 GB)
- **Application**: Updated with Ollama integration

## 🚀 Quick Start:

### 1. Ensure Ollama is Running:
```bash
# Check if Ollama is running
ollama list

# If not running, start Ollama server
ollama serve

# Verify your model
ollama run deepseek-r1:1.5b
```

### 2. Install Updated Dependencies:
```bash
cd "C:\Users\user\Desktop\Applications\RAG system"
pip install -r requirements.txt
```

### 3. Run Your RAG Chatbot:
```bash
streamlit run app.py
```

## 🎯 New Features Added:

### ✅ Ollama Integration:
- **DeepSeek-R1:1.5b** as primary model option
- **Real-time Ollama status** in sidebar
- **Local AI** - No API keys needed!
- **Fast responses** with your local model

### 🔧 Model Options:
1. **🦙 Ollama DeepSeek-R1:1.5b** (Local, Free, Fast) ⭐
2. **OpenAI GPT-3.5** (Cloud, Requires API key)
3. **OpenAI GPT-4** (Cloud, Requires API key)
4. **HuggingFace Demo** (Fallback mode)

### 📊 Complete RAG Pipeline:
```
PDF Upload → Text Chunking → Vector Embeddings → FAISS Storage → DeepSeek-R1 → Smart Answers
```

## 🎮 Usage:

1. **Start Application** - `streamlit run app.py`
2. **Check Ollama Status** - Green checkmark in sidebar
3. **Select "Ollama DeepSeek-R1:1.5b"** as model
4. **Upload PDFs** and process documents
5. **Ask questions** - Get AI answers powered by your local model!

## 🔍 Troubleshooting:

### Ollama Issues:
```bash
# If connection fails
ollama serve

# Check Ollama status
curl http://localhost:11434/api/tags

# Test your model
ollama run deepseek-r1:1.5b "Hello, how are you?"
```

### Application Issues:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Clear cache and restart
streamlit cache clear
streamlit run app.py
```

## 🌟 Advantages of Your Setup:

- **🚀 Local AI**: No internet dependency
- **💰 Free**: No API costs
- **🔒 Private**: Your data stays local
- **⚡ Fast**: Local model responses
- **🎯 Customizable**: Full control over model

## 🎊 Ready to Use!

Your PDF RAG Chatbot is now powered by **DeepSeek-R1:1.5b** running locally via Ollama!

**Upload your PDFs and start chatting with your documents using your own AI model! 🤖📚**