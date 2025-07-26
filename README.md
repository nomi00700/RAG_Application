# PDF RAG Chatbot with Ollama DeepSeek-R1

A sophisticated PDF-based Retrieval Augmented Generation (RAG) chatbot built with Streamlit, Langchain, and FAISS, powered by local Ollama DeepSeek-R1:1.5b model.

## 🚀 Features

- **📚 PDF Processing**: Upload and process multiple PDF files
- **🤖 Local AI**: Powered by Ollama DeepSeek-R1:1.5b (no API keys needed!)
- **🔍 Smart Search**: FAISS vector store for efficient similarity search
- **💬 Interactive Chat**: Beautiful chat interface with history
- **⚙️ Configurable**: Adjustable chunk sizes, temperature, and model selection
- **🎨 Modern UI**: Custom styled Streamlit interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama DeepSeek-R1:1.5b (Local)
- **Embeddings**: HuggingFace Sentence Transformers / OpenAI
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF2
- **Framework**: Langchain

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running
- DeepSeek-R1:1.5b model downloaded

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Ollama
```bash
# Start Ollama server
ollama serve

# Verify DeepSeek-R1 model
ollama run deepseek-r1:1.5b
```

### 4. Run Application
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Upload PDFs**: Select and upload your PDF documents
2. **Process Documents**: Click "Process Documents" to create embeddings
3. **Choose Model**: Select "Ollama DeepSeek-R1:1.5b" for local AI
4. **Ask Questions**: Chat with your documents using natural language
5. **Get Answers**: Receive intelligent responses based on document content

## 🔧 Configuration Options

### Model Selection
- **🦙 Ollama DeepSeek-R1:1.5b** (Local, Free, Recommended)
- **OpenAI GPT-3.5/GPT-4** (Cloud, Requires API key)
- **HuggingFace** (Demo mode)

### Embedding Options
- **HuggingFace Sentence Transformers** (Free)
- **OpenAI Embeddings** (Requires API key)

### Parameters
- **Chunk Size**: 500-2000 tokens
- **Chunk Overlap**: 50-500 tokens
- **Temperature**: 0.0-1.0 (response creativity)

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── requirements_compatible.txt  # Compatible versions
├── README.md             # Project documentation
├── OLLAMA_SETUP.md       # Ollama setup guide
├── QUICK_START.md        # Quick start instructions
├── test_installation.py  # Installation test script
├── install_fix.bat       # Windows installation fix
└── .gitignore           # Git ignore file
```

## 🔑 Environment Variables (Optional)

Create a `.env` file for API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   ollama serve
   ```

2. **Package Installation Issues**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements_compatible.txt
   ```

3. **PDF Reading Errors**
   - Ensure PDFs contain extractable text (not scanned images)
   - Try different PDF files

### Performance Tips

- **Large Documents**: Reduce chunk size for better accuracy
- **Better Responses**: Use OpenAI embeddings if available
- **Memory Issues**: Process fewer documents at once

## 📊 Example Use Cases

- **📖 Research Papers**: Academic literature analysis
- **📋 Legal Documents**: Contract and legal text analysis
- **📘 Technical Manuals**: Documentation Q&A
- **📈 Business Reports**: Corporate document insights
- **🎓 Educational Content**: Interactive learning materials

## 🚀 Advanced Features

### Planned Enhancements
- [ ] Support for DOCX, TXT formats
- [ ] Document similarity analysis
- [ ] Citation tracking
- [ ] Multi-language support
- [ ] Advanced search filters
- [ ] Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM infrastructure
- **DeepSeek** for the powerful R1 model
- **Langchain** for RAG framework
- **Streamlit** for the beautiful UI
- **FAISS** for efficient vector search

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#🐛-troubleshooting) section
2. Review [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed setup
3. Open an issue on GitHub
4. Check Ollama documentation

## ⭐ Star this repo if you found it helpful!

**Happy chatting with your documents! 🤖📚**