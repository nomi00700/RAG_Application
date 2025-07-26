#!/usr/bin/env python3
# test_installation.py - Test if all packages are working

print("Testing package imports...")

try:
    import streamlit as st
    print("✅ Streamlit - OK")
except ImportError as e:
    print(f"❌ Streamlit - Failed: {e}")

try:
    import langchain
    print("✅ Langchain - OK")
except ImportError as e:
    print(f"❌ Langchain - Failed: {e}")

try:
    import faiss
    print("✅ FAISS - OK")
except ImportError as e:
    print(f"❌ FAISS - Failed: {e}")

try:
    import PyPDF2
    print("✅ PyPDF2 - OK")
except ImportError as e:
    print(f"❌ PyPDF2 - Failed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers - OK")
except ImportError as e:
    print(f"❌ Sentence Transformers - Failed: {e}")

try:
    from langchain.llms import Ollama
    print("✅ Ollama (Langchain) - OK")
except ImportError as e:
    print(f"❌ Ollama - Failed: {e}")

print("\n🚀 Ready to test Ollama connection...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code == 200:
        print("✅ Ollama Server - Running")
    else:
        print("⚠️ Ollama Server - Issue")
except:
    print("❌ Ollama Server - Not Running")
    print("Start with: ollama serve")

print("\n🎉 Installation test complete!")