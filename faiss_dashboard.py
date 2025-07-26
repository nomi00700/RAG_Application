import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import json

def show_faiss_dashboard(vector_store):
    """Display FAISS vector store dashboard"""
    
    st.header("🔍 FAISS Vector Store Dashboard")
    
    if vector_store is None:
        st.warning("No vector store available. Process documents first!")
        return
    
    # Get FAISS index info
    index = vector_store.index
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Vectors", index.ntotal)
    
    with col2:
        st.metric("📏 Vector Dimension", index.d)
    
    with col3:
        st.metric("🗃️ Index Type", type(index).__name__)
    
    with col4:
        memory_usage = (index.ntotal * index.d * 4) / (1024 * 1024)  # MB
        st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
    
    # Document sources
    st.subheader("📚 Document Sources")
    if hasattr(vector_store, 'docstore') and vector_store.docstore.docs:
        docs_info = []
        for doc_id, doc in vector_store.docstore.docs.items():
            docs_info.append({
                "Document ID": doc_id,
                "Source": doc.metadata.get('source', 'Unknown'),
                "Content Length": len(doc.page_content),
                "Content Preview": doc.page_content[:100] + "..."
            })
        
        docs_df = pd.DataFrame(docs_info)
        st.dataframe(docs_df, use_container_width=True)
    else:
        st.info("Document store information not available")
    
    # Vector visualization (if small dataset)
    if index.ntotal <= 100:
        st.subheader("🎯 Vector Visualization (2D PCA)")
        try:
            # Get all vectors
            vectors = []
            for i in range(index.ntotal):
                vector = index.reconstruct(i)
                vectors.append(vector)
            
            vectors = np.array(vectors)
            
            # Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            
            # Create scatter plot
            fig = px.scatter(
                x=vectors_2d[:, 0], 
                y=vectors_2d[:, 1],
                title="Document Vectors in 2D Space (PCA)",
                labels={'x': 'PC1', 'y': 'PC2'},
                hover_data={'index': list(range(len(vectors_2d)))}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explained variance
            st.info(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
            
        except Exception as e:
            st.error(f"Vector visualization failed: {str(e)}")
    else:
        st.info(f"Too many vectors ({index.ntotal}) for visualization. Showing first 100 only.")

def show_search_analysis(vector_store, query=""):
    """Show search analysis for a query"""
    
    st.subheader("🔎 Search Analysis")
    
    if not query:
        query = st.text_input("Enter a test query:", "What is the main topic?")
    
    if query and vector_store:
        try:
            # Get similar documents with scores
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.0}
            )
            
            # Get embeddings for the query
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_embedding = model.encode([query])
            
            # Search FAISS directly for scores
            scores, indices = vector_store.index.search(query_embedding.astype('float32'), 10)
            
            # Create results dataframe
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    try:
                        doc_id = vector_store.index_to_docstore_id[idx]
                        doc = vector_store.docstore.docs[doc_id]
                        results.append({
                            "Rank": i + 1,
                            "Similarity Score": f"{score:.4f}",
                            "Source": doc.metadata.get('source', 'Unknown'),
                            "Content Preview": doc.page_content[:200] + "...",
                            "Content Length": len(doc.page_content)
                        })
                    except:
                        results.append({
                            "Rank": i + 1,
                            "Similarity Score": f"{score:.4f}",
                            "Source": "Error loading",
                            "Content Preview": "Could not load content",
                            "Content Length": 0
                        })
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Score distribution
                scores_only = [float(r["Similarity Score"]) for r in results]
                fig = px.bar(
                    x=[f"Rank {r['Rank']}" for r in results],
                    y=scores_only,
                    title="Similarity Scores Distribution",
                    labels={'x': 'Document Rank', 'y': 'Similarity Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No similar documents found!")
                
        except Exception as e:
            st.error(f"Search analysis failed: {str(e)}")

# Add this to your main app.py in the sidebar or as a new tab
def add_faiss_debug_tab():
    """Add FAISS debugging tab to Streamlit app"""
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 FAISS Dashboard", "🔎 Search Analysis"])
    
    with tab1:
        # Your existing chat interface code here
        pass
    
    with tab2:
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            show_faiss_dashboard(st.session_state.vector_store)
        else:
            st.info("Process documents first to see FAISS dashboard")
    
    with tab3:
        if 'vector_store' in st.session_state and st.session_state.vector_store:
            show_search_analysis(st.session_state.vector_store)
        else:
            st.info("Process documents first to analyze search")

# Usage example:
"""
To add this to your app.py, replace your chat interface section with:

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    # Your existing upload section
    pass

with col2:
    add_faiss_debug_tab()  # This replaces your current chat interface
"""