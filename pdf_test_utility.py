import streamlit as st
import PyPDF2
from io import BytesIO

def test_pdf_content(uploaded_file):
    """Test PDF content and provide detailed analysis"""
    
    st.subheader("📋 PDF Analysis Report")
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Basic file info
        file_size = len(uploaded_file.getvalue())
        st.write(f"**File Name:** {uploaded_file.name}")
        st.write(f"**File Size:** {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Read PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        st.write(f"**Total Pages:** {num_pages}")
        
        # Test text extraction page by page
        total_text = ""
        page_results = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                char_count = len(page_text.strip())
                word_count = len(page_text.split()) if page_text else 0
                
                page_results.append({
                    'Page': page_num + 1,
                    'Characters': char_count,
                    'Words': word_count,
                    'Has_Text': char_count > 0,
                    'Preview': page_text[:100] + "..." if page_text else "No text"
                })
                
                if page_text.strip():
                    total_text += page_text + "\n"
                    
            except Exception as e:
                page_results.append({
                    'Page': page_num + 1,
                    'Characters': 0,
                    'Words': 0,
                    'Has_Text': False,
                    'Preview': f"Error: {str(e)}"
                })
        
        # Summary statistics
        st.write(f"**Total Extracted Text:** {len(total_text):,} characters")
        st.write(f"**Pages with Text:** {sum(1 for p in page_results if p['Has_Text'])}/{num_pages}")
        
        # Detailed page analysis
        if st.checkbox("Show detailed page analysis"):
            import pandas as pd
            df = pd.DataFrame(page_results)
            st.dataframe(df, use_container_width=True)
        
        # Text preview
        if total_text.strip():
            st.success("✅ Text extraction successful!")
            
            if st.checkbox("Show extracted text preview"):
                st.text_area(
                    "First 1000 characters:", 
                    total_text[:1000], 
                    height=200
                )
        else:
            st.error("❌ No text could be extracted!")
            st.warning("This appears to be a scanned PDF or image-based document.")
            
            # Suggestions
            st.info("**Possible solutions:**")
            st.info("• Use OCR software to convert to searchable PDF")
            st.info("• Try Adobe Acrobat's 'Make Text Searchable' feature")
            st.info("• Use online OCR tools")
            st.info("• Check if PDF has copy protection")
        
        # PDF metadata
        if pdf_reader.metadata:
            st.subheader("📄 PDF Metadata")
            metadata = pdf_reader.metadata
            for key, value in metadata.items():
                if value:
                    st.write(f"**{key}:** {value}")
        
        return len(total_text) > 0
        
    except Exception as e:
        st.error(f"❌ Error analyzing PDF: {str(e)}")
        return False

# Add this function to your main app.py or create as separate utility