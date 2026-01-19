"""
Streamlit web interface for multi-document summarization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time
import json

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from summarizer.models import MultiDocumentSummarizer
from summarizer.evaluation import SummarizationEvaluator
from data.database import MockDatabase


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multi-Document Summarization",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📄 Multi-Document Summarization</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'database' not in st.session_state:
        st.session_state.database = MockDatabase()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = SummarizationEvaluator()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "BART Large CNN": "bart-large-cnn",
            "T5 Small": "t5-small",
            "Pegasus XSum": "pegasus-xsum",
            "Extractive": "extractive"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        model_name = model_options[selected_model]
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    st.session_state.summarizer = MultiDocumentSummarizer(model_name)
                    st.success(f"✅ {selected_model} loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        
        # Model info
        if st.session_state.summarizer:
            st.success("✅ Model loaded")
            st.info(f"Model: {st.session_state.summarizer.model_config.name}")
        else:
            st.warning("⚠️ No model loaded")
        
        st.divider()
        
        # Database stats
        st.header("📊 Database Stats")
        stats = st.session_state.database.get_document_stats()
        st.metric("Total Documents", stats['total_documents'])
        
        # Category breakdown
        if stats['documents_by_category']:
            st.subheader("Documents by Category")
            for category, count in stats['documents_by_category'].items():
                st.write(f"• {category}: {count}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summarize", "📚 Database", "📊 Evaluation", "ℹ️ About"])
    
    with tab1:
        st.header("Document Summarization")
        
        if not st.session_state.summarizer:
            st.warning("Please load a model from the sidebar first.")
            return
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Select from Database", "Upload Files"],
            horizontal=True
        )
        
        documents = []
        
        if input_method == "Manual Input":
            st.subheader("Enter Documents")
            
            num_docs = st.number_input("Number of documents", min_value=1, max_value=10, value=2)
            
            for i in range(num_docs):
                with st.expander(f"Document {i+1}"):
                    title = st.text_input(f"Title {i+1}", key=f"title_{i}")
                    content = st.text_area(f"Content {i+1}", height=200, key=f"content_{i}")
                    if content.strip():
                        documents.append(content)
        
        elif input_method == "Select from Database":
            st.subheader("Select Documents from Database")
            
            # Category filter
            categories = st.session_state.database.get_categories()
            category_names = [cat['name'] for cat in categories]
            selected_category = st.selectbox("Filter by Category", ["All"] + category_names)
            
            # Get documents
            if selected_category == "All":
                db_docs = st.session_state.database.get_documents(limit=20)
            else:
                db_docs = st.session_state.database.get_documents(category=selected_category, limit=20)
            
            if db_docs:
                st.write(f"Found {len(db_docs)} documents")
                
                # Document selection
                selected_docs = st.multiselect(
                    "Select documents to summarize:",
                    options=[f"{doc['title']} ({doc['category']})" for doc in db_docs],
                    default=[]
                )
                
                # Extract selected document contents
                for selected in selected_docs:
                    doc_title = selected.split(" (")[0]
                    for doc in db_docs:
                        if doc['title'] == doc_title:
                            documents.append(doc['content'])
                            break
            else:
                st.warning("No documents found in the database.")
        
        elif input_method == "Upload Files":
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['txt', 'md']
            )
            
            for file in uploaded_files:
                content = str(file.read(), "utf-8")
                documents.append(content)
                st.write(f"✅ Uploaded: {file.name}")
        
        # Summarization
        if documents and st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                start_time = time.time()
                
                try:
                    # Generate summary
                    result = st.session_state.summarizer.summarize_with_metadata(documents)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Display results
                    st.success("✅ Summary generated successfully!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Documents", result['num_documents'])
                    
                    with col2:
                        st.metric("Total Length", f"{result['total_length']:,} chars")
                    
                    with col3:
                        st.metric("Summary Length", f"{result['summary_length']:,} chars")
                    
                    with col4:
                        st.metric("Compression Ratio", f"{result['compression_ratio']:.2%}")
                    
                    # Summary text
                    st.subheader("📄 Generated Summary")
                    st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
                    
                    # Processing info
                    st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
    
    with tab2:
        st.header("Document Database")
        
        # Search functionality
        st.subheader("Search Documents")
        search_query = st.text_input("Enter search query:")
        
        if search_query:
            search_results = st.session_state.database.search_documents(search_query)
            st.write(f"Found {len(search_results)} results")
            
            for doc in search_results:
                with st.expander(f"{doc['title']} ({doc['category']})"):
                    st.write(f"**Source:** {doc['source']}")
                    st.write(f"**Created:** {doc['created_at']}")
                    st.write("**Content:**")
                    st.write(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
        
        # Add new document
        st.subheader("Add New Document")
        
        with st.form("add_document"):
            new_title = st.text_input("Title")
            new_content = st.text_area("Content", height=200)
            new_category = st.selectbox("Category", [cat['name'] for cat in categories])
            new_source = st.text_input("Source", value="User Input")
            
            if st.form_submit_button("Add Document"):
                if new_title and new_content:
                    doc_id = st.session_state.database.add_document(
                        new_title, new_content, new_category, new_source
                    )
                    st.success(f"✅ Document added with ID: {doc_id}")
                else:
                    st.error("Please fill in all required fields.")
    
    with tab3:
        st.header("Evaluation Metrics")
        
        st.write("This section allows you to evaluate the quality of generated summaries using various metrics.")
        
        # Evaluation form
        with st.form("evaluation"):
            st.subheader("Compare Summary with Reference")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Generated Summary**")
                generated_summary = st.text_area("Generated Summary", height=150)
            
            with col2:
                st.write("**Reference Summary**")
                reference_summary = st.text_area("Reference Summary", height=150)
            
            if st.form_submit_button("📊 Evaluate"):
                if generated_summary and reference_summary:
                    with st.spinner("Evaluating..."):
                        try:
                            results = st.session_state.evaluator.evaluate_comprehensive(
                                [generated_summary], [reference_summary]
                            )
                            
                            # Display results
                            st.success("✅ Evaluation completed!")
                            
                            # ROUGE scores
                            st.subheader("ROUGE Scores")
                            rouge_data = []
                            for metric in ['rouge1', 'rouge2', 'rougeL']:
                                scores = results['rouge'][metric]
                                rouge_data.append({
                                    'Metric': metric.upper(),
                                    'Precision': scores['precision'],
                                    'Recall': scores['recall'],
                                    'F1-Score': scores['fmeasure']
                                })
                            
                            rouge_df = pd.DataFrame(rouge_data)
                            st.dataframe(rouge_df, use_container_width=True)
                            
                            # Other metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("BERTScore F1", f"{results['bertscore']['f1']:.4f}")
                            
                            with col2:
                                st.metric("BLEU Score", f"{results['bleu']['bleu']:.4f}")
                            
                            with col3:
                                st.metric("Coverage", f"{results['coverage']['coverage']:.4f}")
                            
                            with col4:
                                st.metric("Overall Score", f"{results['overall_score']:.4f}")
                            
                            # Visualization
                            st.subheader("Score Visualization")
                            
                            metrics_data = {
                                'ROUGE-1 F1': results['rouge']['rouge1']['fmeasure'],
                                'ROUGE-2 F1': results['rouge']['rouge2']['fmeasure'],
                                'ROUGE-L F1': results['rouge']['rougeL']['fmeasure'],
                                'BERTScore F1': results['bertscore']['f1'],
                                'BLEU': results['bleu']['bleu'],
                                'Coverage': results['coverage']['coverage']
                            }
                            
                            fig = go.Figure(data=go.Scatterpolar(
                                r=list(metrics_data.values()),
                                theta=list(metrics_data.keys()),
                                fill='toself',
                                name='Summary Quality'
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=True,
                                title="Summary Quality Metrics"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Evaluation error: {str(e)}")
                else:
                    st.error("Please provide both generated and reference summaries.")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ## Multi-Document Summarization System
        
        This application provides a comprehensive solution for summarizing multiple documents using state-of-the-art NLP models and techniques.
        
        ### Features
        
        - **Multiple Models**: Support for T5, BART, Pegasus, and extractive summarization
        - **Advanced Preprocessing**: Document chunking, deduplication, and ranking
        - **Evaluation Metrics**: ROUGE, BERTScore, BLEU, and coverage analysis
        - **Database Integration**: Sample documents and document management
        - **Web Interface**: User-friendly Streamlit interface
        
        ### Models Supported
        
        1. **BART Large CNN**: Facebook's model optimized for summarization
        2. **T5 Small**: Google's Text-to-Text Transfer Transformer
        3. **Pegasus XSum**: Google's model trained on extreme summarization
        4. **Extractive**: Graph-based extractive summarization using TextRank
        
        ### Usage Tips
        
        - Load a model from the sidebar before generating summaries
        - Use the database tab to explore sample documents
        - Evaluate summaries using the evaluation tab
        - Try different models to compare results
        
        ### Technical Details
        
        - Built with Python and Streamlit
        - Uses Hugging Face Transformers library
        - Implements advanced preprocessing techniques
        - Supports both abstractive and extractive summarization
        """)
        
        # System info
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Python Version:**", sys.version)
            st.write("**Streamlit Version:**", st.__version__)
        
        with col2:
            st.write("**Available Models:**", len(model_options))
            st.write("**Database Documents:**", stats['total_documents'])


if __name__ == "__main__":
    main()
