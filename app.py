import streamlit as st
import os
import tempfile
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="DocChat", layout="wide")

st.title("📚 DocChat")
st.markdown("Upload up to 5 PDFs and chat with them using an advanced Hybrid RAG pipeline (Multi-Query + HyDE + RAG-Fusion)!")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar config
with st.sidebar:
    st.header("Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password")
    model_selection = st.selectbox(
        "Model Selection",
        ["gpt-5.4-nano", "gpt-5.4-mini"]
    )

    st.header("Controls")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("Clear Files and Context", use_container_width=True):
        st.session_state.uploader_key += 1
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.rerun()
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload 1 to 5 files (PDF, TXT, MD, DOCX)", 
        type=["pdf", "txt", "md", "docx"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        max_upload_size=10
    )

# Validate file upload count
if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload no more than 5 files.")
    elif len(uploaded_files) >= 1:
        if st.sidebar.button("Process Documents", use_container_width=True):
            if not api_key:
                st.sidebar.error("Please provide an OpenAI API key.")
            else:
                with st.spinner("Processing Documents..."):
                    try:
                        # Clear old chat on new process
                        st.session_state.messages = []
                        
                        pipeline = RAGPipeline(api_key=api_key, model_name=model_selection)
                        
                        # Save uploaded files temporarily for LlamaIndex SimpleDirectoryReader
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_file_paths = []
                            for uploaded_file in uploaded_files:
                                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                temp_file_paths.append(temp_file_path)
                                
                            pipeline.process_documents(temp_file_paths)
                            
                        st.session_state.pipeline = pipeline
                        st.sidebar.success("Documents processed successfully!")
                    except Exception as e:
                        st.sidebar.error(f"Error processing documents: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents:"):
    if not st.session_state.pipeline:
        st.error("Please configure API key and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking (Multi-Query + HyDE + RRF)..."):
                try:
                    response = st.session_state.pipeline.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
