# 📚 DocChat

DocChat is an advanced document-based chat application built with **Streamlit** and **LlamaIndex**. It utilizes a sophisticated **Hybrid RAG (Retrieval-Augmented Generation)** pipeline to provide highly accurate and context-aware answers from your uploaded documents.

## 🚀 Features

- **Multi-Format Support:** Upload and chat with PDF, TXT, MD, and DOCX files.
- **Advanced RAG Pipeline:**
    - **Multi-Query:** Automatically generates variations of your question to capture different semantic meanings.
    - **HyDE (Hypothetical Document Embeddings):** Generates "fake" answers to help find the most relevant parts of your documents.
    - **Reciprocal Rank Fusion (RRF):** Intelligently merges and re-ranks search results from multiple queries for superior precision.
- **Interactive UI:** Clean Streamlit interface with real-time processing and chat history management.
- **Flexible Configuration:** Support for multiple OpenAI models and adjustable RAG parameters via `config.yaml`.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyruscsc/docchat.git
   cd docchat
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

You can customize the application behavior in `config.yaml`:

- **LLM Options:** Define which OpenAI models are available in the UI.
- **Upload Limits:** Adjust maximum file counts and size limits.
- **RAG Parameters:** Fine-tune chunk sizes, overlap, retrieval counts, and RRF settings.

## 🏃 Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Configure API Key:** Enter your OpenAI API Key in the sidebar.
3. **Upload Documents:** Drag and drop up to 5 documents (PDF, TXT, MD, or DOCX).
4. **Process:** Click "Process Documents" to build the vector index.
5. **Chat:** Start asking questions about your content!

## 🧬 Project Structure

- `app.py`: Streamlit frontend and application logic.
- `rag_pipeline.py`: Core RAG logic implementing Multi-Query, HyDE, and RRF.
- `prompts.py`: Optimized LLM prompts for query expansion and synthesis.
- `config.yaml`: Centralized configuration for the app and RAG engine.
- `requirements.txt`: Python dependencies.

## 📝 License

[MIT](LICENSE) (or your preferred license)
