# 📚 DocChat

DocChat is an advanced document-based chat application built with **Streamlit** and **LlamaIndex**. It uses a **Hybrid RAG (Retrieval-Augmented Generation)** pipeline backed by a **ReAct agent** to provide accurate, context-aware answers from your uploaded documents — with an optional **web search fallback** for queries that go beyond your documents.

## 🚀 Features

- **Multi-Format Support:** Upload and chat with PDF, TXT, MD, and DOCX files (up to 5 at a time).
- **Hybrid RAG Pipeline:**
  - **Multi-Query:** Automatically generates query variations to capture different semantic meanings.
  - **HyDE (Hypothetical Document Embeddings):** Creates hypothetical passages to improve dense retrieval recall.
  - **Reciprocal Rank Fusion (RRF):** Merges and re-ranks results from all queries for superior precision.
- **Agentic Architecture:** A LlamaIndex `AgentWorkflow` (ReAct) orchestrates tool calls autonomously — always trying documents first, then deciding whether to escalate to web search.
- **Web Search Fallback:** When documents lack sufficient information, the agent can fall back to a [Tavily](https://tavily.com)-powered web search. Sources are listed in a collapsible panel.
- **Flexible Configuration:** RAG parameters and available models are externalized to `config.yaml` for easy tuning without touching code.

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

Customize application behaviour in `config.yaml`:

- **`app.llm_options`:** OpenAI models available in the Settings modal.
- **`app.max_file_upload_limit`** / **`app.max_upload_size_mb`:** Upload constraints.
- **`rag.*`:** Fine-tune chunk size, overlap, retrieval count, and RRF parameters.
- **`web_search.max_results`:** Number of Tavily results fetched per search.

API keys (OpenAI and Tavily) are entered at runtime in the **Settings** modal and are never written to disk.

## 🏃 Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open Settings:** Click the **Settings** button in the sidebar to select a model and enter your OpenAI API key.
3. **Upload Documents:** Drag and drop up to 5 documents (PDF, TXT, MD, or DOCX).
4. **Process:** Click **Process Documents** to build the vector index.
5. **Chat:** Ask questions about your documents. Answers are grounded in the uploaded content and labelled with their source (Documents or Web).
6. *(Optional)* **Enable Web Search:** In Settings, toggle **Web Search Fallback** and provide a [Tavily API key](https://tavily.com) to allow the agent to search the web when your documents don't have the answer.

## 🐳 Docker

**Docker Compose (recommended):**
```bash
docker compose up
```

**Plain Docker:**
```bash
docker build -t docchat .
docker run -p 8501:8501 docchat
```

The app will be available at `http://localhost:8501`.

## 🧬 Project Structure

```
docchat/
├── app.py               # Streamlit UI — sidebar, settings modal, chat interface
├── config.yaml          # Centralised configuration for app, RAG, and web search
├── requirements.txt     # Python dependencies
├── Dockerfile
├── compose.yml
├── core/                # Shared utilities (imported by tools and agents)
│   ├── config.py        # TypedDict-annotated config loader
│   ├── llm.py           # LLM factory — constructs OpenAI model and wires LlamaIndex Settings
│   └── prompts.py       # All LLM prompt templates (RAG, HyDE, Multi-Query, agent system prompt)
├── tools/               # LlamaIndex FunctionTools used by the agent
│   ├── rag.py           # Hybrid RAG pipeline (Multi-Query + HyDE + RRF) + query_documents tool
│   └── web_search.py    # Tavily web search + search_web tool
└── agents/
    └── docchat.py       # DocChatAgent — AgentWorkflow orchestration and public chat() API
```

## 📝 License

[MIT](LICENSE)
