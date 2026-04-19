import streamlit as st
import os
import tempfile
from agents.doc_chat_agent import DocChatAgent
from core.config import APP_CONFIG
from tools.rag_tool import RAGPipeline

st.set_page_config(page_title="DocChat", layout="wide")

st.title("📚 DocChat")
st.markdown("Upload up to 5 documents and chat with them using an advanced Hybrid RAG pipeline (Multi-Query + HyDE + RAG-Fusion)!")

# ── Session state initialisation ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
# Track the settings used to build the current agent so we know when to rebuild
if "_agent_web_enabled" not in st.session_state:
    st.session_state._agent_web_enabled = False
if "_agent_tavily_key" not in st.session_state:
    st.session_state._agent_tavily_key = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("OpenAI API Key", type="password")
    model_selection = st.selectbox(
        "Model Selection",
        APP_CONFIG["llm_options"]
    )

    # ── Web Search Fallback ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🌐 Web Search Fallback")
    st.caption(
        "When enabled, the agent will search the web for queries it cannot "
        "answer from your documents."
    )

    web_search_enabled = st.toggle(
        "Enable Web Search Fallback",
        value=st.session_state.web_search_enabled,
        key="web_search_toggle",
    )
    st.session_state.web_search_enabled = web_search_enabled

    tavily_api_key = ""
    if web_search_enabled:
        tavily_api_key = st.text_input(
            "Tavily API Key",
            type="password",
            help="Get a free key at https://tavily.com",
        )
        if not tavily_api_key:
            st.warning("⚠️ Enter a Tavily API key to activate web search fallback.")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.divider()
    st.header("Controls")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.reset()
        st.rerun()

    if st.button("Clear Files and Context", use_container_width=True):
        st.session_state.uploader_key += 1
        st.session_state.pipeline = None
        st.session_state.agent = None
        st.session_state.messages = []
        st.rerun()

    # ── Document Upload ───────────────────────────────────────────────────────
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        f"Upload 1 to {APP_CONFIG['max_file_upload_limit']} files (PDF, TXT, MD, DOCX)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        max_upload_size=APP_CONFIG["max_upload_size_mb"]
    )

# ── Document processing ───────────────────────────────────────────────────────
if uploaded_files:
    if len(uploaded_files) > APP_CONFIG["max_file_upload_limit"]:
        st.error(f"Please upload no more than {APP_CONFIG['max_file_upload_limit']} files.")
    elif len(uploaded_files) >= 1:
        if st.sidebar.button("Process Documents", use_container_width=True):
            if not api_key:
                st.sidebar.error("Please provide an OpenAI API key.")
            else:
                with st.spinner("Processing Documents..."):
                    try:
                        st.session_state.messages = []

                        pipeline = RAGPipeline(api_key=api_key, model_name=model_selection)

                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_file_paths = []
                            for uploaded_file in uploaded_files:
                                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(temp_file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                temp_file_paths.append(temp_file_path)

                            pipeline.process_documents(temp_file_paths)

                        st.session_state.pipeline = pipeline
                        # Reset the agent so it is rebuilt with the fresh pipeline
                        st.session_state.agent = None
                        st.sidebar.success("Documents processed successfully!")
                    except Exception as e:
                        st.sidebar.error(f"Error processing documents: {str(e)}")

# ── Build / rebuild the DocChatAgent when needed ──────────────────────────────
# The agent is (re)created when:
#   • documents have just been processed  (agent is None)
#   • the web search toggle changed
#   • the Tavily API key changed
if st.session_state.pipeline is not None:
    effective_tavily = tavily_api_key if web_search_enabled else ""
    agent_stale = (
        st.session_state.agent is None
        or st.session_state._agent_web_enabled != web_search_enabled
        or st.session_state._agent_tavily_key != effective_tavily
    )
    if agent_stale:
        try:
            st.session_state.agent = DocChatAgent(
                pipeline=st.session_state.pipeline,
                web_search_enabled=web_search_enabled and bool(tavily_api_key),
                tavily_api_key=tavily_api_key,
            )
            st.session_state._agent_web_enabled = web_search_enabled
            st.session_state._agent_tavily_key = effective_tavily
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")

# ── Chat interface ────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render the web-source banner for assistant messages sourced from web
        if message["role"] == "assistant" and message.get("web_sources"):
            _links = " · ".join(
                f"[{s['title']}]({s['url']})" for s in message["web_sources"]
            )
            st.info(
                f"🌐 **Answered via Web Search** — no sufficient information was "
                f"found in your documents.\n\n**Sources:** {_links}"
            )

# ── User Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents:"):
    if not st.session_state.agent:
        st.error("Please configure your API key and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            spinner_label = (
                "Thinking (Multi-Query + HyDE + RRF · Agent)..."
                if not (web_search_enabled and tavily_api_key)
                else "Thinking (Multi-Query + HyDE + RRF · Agent · Web Search Fallback active)..."
            )
            with st.spinner(spinner_label):
                try:
                    result = st.session_state.agent.chat(prompt)
                    answer = result["answer"]
                    source = result["source"]
                    web_sources = result.get("web_sources")

                    st.markdown(answer)

                    if source == "web" and web_sources:
                        links = " · ".join(
                            f"[{s['title']}]({s['url']})" for s in web_sources
                        )
                        st.info(
                            f"🌐 **Answered via Web Search** — no sufficient information was "
                            f"found in your documents.\n\n**Sources:** {links}"
                        )

                    # Persist the result so the banner is re-rendered on history scroll
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "web_sources": web_sources if source == "web" else None,
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
