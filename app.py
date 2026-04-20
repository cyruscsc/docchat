import streamlit as st
import os
import tempfile
from agents.doc_chat_agent import DocChatAgent
from core.config import APP_CONFIG
from tools.rag_tool import RAGPipeline

st.set_page_config(page_title="DocChat", layout="wide")

st.title("DocChat")
st.markdown(
    "Upload up to 5 documents and chat with them using an advanced Hybrid RAG pipeline with optional web search fallback!"
)

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
# Persist settings across reruns
if "_api_key" not in st.session_state:
    st.session_state._api_key = ""
if "_model_selection" not in st.session_state:
    st.session_state._model_selection = APP_CONFIG["llm_options"][0]
if "_tavily_key" not in st.session_state:
    st.session_state._tavily_key = ""

# ── Settings Modal ────────────────────────────────────────────────────────────
@st.dialog("Settings")
def settings_dialog():
    st.subheader("Model")
    model_sel = st.selectbox(
        "Model Selection",
        APP_CONFIG["llm_options"],
        index=APP_CONFIG["llm_options"].index(st.session_state._model_selection)
        if st.session_state._model_selection in APP_CONFIG["llm_options"]
        else 0,
        help="The core model that powers the agent.",
        key="dialog_model_selection",
    )
    api_key_val = st.text_input(
        "OpenAI API Key",
        value=st.session_state._api_key,
        type="password",
        help="Get your API key from https://openai.com/",
        key="dialog_api_key",
    )
    if not api_key_val:
        st.warning("Enter an OpenAI API key to activate the agent.")

    st.divider()
    st.subheader("Web Search Fallback")
    web_enabled = st.toggle(
        "Web Search Fallback",
        value=st.session_state.web_search_enabled,
        help="When enabled, the agent will search the web for queries it cannot answer from your documents.",
        key="dialog_web_toggle",
    )
    tavily_key_val = ""
    if web_enabled:
        tavily_key_val = st.text_input(
            "Tavily API Key",
            value=st.session_state._tavily_key,
            type="password",
            help="Get a free key at https://tavily.com",
            key="dialog_tavily_key",
        )
        if not tavily_key_val:
            st.warning("Enter a Tavily API key to activate web search fallback.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", use_container_width=True, disabled=not api_key_val or (web_enabled and not tavily_key_val)):
            st.session_state._api_key = api_key_val
            st.session_state._model_selection = model_sel
            st.session_state.web_search_enabled = web_enabled
            st.session_state._tavily_key = tavily_key_val if web_enabled else ""
            st.rerun()
    with col2:
        if st.button("Cancel", type="tertiary", use_container_width=True):
            st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Status ───────────────────────────────────────────────────────────────────
    st.header("Status")

    # Show a compact summary of current status
    with st.container():
        col_label, col_badge = st.columns([2, 1])
        with col_label:
            st.caption("Agent")
        with col_badge:
            if st.session_state._model_selection and st.session_state._api_key:
                st.badge("ready", color="green")
            else:
                st.badge("not ready", color="orange")

        col_label, col_badge = st.columns([2, 1])
        with col_label:
            st.caption("Web Search")
        with col_badge:
            if st.session_state.web_search_enabled and st.session_state._tavily_key:
                st.badge("active", color="green")
            else:
                st.badge("inactive", color="orange")
    
    if st.button("Settings", use_container_width=True):
        settings_dialog()

    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────────
    st.header("Documents")
    uploaded_files = st.file_uploader(
        f"Upload 1–{APP_CONFIG['max_file_upload_limit']} files (PDF, TXT, MD, DOCX)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        max_upload_size=APP_CONFIG["max_upload_size_mb"],
        label_visibility="collapsed"
    )

    _limit = APP_CONFIG["max_file_upload_limit"]
    _over_limit = bool(uploaded_files) and len(uploaded_files) > _limit
    if _over_limit:
        st.error(f"Too many files — maximum is {_limit}. Remove {len(uploaded_files) - _limit} file(s).")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    st.header("Controls")

    # Process Documents
    if st.button("Process Documents", use_container_width=True, disabled=not uploaded_files or _over_limit):
        if not st.session_state._api_key:
            st.error("Please set your OpenAI API key in Settings.")
        else:
            with st.spinner("Processing Documents..."):
                try:
                    st.session_state.messages = []

                    pipeline = RAGPipeline(
                        api_key=st.session_state._api_key,
                        model_name=st.session_state._model_selection,
                    )

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
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

    col_files, col_chat = st.columns(2)
    with col_files:
        if st.button("Clear Files", use_container_width=True, type="tertiary"):
            st.session_state.uploader_key += 1
            st.session_state.pipeline = None
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()
    with col_chat:
        if st.button("Clear Chat", use_container_width=True, type="tertiary"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.rerun()

# Convenience aliases resolved from session state
api_key = st.session_state._api_key
model_selection = st.session_state._model_selection
web_search_enabled = st.session_state.web_search_enabled
tavily_api_key = st.session_state._tavily_key

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


_BADGE_STYLES = {
    "rag": ("Answered via Documents", "blue"),
    "web": ("Answered via Web Search", "violet"),
}

def _render_source_badge(source: str, web_sources: list | None = None):
    """Render a compact source badge; for web answers, show a collapsible sources list."""
    label, color = _BADGE_STYLES.get(source, _BADGE_STYLES["rag"])
    st.badge(label, color=color)
    if source == "web" and web_sources:
        with st.expander("Sources", expanded=False):
            for s in web_sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

# ── Chat interface ────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_source_badge(message.get("source", "rag"), message.get("web_sources"))

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
                "Thinking (RAG → Web Search Fallback)..."
                if web_search_enabled and tavily_api_key
                else "Thinking (RAG only)..."
            )
            with st.spinner(spinner_label):
                try:
                    result = st.session_state.agent.chat(prompt)
                    answer = result["answer"]
                    source = result["source"]
                    web_sources = result.get("web_sources")

                    st.markdown(answer)
                    _render_source_badge(source, web_sources)

                    # Persist the result so the badge is re-rendered on history scroll
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source": source,
                        "web_sources": web_sources if source == "web" else None,
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
