"""LLM factory.

Single place responsible for constructing the OpenAI LLM and embedding model
and for wiring them into LlamaIndex ``Settings``.  Every component that needs
an LLM should call :func:`create_llm` rather than instantiating
``llama_index.llms.openai.OpenAI`` directly.
"""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from core.config import RAG_CONFIG


def create_llm(api_key: str, model_name: str) -> OpenAI:
    """Construct an OpenAI LLM and configure global LlamaIndex settings.

    Args:
        api_key:    OpenAI API key.
        model_name: Model identifier (e.g. ``"gpt-4o"``).

    Returns:
        The configured :class:`OpenAI` LLM instance.
    """
    llm = OpenAI(model=model_name, api_key=api_key)
    embed_model = OpenAIEmbedding(api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )

    return llm
