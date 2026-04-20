"""Public API for the ``tools`` package.

Re-exports the RAG pipeline class and both tool factory functions so that
callers can do::

    from tools import RAGPipeline, create_rag_tool, create_web_search_tool

instead of reaching into sub-modules directly.
"""

from tools.rag import RAGPipeline, create_rag_tool
from tools.web_search import create_web_search_tool

__all__ = [
    "RAGPipeline",
    "create_rag_tool",
    "create_web_search_tool",
]
