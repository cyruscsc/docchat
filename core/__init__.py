"""Public API for the ``core`` package.

Re-exports the three typed configuration objects and the LLM factory so that
any module can do::

    from core import APP_CONFIG, RAG_CONFIG, WEB_SEARCH_CONFIG, create_llm

instead of reaching into sub-modules directly.
"""

from core.config import APP_CONFIG, RAG_CONFIG, WEB_SEARCH_CONFIG
from core.llm import create_llm
from core.prompts import AGENT_SYSTEM_PROMPT, MULTI_QUERY_PROMPT, HYDE_PROMPT, FINAL_GENERATION_PROMPT

__all__ = [
    "APP_CONFIG",
    "RAG_CONFIG",
    "WEB_SEARCH_CONFIG",
    "create_llm",
    "AGENT_SYSTEM_PROMPT",
    "MULTI_QUERY_PROMPT",
    "HYDE_PROMPT",
    "FINAL_GENERATION_PROMPT",
]
