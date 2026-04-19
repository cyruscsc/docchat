"""DocChat agent — orchestrates document RAG and optional web search fallback.

Built on LlamaIndex >=0.14 AgentWorkflow API (``AgentWorkflow.from_tools_or_functions``).
The workflow ``run()`` method is async, so ``chat()`` drives it synchronously via
``asyncio.run()`` — safe in a Streamlit context since Streamlit does not run its own
event loop on the main thread.

The agent receives two potential tools:
  1. ``query_documents`` — Hybrid RAG over the user's uploaded documents.
                            Always tried first (enforced by the system prompt).
  2. ``search_web``      — Tavily web search fallback.
                            Only added when the user has enabled it and provided a
                            valid Tavily API key.

The agent decides autonomously whether to invoke web search based on the RAG
tool's own response — no hardcoded score threshold is used.
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from llama_index.core.agent import AgentWorkflow
from llama_index.core.tools import BaseTool
from core.prompts import AGENT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from tools.rag_tool import RAGPipeline


class DocChatAgent:
    """LlamaIndex AgentWorkflow powering the DocChat question-answering workflow."""

    def __init__(
        self,
        pipeline: "RAGPipeline",
        web_search_enabled: bool = False,
        tavily_api_key: str = "",
    ) -> None:
        """Instantiate the agent with its tool set.

        Args:
            pipeline: An already-initialised RAGPipeline (documents must have
                been processed before the agent is asked any questions).
            web_search_enabled: Whether to attach the web search tool.
            tavily_api_key: Tavily API key (required when web_search_enabled).
        """
        # Per-turn state — reset at the start of each chat() call
        self._web_used: bool = False
        self._web_sources: list[dict] = []

        # ── Assemble tools ────────────────────────────────────────────────────
        from tools.rag_tool import create_rag_tool
        from tools.web_search_tool import create_web_search_tool

        tools: list[BaseTool] = [create_rag_tool(pipeline)]

        if web_search_enabled and tavily_api_key:
            tools.append(
                create_web_search_tool(
                    tavily_api_key=tavily_api_key,
                    on_search_complete=self._on_search_complete,
                )
            )

        # ── Build the AgentWorkflow (LlamaIndex >= 0.14 API) ─────────────────
        self._workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=tools,
            llm=pipeline.llm,
            system_prompt=AGENT_SYSTEM_PROMPT,
            verbose=False,
        )
        # Persistent context across turns (preserves conversation memory)
        self._ctx = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def chat(self, user_query: str) -> dict[str, Any]:
        """Answer a user query and return a structured result.

        Drives the async AgentWorkflow synchronously so it can be called from
        Streamlit's synchronous execution context.

        Args:
            user_query: The user's natural-language question.

        Returns:
            A dict with:
            - ``"answer"``      — The final synthesised answer string.
            - ``"source"``      — ``"rag"`` or ``"web"``.
            - ``"web_sources"`` — List of ``{"title": str, "url": str}`` dicts
                                  when source is ``"web"``, else ``None``.
        """
        # Reset per-turn tracking state
        self._web_used = False
        self._web_sources = []

        response = asyncio.run(self._arun(user_query))

        return {
            "answer": str(response),
            "source": "web" if self._web_used else "rag",
            "web_sources": list(self._web_sources) if self._web_used else None,
        }

    def reset(self) -> None:
        """Clear the agent's in-memory conversation context."""
        self._ctx = None

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _arun(self, user_query: str) -> str:
        """Async entry point — runs the workflow and returns the final response."""
        handler = self._workflow.run(
            user_msg=user_query,
            ctx=self._ctx,
        )
        # Persist the context for multi-turn memory
        response = await handler
        self._ctx = handler.ctx
        return response.response if hasattr(response, "response") else str(response)

    def _on_search_complete(self, sources: list[dict]) -> None:
        """Invoked by the web search tool after a successful Tavily request."""
        self._web_used = True
        self._web_sources = sources
