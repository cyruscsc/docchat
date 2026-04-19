"""Web search tool for the DocChat agent.

Wraps the Tavily Search API as a LlamaIndex FunctionTool.  The tool is only
added to the agent when the user has enabled web search fallback and provided
a valid Tavily API key, so the agent cannot invoke it otherwise.
"""

from __future__ import annotations

from typing import Callable

from llama_index.core.tools import FunctionTool

from core.config import WEB_SEARCH_CONFIG as _WEB_CFG


def create_web_search_tool(
    tavily_api_key: str,
    on_search_complete: Callable[[list[dict]], None] | None = None,
) -> FunctionTool:
    """Return a FunctionTool that searches the web via Tavily.

    Args:
        tavily_api_key: Valid Tavily API key.
        on_search_complete: Optional callback invoked with a list of
            ``{"title": str, "url": str}`` dicts after each successful search.
            The DocChatAgent uses this to track whether web search was used and
            to collect source metadata for the UI banner.
    """
    from tavily import TavilyClient  # lazy import — optional dependency

    _client = TavilyClient(api_key=tavily_api_key)
    _max_results: int = _WEB_CFG["max_results"]

    def search_web(query: str) -> str:
        """Search the web for information about the given query.

        Only call this tool when ``query_documents`` has explicitly stated that
        the uploaded documents do not contain enough information to answer the
        question.  Do NOT call this tool before trying ``query_documents``.

        The results include titles, URLs, and content snippets from relevant
        web pages.  Cite the source URLs inline in your final answer using
        markdown link syntax: ``[Title](URL)``.

        Args:
            query: A focused, self-contained search query string.

        Returns:
            Formatted web search results (title + URL + content snippet per
            result), or a message stating no results were found.
        """
        response = _client.search(
            query=query,
            max_results=_max_results,
            search_depth="advanced",
            include_answer=False,
        )
        results = response.get("results", [])

        if not results:
            return "No relevant web results were found for this query."

        # Notify the agent so it can record source metadata for the UI
        if on_search_complete is not None:
            sources = [
                {"title": r.get("title", r["url"]), "url": r["url"]}
                for r in results
            ]
            on_search_complete(sources)

        # Format the results as readable text for the LLM
        formatted_blocks = []
        for i, r in enumerate(results, start=1):
            block = (
                f"[{i}] {r.get('title', 'Untitled')}\n"
                f"URL: {r['url']}\n"
                f"{r.get('content', '').strip()}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    return FunctionTool.from_defaults(
        fn=search_web,
        name="search_web",
        description=(
            "Search the web for up-to-date information using the Tavily API. "
            "Use this ONLY as a fallback when query_documents indicated that the "
            "uploaded documents do not contain sufficient information. "
            "Provide a focused, self-contained search query as input. "
            "Cite source URLs inline in your final answer."
        ),
    )
