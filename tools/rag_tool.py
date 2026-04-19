"""RAG tool for the DocChat agent.

Wraps the RAGPipeline as a LlamaIndex FunctionTool so the agent can call it
as its primary document-retrieval action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.tools import FunctionTool

if TYPE_CHECKING:
    from rag_pipeline import RAGPipeline


def create_rag_tool(pipeline: "RAGPipeline") -> FunctionTool:
    """Return a FunctionTool that queries documents via the hybrid RAG pipeline.

    The tool description explicitly instructs the agent to call this first for
    every user query before considering any other tool.
    """

    def query_documents(query: str) -> str:
        """Search the uploaded documents and return a grounded answer.

        Uses a hybrid RAG pipeline (Multi-Query + HyDE + Reciprocal Rank Fusion)
        to retrieve the most relevant passages and synthesise an answer from them.

        If the documents do not contain enough information the response will
        clearly say so (e.g. "I do not know" / "not found in the documents").
        That signal tells the agent it may try the web search tool next.

        Args:
            query: The user's question exactly as formulated (or a rephrased
                   variant that captures the same information need).

        Returns:
            A factual answer grounded in the document context, or a statement
            that the information was not found in the uploaded documents.
        """
        return pipeline.query_rag(query)

    return FunctionTool.from_defaults(
        fn=query_documents,
        name="query_documents",
        description=(
            "Search the user's uploaded documents using a hybrid RAG pipeline "
            "(Multi-Query + HyDE + Reciprocal Rank Fusion) and return a grounded answer. "
            "ALWAYS call this tool first for every user query. "
            "If it responds that it cannot find the answer, you may fall back to "
            "the web search tool — but only if that tool is available."
        ),
    )
