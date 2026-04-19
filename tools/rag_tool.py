"""RAG pipeline and tool for the DocChat agent.

This module combines the hybrid RAG pipeline (Multi-Query + HyDE + Reciprocal
Rank Fusion) with the LlamaIndex FunctionTool wrapper so the agent can call it
as its primary document-retrieval action.

Web search fallback is handled at the agent layer — see
``agents/doc_chat_agent.py``.
"""

from __future__ import annotations

from typing import List

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import FunctionTool
from llama_index.readers.file import DocxReader, MarkdownReader, PyMuPDFReader

from core.config import RAG_CONFIG
from core.llm import create_llm
from core.prompts import MULTI_QUERY_PROMPT, HYDE_PROMPT, FINAL_GENERATION_PROMPT


class RAGPipeline:
    """Hybrid RAG pipeline (Multi-Query + HyDE + Reciprocal Rank Fusion)."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.llm = create_llm(api_key=api_key, model_name=model_name)
        self.index = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_documents(self, input_files: List[str]) -> None:
        """Load and index documents."""
        extractor = {
            ".pdf": PyMuPDFReader(),
            ".md": MarkdownReader(),
            ".docx": DocxReader(),
        }
        reader = SimpleDirectoryReader(
            input_files=input_files,
            required_exts=[".pdf", ".txt", ".md", ".docx"],
            file_extractor=extractor,
        )
        documents = reader.load_data()
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)

    def query_rag(self, user_query: str) -> str:
        """Run the full hybrid RAG pipeline and return a grounded answer string.

        Pipeline steps:
          1. Multi-Query  — generate query variations to improve recall.
          2. HyDE         — create hypothetical documents for dense retrieval.
          3. Retrieval    — fetch candidate nodes for all query strings.
          4. RRF          — rerank via Reciprocal Rank Fusion.
          5. Generation   — synthesise a final answer from the top-k nodes.

        If the retrieved context is insufficient the generation prompt instructs
        the LLM to clearly say so, giving the DocChatAgent the signal it needs
        to decide whether to invoke the web search tool.

        Args:
            user_query: The user's original question.

        Returns:
            A grounded answer string, or a clear "I do not know" statement when
            the documents do not contain enough information.
        """
        if not self.index:
            return "No documents have been indexed yet. Please upload and process documents first."

        # 1. Multi-Query
        query_variations = self._generate_query_variations(
            user_query, num_variations=RAG_CONFIG["multi_query_variations"]
        )
        all_queries = [user_query] + query_variations

        # 2. HyDE
        hypothetical_docs = self._generate_hypothetical_docs(all_queries)

        # 3. Retrieval
        retriever = self.index.as_retriever(similarity_top_k=RAG_CONFIG["retrieval_top_k"])
        search_strings = all_queries + hypothetical_docs

        all_results: list[list[NodeWithScore]] = []
        for s in search_strings:
            all_results.append(retriever.retrieve(s))

        # 4. RRF
        reranked_docs = self._reciprocal_rank_fusion(
            all_results,
            top_k=RAG_CONFIG["rrf_top_k"],
            k_param=RAG_CONFIG["rrf_k_parameter"],
        )

        # 5. Generation
        return self._generate_final_answer(user_query, reranked_docs)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(num_variations=num_variations, query=query)
        response = self.llm.complete(prompt)
        variations = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
        return variations[:num_variations]

    def _generate_hypothetical_docs(self, queries: List[str]) -> List[str]:
        hypothetical_docs = []
        for q in queries:
            prompt = HYDE_PROMPT.format(query=q)
            response = self.llm.complete(prompt)
            hypothetical_docs.append(response.text.strip())
        return hypothetical_docs

    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[NodeWithScore]],
        top_k: int = 5,
        k_param: int = 60,
    ) -> List[NodeWithScore]:
        """Merge multiple retrieval result lists via Reciprocal Rank Fusion."""
        fused_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for results in results_list:
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.node_id
                node_map[node_id] = node_with_score
                if node_id not in fused_scores:
                    fused_scores[node_id] = 0.0
                fused_scores[node_id] += 1.0 / (rank + k_param)

        sorted_nodes = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_node_ids = [node_id for node_id, _ in sorted_nodes[:top_k]]
        return [node_map[node_id] for node_id in top_node_ids]

    def _generate_final_answer(self, original_query: str, nodes: List[NodeWithScore]) -> str:
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        prompt = FINAL_GENERATION_PROMPT.format(context_str=context_str, query=original_query)
        response = self.llm.complete(prompt)
        return response.text


# ──────────────────────────────────────────────────────────────────────────────
# Tool factory
# ──────────────────────────────────────────────────────────────────────────────

def create_rag_tool(pipeline: RAGPipeline) -> FunctionTool:
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
