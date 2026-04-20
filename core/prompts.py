"""Prompt templates used across the DocChat pipeline.

All prompts are plain ``str`` constants using ``textwrap.dedent`` for
readability.  They are parameterised with ``.format()`` by their callers.

Prompt inventory
----------------
MULTI_QUERY_PROMPT
    Generates query variations to improve vector-store recall (Multi-Query step).
HYDE_PROMPT
    Generates a hypothetical document passage for dense retrieval (HyDE step).
FINAL_GENERATION_PROMPT
    Synthesises a grounded answer from retrieved context (RAG generation step).
WEB_SEARCH_QUERY_PROMPT
    Rewrites a user question into an optimal web search query.
WEB_SEARCH_GENERATION_PROMPT
    Synthesises an answer from Tavily web search results.
AGENT_SYSTEM_PROMPT
    System-level instructions for the DocChat AgentWorkflow.
"""

from __future__ import annotations

import textwrap

MULTI_QUERY_PROMPT: str = textwrap.dedent("""\
    Your task is to generate {num_variations} different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. 
    Provide these alternative questions separated by newlines, with nothing else in the response.
    Original question: {query}
""").strip()

HYDE_PROMPT: str = textwrap.dedent("""\
    Please write a plausible passage to answer the question below.
    Question: {query}
    Passage:
""").strip()

FINAL_GENERATION_PROMPT: str = textwrap.dedent("""\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query. 
    You must only use the relevant facts provided in the context. 
    If the context does not contain enough information to answer the query, clearly state that you do not know the answer and do not attempt to guess, hallucinate, or make up information outside of the context.
    Query: {query}
    Answer:
""").strip()

WEB_SEARCH_QUERY_PROMPT: str = textwrap.dedent("""\
    You are a search query optimisation expert. Rewrite the user question below into a concise,
    effective web search query that will surface the most relevant and authoritative results.
    Return only the search query string — no explanation, no punctuation other than what belongs
    in the query itself.
    User question: {query}
    Search query:
""").strip()

WEB_SEARCH_GENERATION_PROMPT: str = textwrap.dedent("""\
    You are a helpful assistant. Use the web search results below to answer the user's question.
    Base your answer solely on the provided results. Cite the source URLs inline where relevant
    using markdown link syntax, e.g. [Source Title](https://example.com).
    If the results do not contain enough information, say so clearly — do not guess.

    Web search results:
    ---------------------
    {search_results}
    ---------------------
    Question: {query}
    Answer:
""").strip()

AGENT_SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are DocChat, an expert AI assistant that answers questions strictly based on
    the user's uploaded documents.

    ## Tool usage rules

    1. **ALWAYS call `query_documents` first** — no exceptions, for every user query.
    2. After receiving its response, evaluate quality:
       - If the answer is clear and grounded in document content → use it as your
         final answer verbatim or lightly reformatted. Do NOT call any other tool.
       - If the answer explicitly states it cannot find the information (phrases such
         as "I do not know", "not found in the documents", "the context does not
         contain") → the documents are insufficient.
    3. When documents are insufficient:
       - If `search_web` is available in your tool list → call it with a focused
         search query to retrieve supplementary information.
       - If `search_web` is NOT available → inform the user that the documents do
         not contain the answer and you cannot perform a web search.
    4. **Never call `search_web` before `query_documents`.**
    5. **Never fabricate information** — base every answer solely on tool outputs.
    6. When your final answer comes from web search results, include inline markdown
       citations for each source used: [Title](URL).
""").strip()
