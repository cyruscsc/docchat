import textwrap

MULTI_QUERY_PROMPT = textwrap.dedent("""\
    Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. 
    Provide these alternative questions separated by newlines, with nothing else in the response.
    Original question: {query}
""").strip()

HYDE_PROMPT = textwrap.dedent("""\
    Please write a plausible passage to answer the question below.
    Question: {query}
    Passage:
""").strip()

FINAL_GENERATION_PROMPT = textwrap.dedent("""\
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
