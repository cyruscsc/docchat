from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.readers.file import PyMuPDFReader, MarkdownReader, DocxReader
from prompts import MULTI_QUERY_PROMPT, HYDE_PROMPT, FINAL_GENERATION_PROMPT

class RAGPipeline:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
                
        self.llm = OpenAI(model=model_name, api_key=api_key)
        self.embed_model = OpenAIEmbedding(api_key=api_key)
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        self.index = None
    
    def process_documents(self, input_files: List[str]):
        """Load and index documents."""
        extractor = {
            ".pdf": PyMuPDFReader(),
            ".md": MarkdownReader(),
            ".docx": DocxReader()
        }
        reader = SimpleDirectoryReader(
            input_files=input_files, 
            required_exts=[".pdf", ".txt", ".md", ".docx"],
            file_extractor=extractor
        )
        documents = reader.load_data()
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    def query(self, user_query: str) -> str:
        if not self.index:
            return "Please upload documents first."
            
        # 1. Multi-Query: Create 3 variations
        query_variations = self._generate_query_variations(user_query)
        all_queries = [user_query] + query_variations
        
        # 2. HyDE: Generate plausible answer for each query
        hypothetical_docs = self._generate_hypothetical_docs(all_queries)
        
        # 3. Retrieval
        retriever = self.index.as_retriever(similarity_top_k=20)
        # Combine variations and hypothetical docs to perform search
        search_strings = all_queries + hypothetical_docs
        
        all_results = []
        for search_string in search_strings:
            results = retriever.retrieve(search_string)
            all_results.append(results)
            
        # 4. RRF (Reciprocal Rank Fusion)
        reranked_docs = self._reciprocal_rank_fusion(all_results, top_k=5)
        
        # 5. Final Generation
        final_answer = self._generate_final_answer(user_query, reranked_docs)
        return final_answer
        
    def _generate_query_variations(self, query: str) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(query=query)
        response = self.llm.complete(prompt)
        variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        return variations[:3]
        
    def _generate_hypothetical_docs(self, queries: List[str]) -> List[str]:
        hypothetical_docs = []
        for q in queries:
            prompt = HYDE_PROMPT.format(query=q)
            response = self.llm.complete(prompt)
            hypothetical_docs.append(response.text.strip())
        return hypothetical_docs
        
    def _reciprocal_rank_fusion(self, results_list: List[List[NodeWithScore]], top_k: int = 5) -> List[NodeWithScore]:
        k_param = 60
        fused_scores = {}
        node_map = {}
        
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
