
import torch
import numpy as np
import spacy
import faiss
from transformers import BitsAndBytesConfig
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from peft import PeftModel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# CONFIGURATION
class PipelineConfig:
    """Configuration class for the RAG pipeline."""
    # Embedding and Reranking Models
    EMBEDDING_MODEL = 'BAAI/bge-m3'
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' # Replace with a legal-tuned model when available

    # Generator LLM
    BASE_LLM_MODEL = 'meta-llama/Llama-2-7b-hf'
    # Assumes LoRA adapters are trained and saved in this directory
    LORA_ADAPTER_PATH = './lora_adapters/llama2-legal-tuned'

    # Indexing and Retrieval
    FAISS_INDEX_PATH = 'legal_faiss.index'
    TOP_K_BM25 = 5
    TOP_K_FAISS = 5
    TOP_K_RERANKED = 3

    # Dynamic Loop
    RERANKER_CONFIDENCE_THRESHOLD = 0.5
    MAX_RETRIEVAL_LOOPS = 2

    # Quantization for LLM loading
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

# COMPONENT IMPLEMENTATIONS

class PreprocessingEngine:
    """Handles NER and Jurisdiction Detection."""
    def __init__(self):
        # Using a general-purpose NER model. For better performance,
        # this should be replaced with a model fine-tuned on legal text.
        self.nlp = spacy.load("en_core_web_sm")
        print("PreprocessingEngine: spaCy NER model loaded.")

    def extract_entities(self, query: str) -> list:
        """Extracts named entities from the query."""
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]
        print(f"PreprocessingEngine: Extracted entities -> {entities}")
        return entities


class KnowledgeGraphExpander:
    """Simulates query expansion using a knowledge graph."""
    def __init__(self):
        # In a real system, this would connect to a graph database like Neo4j.
        # Here, we simulate it with a simple dictionary.
        self.simulated_kg = {
            "Japanese Civil Law": ["contract law", "property rights"],
            "Federal Court of Canada": ["judicial review", "intellectual property"],
            "contract termination": ["breach of contract", "notice period", "remedies"]
        }
        print("KnowledgeGraphExpander: Simulated KG initialized.")

    def expand_query(self, query: str, entities: list) -> str:
        """Expands the query with related terms from the KG."""
        expanded_terms = []
        for entity in entities:
            if entity in self.simulated_kg:
                expanded_terms.extend(self.simulated_kg[entity])

        if not expanded_terms:
            print("KnowledgeGraphExpander: No expansion terms found.")
            return query

        expanded_query = query + " " + " ".join(expanded_terms)
        print(f"KnowledgeGraphExpander: Expanded query -> {expanded_query}")
        return expanded_query
    
    
class HybridRetriever:
    """Combines BM25 (sparse) and FAISS (dense) retrieval."""
    def __init__(self, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25 = None
        self.faiss_index = None
        self.documents = []
        self.doc_ids =  []
        print(f"HybridRetriever: Loaded embedding model '{embedding_model_name}'.")

    def build_indexes(self, documents: dict):
        """Builds BM25 and FAISS indexes from a dictionary of documents."""
        self.documents = list(documents.values())
        self.doc_ids = list(documents.keys())
        
        # 1. Build BM25 Index
        tokenized_corpus = [doc.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("HybridRetriever: BM25 index built.")

        # 2. Build FAISS Index
        print("HybridRetriever: Generating embeddings for FAISS...")
        embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=True, show_progress_bar=True)
        embeddings_np = embeddings.cpu().numpy()
        
        dimension = embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_np)
        print(f"HybridRetriever: FAISS index built with {self.faiss_index.ntotal} vectors.")

    def retrieve(self, query: str, top_k_bm25: int, top_k_faiss: int) -> list:
        """Performs hybrid retrieval and returns a list of candidate document IDs."""
        if not self.bm25 or not self.faiss_index:
            raise RuntimeError("Indexes are not built. Call build_indexes() first.")

        # BM25 Retrieval
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]
        bm25_doc_ids = [self.doc_ids[i] for i in top_bm25_indices]
        print(f"HybridRetriever: BM25 retrieved doc IDs -> {bm25_doc_ids}")

        # FAISS Retrieval
        query_embedding = self.embedding_model.encode([query])
        _, top_faiss_indices = self.faiss_index.search(query_embedding, top_k_faiss)
        faiss_doc_ids = [self.doc_ids[i] for i in top_faiss_indices]
        print(f"HybridRetriever: FAISS retrieved doc IDs -> {faiss_doc_ids}")

        # Combine and deduplicate results
        combined_ids = list(dict.fromkeys(bm25_doc_ids + faiss_doc_ids))
        print(f"HybridRetriever: Combined candidate doc IDs -> {combined_ids}")
        return combined_ids
    

class Reranker:
    """Reranks documents using a Cross-Encoder model."""
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        print(f"Reranker: Loaded Cross-Encoder model '{model_name}'.")

    def rerank(self, query: str, docs: dict, doc_ids: list) -> list:
        """Reranks a list of documents and returns them sorted by relevance."""
        pairs = [[query, docs[doc_id]] for doc_id in doc_ids]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        reranked_results = []
        for i, doc_id in enumerate(doc_ids):
            reranked_results.append({
                "id": doc_id,
                "score": scores[i],
                "text": docs[doc_id]
            })
        
        # Sort by score in descending order
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        print(f"Reranker: Reranked {len(reranked_results)} docs. Top score: {reranked_results['score']:.4f}")
        return reranked_results
    
class Generator:
    """Loads a LoRA-fine-tuned LLM and generates text."""
    def __init__(self, base_model_name: str, lora_adapter_path: str, bnb_config):
        print(f"Generator: Loading base model '{base_model_name}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            print(f"Generator: Loading LoRA adapter from '{lora_adapter_path}'...")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            print("Generator: LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"Generator: WARNING - Could not load LoRA adapter. Using base model. Error: {e}")

    def generate(self, query: str, context: str) -> str:
        """Generates an answer based on the query and context."""
        prompt = self._build_prompt(query, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print("Generator: Generating response...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        # Clean up the response to only return the generated answer
        answer = response.split("ANSWER:")[-1].strip()
        print("Generator: Response generated.")
        return answer

    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the prompt for the LLM."""
        return f"""
            You are an expert legal assistant. Your task is to answer the user's question based exclusively on the provided legal texts. Do not use any prior knowledge. For every statement in your answer, you MUST provide an inline citation referencing the exact source document ID (e.g., [doc_id_1]).

            --- CONTEXT START ---
            {context}
            --- CONTEXT END ---

            USER QUESTION:
            {query}

            ANSWER:
        """
  
  
class CorrectiveLayer:
    """Performs a final consistency and hallucination check."""
    def __init__(self, generator: Generator):
        # Uses the same generator LLM for the check, but could use a separate one.
        self.checker_llm = generator
        print("CorrectiveLayer: Initialized.")

    def check(self, response: str, context: str) -> dict:
        """Checks the response for hallucinations against the context."""
        print("CorrectiveLayer: Performing hallucination check...")
        prompt = self._build_check_prompt(response, context)
        
        # In a real system, this would be a more robust check.
        # Here, we simulate by asking the LLM to self-critique.
        raw_check_output = self.checker_llm.generate(
            "Is the following statement fully supported by the provided context?",
            f"Statement: '{response}'\n\nContext: '{context}'"
        )
        
        # Simple logic for demonstration
        is_consistent = "yes" in raw_check_output.lower()
        
        result = {
            "is_consistent": is_consistent,
            "reasoning": raw_check_output
        }
        print(f"CorrectiveLayer: Check complete. Consistent: {is_consistent}")
        return result

    def _build_check_prompt(self, response: str, context: str) -> str:
        return f"""
You are a verification agent. Your task is to determine if the 'RESPONSE' is fully supported by the 'CONTEXT' provided.
Answer with only 'Yes' or 'No', followed by a brief explanation.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

--- RESPONSE START ---
{response}
--- RESPONSE END ---

Is the RESPONSE fully supported by the CONTEXT?
ANSWER:
"""
    

class CitationParser:
    """Utility to parse and format citations."""
    @staticmethod
    def link_citations(response: str, reranked_docs: list) -> str:
        """Finds citation markers and adds placeholder links."""
        # In a real UI, these would be actual hyperlinks.
        # Here we just format them for clarity.
        doc_map = {doc['id']: f"Source: {doc['id']}" for doc in reranked_docs}
        
        def replace_citation(match):
            doc_id = match.group(1)
            return f"[{doc_id}]({doc_map.get(doc_id, 'Source Not Found')})"

        # Regex to find citations like [doc_id_1]
        linked_response = re.sub(r'\[(doc_id_\d+)\]', replace_citation, response)
        return linked_response
    

class LegalRAGPipeline:
    """Orchestrates the entire RAG pipeline."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.documents = {}
        
        # Initialize all components
        self.preprocessor = PreprocessingEngine()
        self.kg_expander = KnowledgeGraphExpander()
        self.retriever = HybridRetriever(self.config.EMBEDDING_MODEL)
        self.reranker = Reranker(self.config.RERANKER_MODEL)
        self.generator = Generator(
            self.config.BASE_LLM_MODEL,
            self.config.LORA_ADAPTER_PATH,
            self.config.BNB_CONFIG
        )
        self.corrective_layer = CorrectiveLayer(self.generator)
        self.citation_parser = CitationParser()

        self.audit_log = []

    def process_documents(self, documents: dict):
        """Loads and indexes the legal documents."""
        print("\n--- Starting Document Processing ---")
        self.documents = documents
        self.retriever.build_indexes(self.documents)
        print("--- Document Processing Complete ---\n")

    def answer_query(self, query: str):
        """Main method to answer a user query, following the full pipeline."""
        print(f"\n--- Answering Query: '{query}' ---")
        self.audit_log.append(f"INITIAL_QUERY: {query}")

        # Layer 1: Query Ingestion & Expansion
        entities = self.preprocessor.extract_entities(query)
        expanded_query = self.kg_expander.expand_query(query, entities)
        self.audit_log.append(f"EXPANDED_QUERY: {expanded_query}")

        # Dynamic Retrieval Loop
        reranked_docs = []
        context_is_sufficient = False
        current_loop = 0
        
        while not context_is_sufficient and current_loop < self.config.MAX_RETRIEVAL_LOOPS:
            current_loop += 1
            print(f"\n--- Retrieval Loop #{current_loop} ---")
            self.audit_log.append(f"LOOP_{current_loop}_QUERY: {expanded_query}")

            # Layer 2: Hybrid Retrieval & Reranking
            candidate_ids = self.retriever.retrieve(
                expanded_query,
                self.config.TOP_K_BM25,
                self.config.TOP_K_FAISS
            )
            
            if not candidate_ids:
                print("No documents retrieved. Aborting.")
                return {"answer": "Could not find any relevant documents to answer the query."}

            reranked_docs = self.reranker.rerank(query, self.documents, candidate_ids)
            self.audit_log.append(f"LOOP_{current_loop}_RERANKED_IDS: {[d['id'] for d in reranked_docs]}")

            # Check for sufficiency
            if reranked_docs and reranked_docs['score'] >= self.config.RERANKER_CONFIDENCE_THRESHOLD:
                context_is_sufficient = True
                print(f"Context deemed sufficient. Top score {reranked_docs['score']:.4f} >= {self.config.RERANKER_CONFIDENCE_THRESHOLD}")
            else:
                print("Context may be insufficient. Reformulating query for next loop.")
                # Simple reformulation for demonstration
                expanded_query += " legal precedent"
        
        # Prepare context for generator
        final_docs = reranked_docs
        context = "\n\n".join([f"Content from [{doc['id']}]:\n{doc['text']}" for doc in final_docs])
        self.audit_log.append(f"FINAL_CONTEXT_DOCS: {[d['id'] for d in final_docs]}")

        # Layer 3: Generation
        generated_answer = self.generator.generate(query, context)
        self.audit_log.append(f"GENERATED_ANSWER: {generated_answer}")
        
        # Layer 4: Quality Assurance & Finalization
        correction_result = self.corrective_layer.check(generated_answer, context)
        self.audit_log.append(f"CORRECTION_RESULT: {correction_result}")

        final_answer = generated_answer
        if not correction_result['is_consistent']:
            final_answer += "\n\n"

        # Add citation links
        final_answer_with_citations = self.citation_parser.link_citations(final_answer, final_docs)

        # Final Output
        output = {
            "query": query,
            "answer": final_answer_with_citations,
            "sources": final_docs,
            "correction_check": correction_result
        }
        
        print("\n--- Query Processing Complete ---")
        return output

if __name__ == "__main__":
    # --- 1. Setup ---
    # This assumes you have downloaded the models and have a Hugging Face token configured.
    # You may need to run `huggingface-cli login` in your terminal.
    
    # Create a dummy LoRA adapter directory for demonstration if it doesn't exist
    import os
    if not os.path.exists(PipelineConfig.LORA_ADAPTER_PATH):
        os.makedirs(PipelineConfig.LORA_ADAPTER_PATH)
        print(f"Created dummy LoRA adapter directory at '{PipelineConfig.LORA_ADAPTER_PATH}'.")
        print("NOTE: The generator will use the base LLaMA model as no real adapters are present.")

    # --- 2. Initialize Pipeline ---
    config = PipelineConfig()
    pipeline = LegalRAGPipeline(config)

    # --- 3. Load and Process Documents (Simulated COLIEE Dataset) ---
    # This is a small, simulated dataset for demonstration purposes.
    # A real implementation would load thousands of documents from the COLIEE corpus.
    legal_documents = {
        "doc_id_1": "Article 90 of the Japanese Civil Law states that a juridical act with a purpose which is against public policy is void. This includes acts related to illegal gambling.",
        "doc_id_2": "In the case of Smith v. Jones (Federal Court of Canada, 2021), the court ruled that contract termination requires a clear and unambiguous notice period as specified in the agreement.",
        "doc_id_3": "The principle of judicial review allows the Federal Court of Canada to assess the lawfulness of a decision or action made by a public body.",
        "doc_id_4": "Under Japanese Civil Law, a contract is formed upon the meeting of minds (offer and acceptance), and does not necessarily require written form unless specified by statute.",
        "doc_id_5": "Canadian intellectual property law protects creations of the mind, such as inventions, literary and artistic works, designs, and symbols, names and images used in commerce."
    }
    pipeline.process_documents(legal_documents)

    # --- 4. Answer a Query ---
    user_query = "What is required for contract termination according to the Federal Court of Canada?"
    result = pipeline.answer_query(user_query)

    # --- 5. Display Results ---
    print("\n\n================ FINAL RESULT ================")
    print(f"Query: {result['query']}")
    print("\nAnswer:")
    print(result['answer'])
    print("\nSources:")
    for source in result['sources']:
        print(f"- ID: {source['id']}, Score: {source['score']:.4f}")
    print("\nCorrection Check:")
    print(f"  - Consistent: {result['correction_check']['is_consistent']}")
    print(f"  - Reasoning: {result['correction_check']['reasoning']}")
    print("============================================")

    # --- 6. View Audit Log ---
    # print("\n--- AUDIT LOG ---")
    # for log_entry in pipeline.audit_log:
    #     print(log_entry)
