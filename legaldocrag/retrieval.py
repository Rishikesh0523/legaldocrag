import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
	"""Combines BM25 (sparse) and FAISS (dense) retrieval."""
	def __init__(self, embedding_model_name: str):
		self.embedding_model = SentenceTransformer(embedding_model_name)
		self.bm25 = None
		self.faiss_index = None
		self.documents = []
		self.doc_ids = []
		print(f"HybridRetriever: Loaded embedding model '{embedding_model_name}'.")

	def build_indexes(self, documents: dict):
		"""Builds BM25 and FAISS indexes from a dictionary of documents."""
		# Filter out None or empty documents
		valid_items = [(doc_id, doc) for doc_id, doc in documents.items() 
		               if doc is not None and isinstance(doc, str) and doc.strip()]
		
		if len(valid_items) != len(documents):
			skipped = len(documents) - len(valid_items)
			print(f"HybridRetriever: Skipped {skipped} invalid/empty documents")
		
		self.doc_ids = [item[0] for item in valid_items]
		self.documents = [item[1] for item in valid_items]
		
		print(f"HybridRetriever: Building indexes for {len(self.documents)} valid documents")

		# 1. Build BM25 Index
		tokenized_corpus = [doc.split(" ") for doc in self.documents]
		self.bm25 = BM25Okapi(tokenized_corpus)
		print("HybridRetriever: BM25 index built.")

		# 2. Build FAISS Index
		print("HybridRetriever: Generating embeddings for FAISS...")
		embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=True, show_progress_bar=True)
		embeddings_np = embeddings.cpu().numpy().astype('float32')

		dimension = embeddings_np.shape[1]
		self.faiss_index = faiss.IndexFlatL2(dimension)
		self.faiss_index.add(embeddings_np)
		print(f"HybridRetriever: FAISS index built with {self.faiss_index.ntotal} vectors.")

	def retrieve(self, query: str, top_k_bm25: int, top_k_faiss: int) -> list:
		"""Performs hybrid retrieval and returns a list of candidate document IDs."""
		if self.bm25 is None or self.faiss_index is None:
			raise RuntimeError("Indexes are not built. Call build_indexes() first.")

		# BM25 Retrieval
		tokenized_query = query.split(" ")
		bm25_scores = self.bm25.get_scores(tokenized_query)
		top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]
		bm25_doc_ids = [self.doc_ids[i] for i in top_bm25_indices]
		print(f"HybridRetriever: BM25 retrieved doc IDs -> {bm25_doc_ids}")

		# FAISS Retrieval
		query_embedding = self.embedding_model.encode([query])
		if not isinstance(query_embedding, np.ndarray):
			query_embedding = np.array(query_embedding)
		_, top_faiss_indices = self.faiss_index.search(query_embedding.astype('float32'), top_k_faiss)
		top_faiss_indices = top_faiss_indices[0].tolist()
		faiss_doc_ids = [self.doc_ids[i] for i in top_faiss_indices]
		print(f"HybridRetriever: FAISS retrieved doc IDs -> {faiss_doc_ids}")

		# Combine and deduplicate results
		combined_ids = list(dict.fromkeys(bm25_doc_ids + faiss_doc_ids))
		print(f"HybridRetriever: Combined candidate doc IDs -> {combined_ids}")
		return combined_ids
