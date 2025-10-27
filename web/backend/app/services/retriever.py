import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.faiss_index = None
        self.bm25 = None
        self.chunks = []
        self.index_path = settings.INDEX_STORAGE_PATH
        
    def build_index(self, chunks: List[Dict[str, any]]):
        """Build FAISS and BM25 indices from chunks"""
        logger.info(f"Building index from {len(chunks)} chunks")
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Index building complete")
    
    def add_documents(self, new_chunks: List[Dict[str, any]]):
        """Add new documents to existing indices"""
        logger.info(f"Adding {len(new_chunks)} new chunks to index")
        
        if not self.chunks:
            self.build_index(new_chunks)
            return
        
        # Add to chunks list
        self.chunks.extend(new_chunks)
        
        # Rebuild indices (for production, consider incremental updates)
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Rebuild FAISS
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        # Rebuild BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Indices updated successfully")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict[str, any], float]]:
        """Hybrid retrieval using both FAISS and BM25"""
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        if not self.chunks:
            logger.warning("No documents indexed")
            return []
        
        logger.info(f"Retrieving top {top_k} results for query")
        
        # FAISS retrieval
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        faiss_distances, faiss_indices = self.faiss_index.search(query_embedding, top_k * 2)
        faiss_scores = 1 / (1 + faiss_distances[0])  # Convert distance to similarity
        
        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        faiss_scores_norm = faiss_scores / (faiss_scores.max() + 1e-6)
        bm25_scores_norm = bm25_scores / (bm25_scores.max() + 1e-6)
        
        # Hybrid scoring (weighted combination)
        hybrid_scores = {}
        
        # Add FAISS scores
        for idx, score in zip(faiss_indices[0], faiss_scores_norm):
            if idx < len(self.chunks):
                hybrid_scores[idx] = hybrid_scores.get(idx, 0) + 0.6 * score
        
        # Add BM25 scores
        for idx, score in enumerate(bm25_scores_norm):
            hybrid_scores[idx] = hybrid_scores.get(idx, 0) + 0.4 * score
        
        # Sort by hybrid score
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return chunks with scores
        results = [(self.chunks[idx], score) for idx, score in sorted_results]
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    
    def save_index(self, index_name: str = "main_index"):
        """Save indices to disk"""
        logger.info(f"Saving index: {index_name}")
        
        index_dir = os.path.join(self.index_path, index_name)
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(index_dir, "faiss.index"))
        
        # Save BM25 and chunks
        with open(os.path.join(index_dir, "bm25_chunks.pkl"), "wb") as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        
        logger.info("Index saved successfully")
    
    def load_index(self, index_name: str = "main_index") -> bool:
        """Load indices from disk"""
        logger.info(f"Loading index: {index_name}")
        
        index_dir = os.path.join(self.index_path, index_name)
        
        if not os.path.exists(index_dir):
            logger.warning(f"Index {index_name} not found")
            return False
        
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
            
            # Load BM25 and chunks
            with open(os.path.join(index_dir, "bm25_chunks.pkl"), "rb") as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
            
            logger.info(f"Index loaded successfully with {len(self.chunks)} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False