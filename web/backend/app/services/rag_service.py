import uuid
from typing import Dict, List
from app.services.pdf_processor import PDFProcessor
from app.services.retriever import HybridRetriever
from app.services.claude_service import ClaudeService
from app.models.schemas import ChatResponse, SourceChunk, DocumentMetadata
from app.config import settings
from app.utils.logger import setup_logger
import os

logger = setup_logger(__name__)

class RAGService:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.retriever = HybridRetriever()
        self.claude_service = ClaudeService()
        
        # Load existing index if available
        self.retriever.load_index()
    
    def ingest_document(self, pdf_path: str, filename: str) -> Dict[str, any]:
        """Ingest a new PDF document"""
        logger.info(f"Ingesting document: {filename}")
        
        try:
            # Process PDF
            doc_id, chunks = self.pdf_processor.process_pdf(pdf_path, filename)
            
            # Add to retriever
            self.retriever.add_documents(chunks)
            
            # Save updated index
            self.retriever.save_index()
            
            return {
                'doc_id': doc_id,
                'filename': filename,
                'chunks_created': len(chunks),
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    def query(self, question: str, conversation_id: str = None) -> ChatResponse:
        """Process a question and generate answer with sources"""
        logger.info(f"Processing query: {question[:50]}...")
        
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Retrieve relevant chunks
            retrieved_results = self.retriever.retrieve(question)
            
            # Filter by confidence threshold
            filtered_results = [
                (chunk, score) for chunk, score in retrieved_results 
                if score >= settings.CONFIDENCE_THRESHOLD
            ]
            
            if not filtered_results:
                logger.warning("No relevant context found above threshold")
                return ChatResponse(
                    answer="I cannot find relevant information in the available documents to answer your question.",
                    sources=[],
                    is_context_based=False,
                    conversation_id=conversation_id
                )
            
            # Extract chunks for Claude
            context_chunks = [chunk for chunk, score in filtered_results]
            
            # Generate answer
            result = self.claude_service.generate_answer(question, context_chunks)
            
            # Prepare source chunks for response
            source_chunks = [
                SourceChunk(
                    text=chunk['text'],
                    metadata=DocumentMetadata(**chunk['metadata']),
                    score=round(score, 3),
                    page_number=chunk['metadata']['page_number']
                )
                for chunk, score in filtered_results
            ]
            
            return ChatResponse(
                answer=result['answer'],
                sources=source_chunks,
                is_context_based=result['is_context_based'],
                conversation_id=conversation_id
            )
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_indexed_documents(self) -> List[Dict[str, any]]:
        """Get list of indexed documents"""
        if not self.retriever.chunks:
            return []
        
        # Extract unique documents
        docs = {}
        for chunk in self.retriever.chunks:
            doc_id = chunk['metadata']['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'filename': chunk['metadata']['filename'],
                    'total_chunks': chunk['metadata']['total_chunks']
                }
        
        return list(docs.values())