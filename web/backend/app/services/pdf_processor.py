import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import hashlib
import os
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def generate_doc_id(self, filename: str) -> str:
        """Generate unique document ID from filename"""
        return hashlib.md5(filename.encode()).hexdigest()[:12]
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extract text from PDF with page information"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': text.strip()
                })
            
            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path}")
            return pages_data
        
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    def chunk_text(self, text: str, page_number: int) -> List[Dict[str, any]]:
        """Split text into overlapping chunks"""
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [{'text': text, 'page_number': page_number, 'chunk_index': 0}]
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunks.append({
                'text': chunk,
                'page_number': page_number,
                'chunk_index': chunk_index
            })
            
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def process_pdf(self, pdf_path: str, filename: str) -> Tuple[str, List[Dict[str, any]]]:
        """Process PDF and return doc_id and chunks with metadata"""
        doc_id = self.generate_doc_id(filename)
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.chunk_text(page_data['text'], page_data['page_number'])
            all_chunks.extend(page_chunks)
        
        # Add metadata to chunks
        chunks_with_metadata = []
        for idx, chunk in enumerate(all_chunks):
            chunks_with_metadata.append({
                'text': chunk['text'],
                'metadata': {
                    'doc_id': doc_id,
                    'filename': filename,
                    'page_number': chunk['page_number'],
                    'chunk_index': idx,
                    'total_chunks': len(all_chunks)
                }
            })
        
        logger.info(f"Processed PDF into {len(chunks_with_metadata)} chunks")
        return doc_id, chunks_with_metadata