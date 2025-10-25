"""
Document Loader Module

Handles loading and preprocessing of legal documents from various sources:
- Mexico Civil Law (COLIEE dataset)
- Nepal Constitution
- Multi-jurisdiction support with metadata
- PDF processing with smart caching
"""

import json
import os
import hashlib
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class LegalDocumentLoader:
	"""
	Loads legal documents from various formats and jurisdictions.
	
	Supports:
	- JSON format documents
	- PDF documents with caching
	- Metadata extraction
	- Jurisdiction tagging
	- Document versioning
	"""
	
	def __init__(self, cache_dir: str = './data/.cache'):
		"""
		Initialize document loader.
		
		Args:
			cache_dir: Directory to store processed PDF cache files
		"""
		self.documents = {}
		self.metadata = {}
		self.cache_dir = Path(cache_dir)
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		print(f"LegalDocumentLoader: Initialized (cache: {self.cache_dir})")
	
	def load_mexico_civil_law(self, filepath: str) -> Dict[str, str]:
		"""
		Load Mexico Civil Law from COLIEE dataset (articlesFull.json).
		
		Args:
			filepath: Path to articlesFull.json
			
		Returns:
			Dictionary mapping article IDs to article text
		"""
		print(f"LegalDocumentLoader: Loading Mexico Civil Law from '{filepath}'...")
		
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"Mexico Civil Law file not found: {filepath}")
		
		with open(filepath, 'r', encoding='utf-8') as f:
			articles = json.load(f)
		
		# Add jurisdiction metadata
		documents = {}
		for article_id, text in articles.items():
			doc_id = f"mexico_article_{article_id}"
			documents[doc_id] = text
			
			# Store metadata
			self.metadata[doc_id] = {
				'jurisdiction': 'mexico',
				'source': 'civil_law',
				'article_number': article_id,
				'type': 'statute'
			}
		
		print(f"LegalDocumentLoader: Loaded {len(documents)} Mexico Civil Law articles")
		return documents
	
	def load_mexico_training_data(self, filepath: str) -> Dict[str, str]:
		"""
		Load Mexico training data (TrainingData.json).
		
		This contains question-answer pairs with article references.
		
		Args:
			filepath: Path to TrainingData.json
			
		Returns:
			Dictionary mapping doc IDs to legal statements
		"""
		print(f"LegalDocumentLoader: Loading Mexico training data from '{filepath}'...")
		
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"Training data file not found: {filepath}")
		
		with open(filepath, 'r', encoding='utf-8') as f:
			training_data = json.load(f)
		
		documents = {}
		idx = 0
		
		for statement, article_refs in training_data.items():
			doc_id = f"mexico_case_{idx}"
			documents[doc_id] = statement.strip()
			
			# Store metadata with article references
			self.metadata[doc_id] = {
				'jurisdiction': 'mexico',
				'source': 'training_data',
				'article_references': article_refs,
				'type': 'case_statement'
			}
			idx += 1
		
		print(f"LegalDocumentLoader: Loaded {len(documents)} Mexico training statements")
		return documents
	
	def load_nepal_constitution(self, filepath: str) -> Dict[str, str]:
		"""
		Load Nepal Constitution.
		
		Args:
			filepath: Path to Nepal constitution JSON file
			
		Returns:
			Dictionary mapping section IDs to text
		"""
		print(f"LegalDocumentLoader: Loading Nepal Constitution from '{filepath}'...")
		
		if not os.path.exists(filepath):
			print(f"LegalDocumentLoader: Nepal Constitution not found at '{filepath}' - skipping")
			return {}
		
		with open(filepath, 'r', encoding='utf-8') as f:
			constitution = json.load(f)
		
		documents = {}
		
		# Handle different possible JSON structures
		if isinstance(constitution, dict):
			for section_id, text in constitution.items():
				doc_id = f"nepal_constitution_{section_id}"
				documents[doc_id] = text
				
				self.metadata[doc_id] = {
					'jurisdiction': 'nepal',
					'source': 'constitution',
					'section_number': section_id,
					'type': 'constitutional_provision'
				}
		elif isinstance(constitution, list):
			# If it's a list of sections
			for idx, section in enumerate(constitution):
				doc_id = f"nepal_constitution_{idx}"
				if isinstance(section, dict):
					text = section.get('text', section.get('content', str(section)))
				else:
					text = str(section)
				
				documents[doc_id] = text
				
				self.metadata[doc_id] = {
					'jurisdiction': 'nepal',
					'source': 'constitution',
					'section_number': str(idx),
					'type': 'constitutional_provision'
				}
		
		print(f"LegalDocumentLoader: Loaded {len(documents)} Nepal Constitution sections")
		return documents
	
	def _get_pdf_hash(self, pdf_path: str) -> str:
		"""
		Calculate hash of PDF file for cache validation.
		
		Args:
			pdf_path: Path to PDF file
			
		Returns:
			MD5 hash of the file
		"""
		hasher = hashlib.md5()
		with open(pdf_path, 'rb') as f:
			# Read in chunks for large files
			for chunk in iter(lambda: f.read(8192), b""):
				hasher.update(chunk)
		return hasher.hexdigest()
	
	def _get_cache_path(self, pdf_path: str) -> Path:
		"""
		Get cache file path for a PDF.
		
		Args:
			pdf_path: Path to PDF file
			
		Returns:
			Path to cache file
		"""
		pdf_hash = self._get_pdf_hash(pdf_path)
		pdf_name = Path(pdf_path).stem
		cache_filename = f"{pdf_name}_{pdf_hash}.cache"
		return self.cache_dir / cache_filename
	
	def _load_from_cache(self, pdf_path: str) -> Optional[Tuple[Dict[str, str], Dict[str, Dict]]]:
		"""
		Load processed PDF data from cache.
		
		Args:
			pdf_path: Path to PDF file
			
		Returns:
			Tuple of (documents, metadata) if cache exists and is valid, None otherwise
		"""
		cache_path = self._get_cache_path(pdf_path)
		
		if not cache_path.exists():
			return None
		
		try:
			with open(cache_path, 'rb') as f:
				cache_data = pickle.load(f)
			
			# Validate cache structure
			if not isinstance(cache_data, dict) or 'documents' not in cache_data or 'metadata' not in cache_data:
				print(f"  - Cache file corrupted, will reprocess")
				return None
			
			print(f"  - ✓ Loaded from cache: {cache_path.name}")
			return cache_data['documents'], cache_data['metadata']
			
		except Exception as e:
			print(f"  - Warning: Could not load cache: {e}")
			return None
	
	def _save_to_cache(self, pdf_path: str, documents: Dict[str, str], metadata: Dict[str, Dict]):
		"""
		Save processed PDF data to cache.
		
		Args:
			pdf_path: Path to PDF file
			documents: Processed documents dictionary
			metadata: Metadata dictionary
		"""
		cache_path = self._get_cache_path(pdf_path)
		
		try:
			cache_data = {
				'documents': documents,
				'metadata': metadata,
				'processed_at': datetime.now().isoformat(),
				'pdf_path': str(pdf_path)
			}
			
			with open(cache_path, 'wb') as f:
				pickle.dump(cache_data, f)
			
			print(f"  - ✓ Cached to: {cache_path.name}")
			
		except Exception as e:
			print(f"  - Warning: Could not save cache: {e}")
	
	def _extract_text_from_pdf_pypdf2(self, pdf_path: str) -> List[Tuple[int, str]]:
		"""
		Extract text from PDF using PyPDF2.
		
		Args:
			pdf_path: Path to PDF file
			
		Returns:
			List of (page_number, text) tuples
		"""
		try:
			import PyPDF2
		except ImportError:
			raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install pypdf2")
		
		pages = []
		with open(pdf_path, 'rb') as f:
			pdf_reader = PyPDF2.PdfReader(f)
			total_pages = len(pdf_reader.pages)
			
			for page_num in range(total_pages):
				page = pdf_reader.pages[page_num]
				text = page.extract_text()
				if text.strip():  # Only add non-empty pages
					pages.append((page_num + 1, text))
		
		return pages
	
	def _extract_text_from_pdf_pdfplumber(self, pdf_path: str) -> List[Tuple[int, str]]:
		"""
		Extract text from PDF using pdfplumber (better for complex layouts).
		
		Args:
			pdf_path: Path to PDF file
			
		Returns:
			List of (page_number, text) tuples
		"""
		try:
			import pdfplumber
		except ImportError:
			raise ImportError("pdfplumber is required for PDF processing. Install with: pip install pdfplumber")
		
		pages = []
		with pdfplumber.open(pdf_path) as pdf:
			for page_num, page in enumerate(pdf.pages, start=1):
				text = page.extract_text()
				if text and text.strip():
					pages.append((page_num, text))
		
		return pages
	
	def load_pdf(
		self,
		pdf_path: str,
		jurisdiction: Optional[str] = None,
		doc_type: str = 'legal_document',
		use_cache: bool = True,
		parser: str = 'pypdf2',
		chunk_by: str = 'page'  # 'page' or 'section'
	) -> Dict[str, str]:
		"""
		Load and process a PDF document with smart caching.
		
		Args:
			pdf_path: Path to PDF file
			jurisdiction: Jurisdiction tag (e.g., 'nepal', 'mexico')
			doc_type: Type of document (e.g., 'constitution', 'statute', 'case_law')
			use_cache: Whether to use cached version if available
			parser: PDF parser to use ('pypdf2' or 'pdfplumber')
			chunk_by: How to chunk the document ('page' or 'section')
			
		Returns:
			Dictionary mapping document IDs to text chunks
		"""
		print(f"LegalDocumentLoader: Loading PDF '{pdf_path}'...")
		
		if not os.path.exists(pdf_path):
			raise FileNotFoundError(f"PDF file not found: {pdf_path}")
		
		pdf_filename = Path(pdf_path).stem
		
		# Try to load from cache first
		if use_cache:
			cached = self._load_from_cache(pdf_path)
			if cached is not None:
				documents, metadata = cached
				# Merge into instance state
				self.documents.update(documents)
				self.metadata.update(metadata)
				print(f"  - Loaded {len(documents)} chunks from cache (no processing needed)")
				return documents
		
		# Cache miss - process the PDF
		print(f"  - Processing PDF (this may take a moment)...")
		
		# Extract text from PDF
		if parser == 'pdfplumber':
			try:
				pages = self._extract_text_from_pdf_pdfplumber(pdf_path)
			except ImportError:
				print(f"  - pdfplumber not available, falling back to PyPDF2")
				pages = self._extract_text_from_pdf_pypdf2(pdf_path)
		else:
			pages = self._extract_text_from_pdf_pypdf2(pdf_path)
		
		print(f"  - Extracted {len(pages)} pages")
		
		# Create documents and metadata
		documents = {}
		doc_metadata = {}
		
		if chunk_by == 'page':
			# One document per page
			for page_num, text in pages:
				doc_id = f"{pdf_filename}_page_{page_num}"
				documents[doc_id] = text
				
				doc_metadata[doc_id] = {
					'jurisdiction': jurisdiction or 'unknown',
					'source': pdf_filename,
					'file_path': str(pdf_path),
					'type': doc_type,
					'page_number': page_num,
					'total_pages': len(pages),
					'chunk_type': 'page'
				}
		
		elif chunk_by == 'section':
			# Try to detect sections (basic implementation)
			# You can enhance this with more sophisticated section detection
			current_section = []
			section_num = 1
			
			for page_num, text in pages:
				# Simple heuristic: detect section breaks
				# Customize based on your document structure
				lines = text.split('\n')
				
				for line in lines:
					current_section.append(line)
					
					# Check if this line indicates a section break
					# (e.g., "Article 1", "Section 1", "Part I", etc.)
					if self._is_section_header(line):
						# Save current section
						if current_section:
							section_text = '\n'.join(current_section[:-1])  # Exclude the header
							if section_text.strip():
								doc_id = f"{pdf_filename}_section_{section_num}"
								documents[doc_id] = section_text
								
								doc_metadata[doc_id] = {
									'jurisdiction': jurisdiction or 'unknown',
									'source': pdf_filename,
									'file_path': str(pdf_path),
									'type': doc_type,
									'section_number': section_num,
									'chunk_type': 'section',
									'page_start': page_num
								}
								
								section_num += 1
						
						# Start new section
						current_section = [line]
			
			# Save last section
			if current_section:
				section_text = '\n'.join(current_section)
				if section_text.strip():
					doc_id = f"{pdf_filename}_section_{section_num}"
					documents[doc_id] = section_text
					
					doc_metadata[doc_id] = {
						'jurisdiction': jurisdiction or 'unknown',
						'source': pdf_filename,
						'file_path': str(pdf_path),
						'type': doc_type,
						'section_number': section_num,
						'chunk_type': 'section'
					}
		
		# Save to cache
		if use_cache:
			self._save_to_cache(pdf_path, documents, doc_metadata)
		
		# Merge into instance state
		self.documents.update(documents)
		self.metadata.update(doc_metadata)
		
		print(f"  - ✓ Processed {len(documents)} chunks from PDF")
		return documents
	
	def _is_section_header(self, line: str) -> bool:
		"""
		Detect if a line is a section header.
		
		This is a simple heuristic - customize based on your document structure.
		
		Args:
			line: Text line to check
			
		Returns:
			True if line appears to be a section header
		"""
		import re
		
		line = line.strip()
		
		# Common patterns for section headers
		patterns = [
			r'^(Article|ARTICLE)\s+\d+',
			r'^(Section|SECTION)\s+\d+',
			r'^(Part|PART)\s+[IVX\d]+',
			r'^(Chapter|CHAPTER)\s+\d+',
			r'^(Title|TITLE)\s+\d+',
			r'^\d+\.\s+[A-Z]',  # "1. TITLE"
			r'^[A-Z\s]{5,}$',  # All caps lines (potential titles)
		]
		
		for pattern in patterns:
			if re.search(pattern, line):
				return True
		
		return False
	
	def load_from_directory(
		self,
		directory: str,
		jurisdiction: Optional[str] = None,
		recursive: bool = True
	) -> Dict[str, str]:
		"""
		Load all JSON documents from a directory.
		
		Args:
			directory: Path to directory
			jurisdiction: Optional jurisdiction tag for all docs
			recursive: Whether to search subdirectories
			
		Returns:
			Dictionary of loaded documents
		"""
		print(f"LegalDocumentLoader: Loading documents from directory '{directory}'...")
		
		path = Path(directory)
		if not path.exists():
			raise FileNotFoundError(f"Directory not found: {directory}")
		
		documents = {}
		pattern = "**/*.json" if recursive else "*.json"
		
		for json_file in path.glob(pattern):
			try:
				with open(json_file, 'r', encoding='utf-8') as f:
					data = json.load(f)
				
				# Handle different JSON structures
				if isinstance(data, dict):
					for key, value in data.items():
						doc_id = f"{json_file.stem}_{key}"
						documents[doc_id] = str(value)
						
						self.metadata[doc_id] = {
							'jurisdiction': jurisdiction or 'unknown',
							'source': json_file.name,
							'file_path': str(json_file),
							'type': 'document'
						}
				
				print(f"  - Loaded {len(data)} documents from {json_file.name}")
				
			except Exception as e:
				print(f"  - Warning: Could not load {json_file.name}: {e}")
		
		print(f"LegalDocumentLoader: Loaded total of {len(documents)} documents from directory")
		return documents
	
	def load_all_datasets(
		self,
		mexico_articles_path: str = './data/articlesFull.json',
		mexico_training_path: str = './data/TrainingData.json',
		nepal_constitution_path: str = './data/nepal_constitution.json',
		nepal_constitution_pdf_path: str = './data/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf',
		load_pdfs: bool = True,
		use_cache: bool = True
	) -> Dict[str, str]:
		"""
		Convenience method to load all available datasets.
		
		Args:
			mexico_articles_path: Path to Mexico Civil Law articles
			mexico_training_path: Path to Mexico training data
			nepal_constitution_path: Path to Nepal Constitution JSON
			nepal_constitution_pdf_path: Path to Nepal Constitution PDF
			load_pdfs: Whether to load PDF files
			use_cache: Whether to use cached PDF data
			
		Returns:
			Combined dictionary of all documents
		"""
		print("\n" + "="*80)
		print("Loading All Legal Datasets")
		print("="*80)
		
		all_documents = {}
		
		# Load Mexico Civil Law
		try:
			mexico_articles = self.load_mexico_civil_law(mexico_articles_path)
			all_documents.update(mexico_articles)
		except Exception as e:
			print(f"Warning: Could not load Mexico Civil Law: {e}")
		
		# Load Mexico Training Data
		try:
			mexico_training = self.load_mexico_training_data(mexico_training_path)
			all_documents.update(mexico_training)
		except Exception as e:
			print(f"Warning: Could not load Mexico Training Data: {e}")
		
		# Load Nepal Constitution JSON (if exists)
		try:
			nepal_const = self.load_nepal_constitution(nepal_constitution_path)
			all_documents.update(nepal_const)
		except Exception as e:
			print(f"Warning: Could not load Nepal Constitution JSON: {e}")
		
		# Load Nepal Constitution PDF (if enabled and exists)
		if load_pdfs and os.path.exists(nepal_constitution_pdf_path):
			try:
				print(f"\n📄 Processing Nepal Constitution PDF...")
				nepal_pdf = self.load_pdf(
					pdf_path=nepal_constitution_pdf_path,
					jurisdiction='nepal',
					doc_type='constitution',
					use_cache=use_cache,
					parser='pypdf2',  # Use 'pdfplumber' for better quality
					chunk_by='page'  # or 'section' for semantic chunking
				)
				all_documents.update(nepal_pdf)
			except Exception as e:
				print(f"Warning: Could not load Nepal Constitution PDF: {e}")
				print(f"  Tip: Install PDF libraries with: pip install pypdf2 pdfplumber")
		
		# Store for later access
		self.documents = all_documents
		
		print("="*80)
		print(f"Total Documents Loaded: {len(all_documents)}")
		self._print_load_summary()
		print("="*80 + "\n")
		
		return all_documents
	
	def _print_load_summary(self):
		"""Print summary of loaded documents."""
		stats = self.get_statistics()
		
		print(f"\n📊 Document Statistics:")
		print(f"   • Total Documents: {stats['total_documents']}")
		
		if stats['by_jurisdiction']:
			print(f"\n   By Jurisdiction:")
			for jurisdiction, count in sorted(stats['by_jurisdiction'].items()):
				print(f"     - {jurisdiction.capitalize()}: {count}")
		
		if stats['by_type']:
			print(f"\n   By Type:")
			for doc_type, count in sorted(stats['by_type'].items()):
				print(f"     - {doc_type.replace('_', ' ').title()}: {count}")
	
	def get_metadata(self, doc_id: str) -> Dict:
		"""Get metadata for a specific document."""
		return self.metadata.get(doc_id, {})
	
	def get_documents_by_jurisdiction(self, jurisdiction: str) -> Dict[str, str]:
		"""Get all documents for a specific jurisdiction."""
		filtered = {}
		for doc_id, text in self.documents.items():
			meta = self.metadata.get(doc_id, {})
			if meta.get('jurisdiction') == jurisdiction:
				filtered[doc_id] = text
		return filtered
	
	def get_statistics(self) -> Dict:
		"""Get statistics about loaded documents."""
		stats = {
			'total_documents': len(self.documents),
			'by_jurisdiction': {},
			'by_type': {},
			'by_source': {}
		}
		
		for doc_id, meta in self.metadata.items():
			jurisdiction = meta.get('jurisdiction', 'unknown')
			doc_type = meta.get('type', 'unknown')
			source = meta.get('source', 'unknown')
			
			stats['by_jurisdiction'][jurisdiction] = stats['by_jurisdiction'].get(jurisdiction, 0) + 1
			stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
			stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
		
		return stats
