"""
Jurisdiction Detection Module

This module handles automatic detection of legal jurisdictions from queries and documents.
It supports filtering documents based on jurisdiction metadata and provides jurisdiction-aware retrieval.
"""

import re
from typing import List, Dict, Optional, Set
from collections import Counter


class JurisdictionDetector:
	"""
	Detects jurisdictions in queries and documents for jurisdiction-aware retrieval.
	
	Supports multiple jurisdictions including Mexico, Nepal, Japan, Canada, and international law.
	"""
	
	def __init__(self):
		"""Initialize the jurisdiction detector with keyword mappings."""
		
		# Jurisdiction keyword patterns (case-insensitive)
		self.jurisdiction_keywords = {
			'mexico': [
				r'\bmexic(o|an)\b',
				r'\bmexico city\b',
				r'\bmexican civil (law|code)\b',
				r'\bcoliee\b',  # COLIEE dataset reference
			],
			'nepal': [
				r'\bnepal(i|ese)?\b',
				r'\bkathmandu\b',
				r'\bnepalese constitution\b',
				r'\bconstitution of nepal\b',
			],
			'japan': [
				r'\bjapan(ese)?\b',
				r'\btokyo\b',
				r'\bjapanese civil (law|code)\b',
			],
			'canada': [
				r'\bcanad(a|ian)\b',
				r'\bfederal court of canada\b',
				r'\bcanadian\b',
				r'\bontario\b',
				r'\bquebec\b',
			],
			'international': [
				r'\binternational\b',
				r'\bunited nations\b',
				r'\bu\.?n\.\b',
				r'\bworld court\b',
				r'\binternational court of justice\b',
				r'\bgeneva convention\b',
			],
		}
		
		# Compile regex patterns for efficient matching
		self.compiled_patterns = {
			jurisdiction: [re.compile(pattern, re.IGNORECASE) 
			               for pattern in patterns]
			for jurisdiction, patterns in self.jurisdiction_keywords.items()
		}
		
		print("JurisdictionDetector: Initialized with support for", 
		      list(self.jurisdiction_keywords.keys()))
	
	def detect_from_query(self, query: str) -> List[str]:
		"""
		Detect jurisdictions mentioned in a user query.
		
		Args:
			query: The user's legal question
			
		Returns:
			List of detected jurisdiction identifiers (e.g., ['mexico', 'nepal'])
		"""
		detected = []
		
		for jurisdiction, patterns in self.compiled_patterns.items():
			for pattern in patterns:
				if pattern.search(query):
					detected.append(jurisdiction)
					break  # One match per jurisdiction is enough
		
		# Remove duplicates while preserving order
		detected = list(dict.fromkeys(detected))
		
		if detected:
			print(f"JurisdictionDetector: Detected jurisdictions in query -> {detected}")
		else:
			print("JurisdictionDetector: No specific jurisdiction detected in query")
		
		return detected
	
	def detect_from_document(self, document: str, doc_id: str = None) -> List[str]:
		"""
		Detect jurisdictions in a legal document.
		
		Args:
			document: The document text
			doc_id: Optional document identifier for logging
			
		Returns:
			List of detected jurisdiction identifiers
		"""
		# Handle None or invalid documents
		if document is None or not isinstance(document, str):
			if doc_id:
				print(f"JurisdictionDetector: Skipping {doc_id} - invalid content (None or not string)")
			return ['international']  # Default fallback
		
		# SMART FIX: Use document ID patterns for known datasets (more accurate than text analysis)
		if doc_id:
			# Mexico documents (from COLIEE dataset and training data)
			if doc_id.startswith('mexico_'):
				return ['mexico']
			
			# Nepal documents (from Constitution PDF and JSON)
			if doc_id.startswith('nepal_') or 'Constitution-of-Nepal' in doc_id:
				return ['nepal']
			
			# Other known patterns
			if doc_id.startswith('canada_'):
				return ['canada']
			if doc_id.startswith('japan_'):
				return ['japan']
		
		# Fallback: Use text pattern matching (but this can be inaccurate)
		detected = []
		
		for jurisdiction, patterns in self.compiled_patterns.items():
			for pattern in patterns:
				if pattern.search(document):
					detected.append(jurisdiction)
					break
		
		# Remove duplicates
		detected = list(dict.fromkeys(detected))
		
		if doc_id and detected:
			print(f"JurisdictionDetector: Document {doc_id} -> jurisdictions {detected}")
		
		return detected
	
	def analyze_corpus_jurisdictions(self, documents: Dict[str, str]) -> Dict[str, List[str]]:
		"""
		Analyze all documents in corpus to build jurisdiction mapping.
		
		Args:
			documents: Dictionary mapping doc_id to document text
			
		Returns:
			Dictionary mapping doc_id to list of jurisdictions
		"""
		print(f"JurisdictionDetector: Analyzing {len(documents)} documents for jurisdictions...")
		
		jurisdiction_map = {}
		jurisdiction_counts = Counter()
		
		for doc_id, document in documents.items():
			detected = self.detect_from_document(document, doc_id)
			jurisdiction_map[doc_id] = detected
			
			for jurisdiction in detected:
				jurisdiction_counts[jurisdiction] += 1
		
		# Print summary
		print(f"JurisdictionDetector: Corpus analysis complete")
		for jurisdiction, count in jurisdiction_counts.most_common():
			print(f"  - {jurisdiction}: {count} documents")
		
		return jurisdiction_map
	
	def filter_by_jurisdiction(
		self,
		doc_ids: List[str],
		jurisdiction_map: Dict[str, List[str]],
		target_jurisdictions: List[str],
		require_all: bool = False
	) -> List[str]:
		"""
		Filter document IDs by jurisdiction requirements.
		
		Args:
			doc_ids: List of document IDs to filter
			jurisdiction_map: Mapping of doc_id to jurisdictions
			target_jurisdictions: Jurisdictions to filter for
			require_all: If True, document must match ALL jurisdictions; 
			             if False, match ANY jurisdiction
			
		Returns:
			Filtered list of document IDs matching jurisdiction criteria
		"""
		if not target_jurisdictions:
			return doc_ids  # No filtering needed
		
		filtered = []
		target_set = set(target_jurisdictions)
		
		for doc_id in doc_ids:
			doc_jurisdictions = set(jurisdiction_map.get(doc_id, []))
			
			if require_all:
				# Document must have all target jurisdictions
				if target_set.issubset(doc_jurisdictions):
					filtered.append(doc_id)
			else:
				# Document must have at least one target jurisdiction
				if target_set.intersection(doc_jurisdictions):
					filtered.append(doc_id)
				# Also include documents with no detected jurisdiction (may be general)
				elif not doc_jurisdictions:
					filtered.append(doc_id)
		
		print(f"JurisdictionDetector: Filtered {len(doc_ids)} -> {len(filtered)} documents")
		
		return filtered
	
	def suggest_jurisdiction(self, query: str, top_n: int = 3) -> List[tuple]:
		"""
		Suggest most likely jurisdictions for a query with confidence scores.
		
		Args:
			query: The user's query
			top_n: Number of top suggestions to return
			
		Returns:
			List of (jurisdiction, confidence_score) tuples
		"""
		scores = {}
		
		for jurisdiction, patterns in self.compiled_patterns.items():
			score = 0
			for pattern in patterns:
				matches = pattern.findall(query)
				score += len(matches)
			
			if score > 0:
				scores[jurisdiction] = score
		
		# Normalize scores
		total = sum(scores.values())
		if total > 0:
			scores = {j: s/total for j, s in scores.items()}
		
		# Sort by score descending
		suggestions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
		
		return suggestions
	
	def add_custom_jurisdiction(self, name: str, keywords: List[str]):
		"""
		Add a custom jurisdiction with its keywords at runtime.
		
		Args:
			name: Jurisdiction identifier (e.g., 'india')
			keywords: List of regex patterns for detection
		"""
		self.jurisdiction_keywords[name] = keywords
		self.compiled_patterns[name] = [
			re.compile(pattern, re.IGNORECASE) for pattern in keywords
		]
		print(f"JurisdictionDetector: Added custom jurisdiction '{name}'")
