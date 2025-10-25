"""
Answerability and Faithfulness Checking Module

This module provides production-ready reliability features:
1. Answerability detection - determines if a question CAN be answered from the context
2. Faithfulness checking - ensures answers are grounded in retrieved context
3. Hallucination detection - prevents fabricated information
"""

from typing import Dict, List, Optional, Tuple
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np


class AnswerabilityChecker:
	"""
	Determines whether a question can be answered from available context.
	
	Uses semantic similarity and keyword overlap to assess answerability.
	"""
	
	def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
		"""
		Initialize answerability checker.
		
		Args:
			embedding_model: Optional pre-loaded embedding model for efficiency
		"""
		self.embedding_model = embedding_model
		print("AnswerabilityChecker: Initialized")
	
	def can_answer(
		self,
		query: str,
		context: str,
		reranked_docs: List[Dict],
		threshold: float = 0.4
	) -> Tuple[bool, float, str]:
		"""
		Determine if query can be answered from context.
		
		Args:
			query: User's question
			context: Retrieved context
			reranked_docs: List of reranked documents with scores
			threshold: Minimum confidence threshold
			
		Returns:
			Tuple of (can_answer: bool, confidence: float, reason: str)
		"""
		if not context or not context.strip():
			return False, 0.0, "No context retrieved"
		
		if not reranked_docs:
			return False, 0.0, "No documents found"
		
		# Check 1: Reranker confidence
		top_score = reranked_docs[0]['score'] if reranked_docs else 0.0
		
		if top_score < threshold:
			return False, top_score, f"Top document score ({top_score:.3f}) below threshold ({threshold})"
		
		# Check 2: Keyword overlap
		query_keywords = set(self._extract_keywords(query))
		context_keywords = set(self._extract_keywords(context))
		
		if query_keywords:
			keyword_overlap = len(query_keywords.intersection(context_keywords)) / len(query_keywords)
		else:
			keyword_overlap = 0.0
		
		# Check 3: Semantic similarity (if embedding model available)
		semantic_sim = 0.0
		if self.embedding_model:
			try:
				query_emb = self.embedding_model.encode([query], convert_to_tensor=True)
				context_emb = self.embedding_model.encode([context], convert_to_tensor=True)
				semantic_sim = float(util.cos_sim(query_emb, context_emb)[0][0])
			except Exception as e:
				print(f"AnswerabilityChecker: Warning - semantic similarity failed: {e}")
		
		# Combine signals
		# Weight: 50% reranker score, 30% semantic similarity, 20% keyword overlap
		combined_confidence = (
			0.5 * top_score +
			0.3 * semantic_sim +
			0.2 * keyword_overlap
		)
		
		can_answer = combined_confidence >= threshold
		
		reason = (
			f"Combined confidence: {combined_confidence:.3f} "
			f"(reranker: {top_score:.3f}, semantic: {semantic_sim:.3f}, keywords: {keyword_overlap:.3f})"
		)
		
		print(f"AnswerabilityChecker: {reason}")
		print(f"AnswerabilityChecker: Can answer = {can_answer}")
		
		return can_answer, combined_confidence, reason
	
	def _extract_keywords(self, text: str) -> List[str]:
		"""Extract important keywords from text (simple implementation)."""
		# Remove common legal stopwords but keep important terms
		stopwords = {
			'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
			'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'what', 'when', 'where', 'who', 'which', 'how', 'why'
		}
		
		# Tokenize and filter
		words = re.findall(r'\b[a-z]{3,}\b', text.lower())
		keywords = [w for w in words if w not in stopwords]
		
		return keywords


class FaithfulnessChecker:
	"""
	Ensures generated answers are faithful to the retrieved context.
	
	Detects hallucinations and unsupported claims by checking answer-context alignment.
	"""
	
	def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
		"""
		Initialize faithfulness checker.
		
		Args:
			embedding_model: Optional pre-loaded embedding model
		"""
		self.embedding_model = embedding_model
		print("FaithfulnessChecker: Initialized")
	
	def check_faithfulness(
		self,
		answer: str,
		context: str,
		threshold: float = 0.7,
		require_citations: bool = True
	) -> Dict:
		"""
		Check if answer is faithful to context.
		
		Args:
			answer: Generated answer
			context: Retrieved context
			threshold: Minimum faithfulness score
			require_citations: Whether to require citation markers
			
		Returns:
			Dictionary with faithfulness results
		"""
		result = {
			'is_faithful': True,
			'faithfulness_score': 1.0,
			'issues': [],
			'warnings': [],
		}
		
		# Check 1: Citation presence
		citations = self._extract_citations(answer)
		if require_citations and not citations:
			result['is_faithful'] = False
			result['issues'].append("No citations found in answer")
		elif citations:
			result['citations_found'] = citations
		
		# Check 2: Answer-context semantic similarity
		if self.embedding_model:
			try:
				# Split answer into sentences
				sentences = self._split_sentences(answer)
				
				if sentences:
					sentence_scores = []
					for sentence in sentences:
						# Skip very short sentences or citation-only sentences
						if len(sentence.split()) < 3:
							continue
						
						sent_emb = self.embedding_model.encode([sentence], convert_to_tensor=True)
						ctx_emb = self.embedding_model.encode([context], convert_to_tensor=True)
						sim = float(util.cos_sim(sent_emb, ctx_emb)[0][0])
						sentence_scores.append(sim)
					
					if sentence_scores:
						avg_sim = np.mean(sentence_scores)
						min_sim = np.min(sentence_scores)
						
						result['faithfulness_score'] = float(avg_sim)
						result['min_sentence_score'] = float(min_sim)
						
						# Flag low-confidence sentences
						if min_sim < threshold * 0.8:  # 80% of threshold
							result['warnings'].append(
								f"Some sentences have low context similarity (min: {min_sim:.3f})"
							)
						
						if avg_sim < threshold:
							result['is_faithful'] = False
							result['issues'].append(
								f"Answer faithfulness score ({avg_sim:.3f}) below threshold ({threshold})"
							)
			
			except Exception as e:
				result['warnings'].append(f"Semantic similarity check failed: {e}")
		
		# Check 3: Detect common hallucination patterns
		hallucination_patterns = [
			r'\bas (everyone|we all) know\b',
			r'\bit is common knowledge\b',
			r'\bhistorically\b',
			r'\bin my (opinion|experience)\b',
			r'\bgenerally speaking\b',
		]
		
		for pattern in hallucination_patterns:
			if re.search(pattern, answer, re.IGNORECASE):
				result['warnings'].append(
					f"Detected potential hallucination pattern: '{pattern}'"
				)
		
		# Check 4: Keyword coverage
		answer_keywords = set(self._extract_keywords(answer))
		context_keywords = set(self._extract_keywords(context))
		
		if answer_keywords:
			# How many answer keywords are in context?
			coverage = len(answer_keywords.intersection(context_keywords)) / len(answer_keywords)
			result['keyword_coverage'] = coverage
			
			if coverage < 0.5:  # Less than 50% overlap
				result['warnings'].append(
					f"Low keyword overlap with context ({coverage:.2%})"
				)
		
		print(f"FaithfulnessChecker: Faithfulness score = {result['faithfulness_score']:.3f}")
		print(f"FaithfulnessChecker: Is faithful = {result['is_faithful']}")
		
		return result
	
	def _extract_citations(self, text: str) -> List[str]:
		"""Extract citation markers like [doc_id_1] from text."""
		return re.findall(r'\[([^\]]+)\]', text)
	
	def _split_sentences(self, text: str) -> List[str]:
		"""Simple sentence splitter."""
		# Split on period, exclamation, question mark followed by space/newline
		sentences = re.split(r'[.!?]\s+', text)
		return [s.strip() for s in sentences if s.strip()]
	
	def _extract_keywords(self, text: str) -> List[str]:
		"""Extract important keywords from text."""
		stopwords = {
			'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
			'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'what', 'when', 'where', 'who', 'which', 'how', 'why'
		}
		
		words = re.findall(r'\b[a-z]{3,}\b', text.lower())
		keywords = [w for w in words if w not in stopwords]
		
		return keywords


class ProductionReliabilityChecker:
	"""
	Unified checker that combines answerability and faithfulness.
	
	Provides comprehensive reliability assessment for production RAG systems.
	"""
	
	def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
		"""Initialize with optional embedding model for efficiency."""
		self.answerability_checker = AnswerabilityChecker(embedding_model)
		self.faithfulness_checker = FaithfulnessChecker(embedding_model)
		print("ProductionReliabilityChecker: Initialized")
	
	def check_before_generation(
		self,
		query: str,
		context: str,
		reranked_docs: List[Dict],
		answerability_threshold: float = 0.4
	) -> Dict:
		"""
		Check if query should be answered (pre-generation check).
		
		Returns decision on whether to proceed with generation.
		"""
		can_answer, confidence, reason = self.answerability_checker.can_answer(
			query, context, reranked_docs, answerability_threshold
		)
		
		return {
			'should_answer': can_answer,
			'confidence': confidence,
			'reason': reason,
			'check_type': 'answerability'
		}
	
	def check_after_generation(
		self,
		answer: str,
		context: str,
		faithfulness_threshold: float = 0.7,
		require_citations: bool = True
	) -> Dict:
		"""
		Check answer quality after generation (post-generation check).
		
		Returns faithfulness assessment and potential issues.
		"""
		return self.faithfulness_checker.check_faithfulness(
			answer, context, faithfulness_threshold, require_citations
		)
