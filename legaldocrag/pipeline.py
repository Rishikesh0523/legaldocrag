from .config import PipelineConfig
from .preprocessing import PreprocessingEngine
from .knowledge import KnowledgeGraphExpander
from .retrieval import HybridRetriever
from .reranker import Reranker
from .generator import Generator
from .api_generator import APIGenerator
from .corrective import CorrectiveLayer
from .citations import CitationParser
from .jurisdiction import JurisdictionDetector
from .reliability import ProductionReliabilityChecker
from .document_loader import LegalDocumentLoader
from .monitoring import MetricsTracker

import time
from typing import Dict, List, Optional


class LegalRAGPipeline:
	"""
	Production-ready Legal RAG Pipeline with comprehensive reliability features.
	
	Features:
	- Hybrid retrieval (BM25 + FAISS)
	- Jurisdiction detection and filtering
	- Answerability checking (refuses to answer when confidence is low)
	- Faithfulness validation (prevents hallucinations)
	- Support for both local and API-based models (OpenAI, Claude)
	- Comprehensive monitoring and metrics
	- Adaptive query reformulation
	- Citation enforcement
	"""
	
	def __init__(self, config: PipelineConfig):
		print("\n" + "="*80)
		print("Initializing Production Legal RAG Pipeline")
		print("="*80 + "\n")
		
		self.config = config
		self.documents = {}
		self.jurisdiction_map = {}

		# Initialize all components
		print("Loading core components...")
		
		self.preprocessor = PreprocessingEngine()
		self.kg_expander = KnowledgeGraphExpander()
		self.retriever = HybridRetriever(self.config.EMBEDDING_MODEL)
		self.reranker = Reranker(self.config.RERANKER_MODEL)
		
		# Initialize generator based on type (local or API)
		if self.config.GENERATOR_TYPE == 'local':
			print(f"Initializing LOCAL generator: {self.config.BASE_LLM_MODEL}")
			self.generator = Generator(
				self.config.BASE_LLM_MODEL,
				self.config.LORA_ADAPTER_PATH,
				self.config.BNB_CONFIG,
			)
		elif self.config.GENERATOR_TYPE == 'openai':
			print(f"Initializing OPENAI generator: {self.config.OPENAI_MODEL}")
			self.generator = APIGenerator(
				provider='openai',
				model_name=self.config.OPENAI_MODEL,
				api_key=self.config.OPENAI_API_KEY,
				timeout=self.config.API_TIMEOUT_SECONDS,
				max_retries=self.config.API_MAX_RETRIES
			)
		elif self.config.GENERATOR_TYPE == 'claude':
			print(f"Initializing CLAUDE generator: {self.config.CLAUDE_MODEL}")
			self.generator = APIGenerator(
				provider='claude',
				model_name=self.config.CLAUDE_MODEL,
				api_key=self.config.ANTHROPIC_API_KEY,
				timeout=self.config.API_TIMEOUT_SECONDS,
				max_retries=self.config.API_MAX_RETRIES
			)
		else:
			raise ValueError(f"Unsupported generator type: {self.config.GENERATOR_TYPE}")
		
		self.corrective_layer = CorrectiveLayer(self.generator)
		self.citation_parser = CitationParser()
		
		# Initialize production features
		print("Loading production features...")
		self.jurisdiction_detector = JurisdictionDetector()
		self.reliability_checker = ProductionReliabilityChecker(
			self.retriever.embedding_model
		)
		self.document_loader = LegalDocumentLoader()
		self.metrics_tracker = MetricsTracker()

		self.audit_log = []
		
		print("\n" + "="*80)
		print("Pipeline Initialization Complete")
		print("="*80 + "\n")

	def process_documents(self, documents: Dict[str, str] = None, load_from_datasets: bool = True):
		"""
		Loads and indexes legal documents.
		
		Args:
			documents: Optional custom documents dictionary
			load_from_datasets: If True, loads from configured datasets (Mexico, Nepal, etc.)
		"""
		print("\n--- Starting Document Processing ---")
		
		if load_from_datasets:
			# Load from configured datasets
			self.documents = self.document_loader.load_all_datasets(
				mexico_articles_path=self.config.MEXICO_CIVIL_LAW_PATH,
				mexico_training_path=self.config.MEXICO_TRAINING_DATA_PATH,
				nepal_constitution_path=self.config.NEPAL_CONSTITUTION_PATH
			)
		elif documents:
			self.documents = documents
			# Extract metadata for custom documents
			for doc_id, text in documents.items():
				detected_jurisdictions = self.jurisdiction_detector.detect_from_document(text, doc_id)
				self.document_loader.metadata[doc_id] = {
					'jurisdiction': detected_jurisdictions[0] if detected_jurisdictions else 'unknown',
					'jurisdictions': detected_jurisdictions,
					'type': 'custom'
				}
		else:
			raise ValueError("Either provide documents or set load_from_datasets=True")
		
		# Analyze jurisdictions across corpus
		if self.config.ENABLE_JURISDICTION_DETECTION:
			self.jurisdiction_map = self.jurisdiction_detector.analyze_corpus_jurisdictions(
				self.documents
			)
		
		# Build search indexes
		self.retriever.build_indexes(self.documents)
		
		# Print statistics
		stats = self.document_loader.get_statistics()
		print("\n--- Document Statistics ---")
		print(f"Total documents: {stats['total_documents']}")
		print(f"By jurisdiction: {stats['by_jurisdiction']}")
		print(f"By type: {stats['by_type']}")
		print("--- Document Processing Complete ---\n")

	def answer_query(self, query: str, target_jurisdictions: Optional[List[str]] = None):
		"""
		Main method to answer a user query with production-ready reliability.
		
		Args:
			query: User's legal question
			target_jurisdictions: Optional list of jurisdictions to filter by
			
		Returns:
			Dictionary containing answer, sources, metrics, and status
		"""
		print(f"\n{'='*80}")
		print(f"Processing Query: '{query}'")
		print(f"{'='*80}\n")
		
		# Start metrics tracking
		query_id = self.metrics_tracker.log_query_start(query)
		start_time = time.time()
		
		self.audit_log.append(f"QUERY_ID: {query_id}")
		self.audit_log.append(f"INITIAL_QUERY: {query}")
		
		try:
			# =====================================================================
			# PHASE 1: Query Understanding & Jurisdiction Detection
			# =====================================================================
			print("--- Phase 1: Query Understanding ---")
			
			# Extract entities
			entities = self.preprocessor.extract_entities(query)
			
			# Detect jurisdiction from query if not specified
			if target_jurisdictions is None and self.config.ENABLE_JURISDICTION_DETECTION:
				target_jurisdictions = self.jurisdiction_detector.detect_from_query(query)
				self.audit_log.append(f"DETECTED_JURISDICTIONS: {target_jurisdictions}")
			
			# Expand query with knowledge graph
			expanded_query = self.kg_expander.expand_query(query, entities)
			self.audit_log.append(f"EXPANDED_QUERY: {expanded_query}")
			
			# =====================================================================
			# PHASE 2: Adaptive Retrieval Loop
			# =====================================================================
			print("\n--- Phase 2: Adaptive Retrieval ---")
			
			reranked_docs = []
			context_is_sufficient = False
			current_loop = 0
			retrieval_start = time.time()

			while not context_is_sufficient and current_loop < self.config.MAX_RETRIEVAL_LOOPS:
				current_loop += 1
				print(f"\n  Retrieval Attempt #{current_loop}")
				self.audit_log.append(f"LOOP_{current_loop}_QUERY: {expanded_query}")

				# Hybrid Retrieval (BM25 + FAISS)
				candidate_ids = self.retriever.retrieve(
					expanded_query,
					self.config.TOP_K_BM25,
					self.config.TOP_K_FAISS,
				)

				if not candidate_ids:
					print("  ⚠ No documents retrieved.")
					break
				
				# Jurisdiction filtering
				if target_jurisdictions and self.config.ENABLE_METADATA_FILTERING:
					print(f"  Filtering by jurisdictions: {target_jurisdictions}")
					candidate_ids = self.jurisdiction_detector.filter_by_jurisdiction(
						candidate_ids,
						self.jurisdiction_map,
						target_jurisdictions,
						require_all=False
					)
					
					if not candidate_ids:
						print("  ⚠ No documents match jurisdiction filter.")
						break

				# Rerank candidates
				reranked_docs = self.reranker.rerank(query, self.documents, candidate_ids)
				self.audit_log.append(f"LOOP_{current_loop}_RERANKED: {[d['id'] for d in reranked_docs[:5]]}")

				# Check sufficiency
				if reranked_docs:
					top_score = reranked_docs[0]['score']
					if top_score >= self.config.RERANKER_CONFIDENCE_THRESHOLD:
						context_is_sufficient = True
						print(f"  ✓ Context sufficient (score: {top_score:.4f})")
					else:
						print(f"  ⚠ Low confidence (score: {top_score:.4f}). Reformulating...")
						# Query reformulation
						expanded_query = self._reformulate_query(expanded_query, current_loop)
				else:
					print("  ⚠ No documents after reranking.")
					break
			
			retrieval_time = (time.time() - retrieval_start) * 1000
			
			# Log retrieval metrics
			if reranked_docs:
				self.metrics_tracker.log_retrieval_metrics(
					query_id,
					len(candidate_ids) if candidate_ids else 0,
					len(reranked_docs),
					reranked_docs[0]['score'],
					retrieval_time
				)
			
			# =====================================================================
			# PHASE 3: Answerability Check (Production Reliability)
			# =====================================================================
			print("\n--- Phase 3: Answerability Check ---")
			
			# Prepare context
			final_docs = reranked_docs[: self.config.TOP_K_RERANKED] if reranked_docs else []
			context = "\n\n".join([
				f"[{doc['id']}]:\n{doc['text']}" 
				for doc in final_docs
			])
			
			# Check if we should attempt to answer
			if self.config.ENABLE_ANSWERABILITY_CHECK:
				answerability_result = self.reliability_checker.check_before_generation(
					query,
					context,
					reranked_docs if reranked_docs else [],
					self.config.ANSWERABILITY_THRESHOLD
				)
				
				if not answerability_result['should_answer']:
					print(f"  ✗ Cannot answer: {answerability_result['reason']}")
					
					# Handle fallback strategy
					if self.config.FALLBACK_STRATEGY == 'decline':
						return self._decline_to_answer(
							query, query_id, answerability_result,
							time.time() - start_time
						)
					elif self.config.FALLBACK_STRATEGY == 'base_model':
						print("  → Fallback: Using base model without context")
						context = ""  # Clear context to use general knowledge
					# 'retry' strategy would loop again, but we've exhausted loops
				else:
					print(f"  ✓ Answerable (confidence: {answerability_result['confidence']:.4f})")
			
			self.audit_log.append(f"FINAL_CONTEXT_DOCS: {[d['id'] for d in final_docs]}")
			
			# =====================================================================
			# PHASE 4: Generation
			# =====================================================================
			print("\n--- Phase 4: Answer Generation ---")
			
			generation_start = time.time()
			generated_answer = self.generator.generate(query, context)
			generation_time = (time.time() - generation_start) * 1000
			
			self.audit_log.append(f"GENERATED_ANSWER: {generated_answer[:200]}...")
			
			# Log generation metrics
			tokens_used = None
			if hasattr(self.generator, 'total_tokens'):
				tokens_used = self.generator.total_tokens
			
			self.metrics_tracker.log_generation_metrics(
				query_id,
				tokens_used,
				generation_time,
				self.config.GENERATOR_TYPE
			)
			
			# =====================================================================
			# PHASE 5: Quality Assurance & Faithfulness Check
			# =====================================================================
			print("\n--- Phase 5: Quality Assurance ---")
			
			# Check faithfulness
			faithfulness_result = None
			if self.config.ENABLE_FAITHFULNESS_CHECK and context:
				faithfulness_result = self.reliability_checker.check_after_generation(
					generated_answer,
					context,
					self.config.FAITHFULNESS_THRESHOLD,
					self.config.REQUIRE_CITATIONS
				)
				
				self.metrics_tracker.log_faithfulness_check(
					query_id,
					faithfulness_result['is_faithful'],
					faithfulness_result['faithfulness_score'],
					faithfulness_result.get('issues', [])
				)
				
				if not faithfulness_result['is_faithful']:
					print(f"  ⚠ Faithfulness issues detected:")
					for issue in faithfulness_result.get('issues', []):
						print(f"    - {issue}")
					
					# Decide whether to return answer or decline
					if len(faithfulness_result.get('issues', [])) > 2:
						print("  ✗ Too many faithfulness issues. Declining to answer.")
						return self._decline_to_answer(
							query, query_id, 
							{'reason': 'Generated answer failed faithfulness check'},
							time.time() - start_time
						)
				else:
					print(f"  ✓ Answer is faithful (score: {faithfulness_result['faithfulness_score']:.4f})")
			
			# Legacy correction check (kept for backwards compatibility)
			correction_result = self.corrective_layer.check(generated_answer, context)
			self.audit_log.append(f"CORRECTION_RESULT: {correction_result}")

			# Add citation links
			final_answer = self.citation_parser.link_citations(generated_answer, final_docs)

			# =====================================================================
			# PHASE 6: Finalization & Metrics
			# =====================================================================
			total_time = (time.time() - start_time) * 1000
			
			# Log completion
			self.metrics_tracker.log_query_complete(
				query_id,
				success=True,
				answer_length=len(final_answer),
				total_latency_ms=total_time
			)
			
			# Build output
			output = {
				'success': True,
				'query_id': query_id,
				'query': query,
				'answer': final_answer,
				'sources': final_docs,
				'jurisdictions_detected': target_jurisdictions,
				'metrics': {
					'retrieval_time_ms': retrieval_time,
					'generation_time_ms': generation_time,
					'total_time_ms': total_time,
					'num_sources': len(final_docs),
					'tokens_used': tokens_used
				},
				'quality_checks': {
					'faithfulness': faithfulness_result,
					'correction': correction_result
				}
			}

			print(f"\n{'='*80}")
			print("✓ Query Processing Complete")
			print(f"  Total time: {total_time:.2f}ms")
			print(f"{'='*80}\n")
			
			return output
		
		except Exception as e:
			# Handle errors gracefully
			print(f"\n✗ Error processing query: {e}")
			
			total_time = (time.time() - start_time) * 1000
			self.metrics_tracker.log_query_complete(
				query_id,
				success=False,
				answer_length=0,
				total_latency_ms=total_time,
				failure_reason=str(e)
			)
			
			return {
				'success': False,
				'query_id': query_id,
				'query': query,
				'error': str(e),
				'answer': "An error occurred while processing your query. Please try again.",
				'metrics': {
					'total_time_ms': total_time
				}
			}
	
	def _reformulate_query(self, query: str, attempt: int) -> str:
		"""
		Reformulate query for better retrieval.
		
		Simple strategy: add legal context terms based on attempt number.
		"""
		reformulations = [
			" legal provision statute",
			" case law precedent ruling",
			" article section clause"
		]
		
		if attempt <= len(reformulations):
			return query + reformulations[attempt - 1]
		return query
	
	def _decline_to_answer(
		self,
		query: str,
		query_id: str,
		reason_info: Dict,
		elapsed_time: float
	) -> Dict:
		"""
		Generate a response when declining to answer.
		"""
		print("\n✗ Declining to answer query")
		
		self.metrics_tracker.log_query_complete(
			query_id,
			success=False,
			answer_length=0,
			total_latency_ms=elapsed_time * 1000,
			failure_reason=reason_info.get('reason', 'Insufficient confidence')
		)
		
		return {
			'success': False,
			'query_id': query_id,
			'query': query,
			'answer': (
				"I cannot provide a reliable answer to this question based on the available "
				"legal documents. The retrieved information does not have sufficient relevance "
				"or confidence to ensure an accurate response. Please try rephrasing your "
				"question or provide more specific details."
			),
			'declined': True,
			'reason': reason_info.get('reason'),
			'sources': [],
			'metrics': {
				'total_time_ms': elapsed_time * 1000
			}
		}
	
	def get_metrics_summary(self):
		"""Get summary of pipeline metrics."""
		return self.metrics_tracker.get_summary_statistics()
	
	def print_metrics(self):
		"""Print metrics summary to console."""
		self.metrics_tracker.print_summary()
	
	def export_metrics(self, filepath: Optional[str] = None):
		"""Export metrics to JSON file."""
		return self.metrics_tracker.export_metrics(filepath)
