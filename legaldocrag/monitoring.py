"""
Monitoring and Metrics Module

Tracks performance metrics, logs operations, and provides feedback mechanisms
for continuous improvement of the RAG pipeline.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import statistics


class MetricsTracker:
	"""
	Tracks performance metrics for the RAG pipeline.
	
	Monitors:
	- Retrieval metrics (precision, recall, latency)
	- Generation metrics (token usage, latency)
	- End-to-end query processing time
	- Success/failure rates
	"""
	
	def __init__(self, log_dir: str = './logs'):
		"""
		Initialize metrics tracker.
		
		Args:
			log_dir: Directory to store logs and metrics
		"""
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(exist_ok=True, parents=True)
		
		# Setup logging
		self.logger = logging.getLogger('RAGMetrics')
		self.logger.setLevel(logging.INFO)
		
		# File handler for metrics
		log_file = self.log_dir / f'metrics_{datetime.now().strftime("%Y%m%d")}.log'
		fh = logging.FileHandler(log_file)
		fh.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		self.logger.addHandler(fh)
		
		# In-memory metrics
		self.metrics = defaultdict(list)
		self.query_history = []
		
		print(f"MetricsTracker: Initialized. Logging to '{log_file}'")
	
	def log_query_start(self, query: str) -> str:
		"""
		Log the start of query processing.
		
		Args:
			query: The user query
			
		Returns:
			Query ID for tracking
		"""
		query_id = f"q_{int(time.time() * 1000)}"
		
		self.query_history.append({
			'query_id': query_id,
			'query': query,
			'start_time': time.time(),
			'timestamp': datetime.now().isoformat()
		})
		
		self.logger.info(f"Query started: {query_id} - '{query}'")
		return query_id
	
	def log_retrieval_metrics(
		self,
		query_id: str,
		num_retrieved: int,
		num_reranked: int,
		top_score: float,
		latency_ms: float
	):
		"""Log retrieval stage metrics."""
		metrics = {
			'query_id': query_id,
			'num_retrieved': num_retrieved,
			'num_reranked': num_reranked,
			'top_score': top_score,
			'latency_ms': latency_ms,
			'timestamp': datetime.now().isoformat()
		}
		
		self.metrics['retrieval'].append(metrics)
		self.logger.info(
			f"Retrieval - {query_id}: retrieved={num_retrieved}, "
			f"reranked={num_reranked}, top_score={top_score:.4f}, latency={latency_ms:.2f}ms"
		)
	
	def log_generation_metrics(
		self,
		query_id: str,
		tokens_used: Optional[int],
		latency_ms: float,
		model_type: str
	):
		"""Log generation stage metrics."""
		metrics = {
			'query_id': query_id,
			'tokens_used': tokens_used,
			'latency_ms': latency_ms,
			'model_type': model_type,
			'timestamp': datetime.now().isoformat()
		}
		
		self.metrics['generation'].append(metrics)
		
		token_info = f"tokens={tokens_used}" if tokens_used else "tokens=N/A"
		self.logger.info(
			f"Generation - {query_id}: {token_info}, "
			f"latency={latency_ms:.2f}ms, model={model_type}"
		)
	
	def log_query_complete(
		self,
		query_id: str,
		success: bool,
		answer_length: int,
		total_latency_ms: float,
		failure_reason: Optional[str] = None
	):
		"""Log completion of query processing."""
		# Find query in history
		for query_record in self.query_history:
			if query_record['query_id'] == query_id:
				query_record.update({
					'success': success,
					'answer_length': answer_length,
					'total_latency_ms': total_latency_ms,
					'failure_reason': failure_reason,
					'end_time': time.time()
				})
				break
		
		status = "SUCCESS" if success else f"FAILED ({failure_reason})"
		self.logger.info(
			f"Query complete - {query_id}: {status}, "
			f"answer_len={answer_length}, total_latency={total_latency_ms:.2f}ms"
		)
		
		# Track success rate
		self.metrics['success_rate'].append(1 if success else 0)
	
	def log_faithfulness_check(
		self,
		query_id: str,
		is_faithful: bool,
		faithfulness_score: float,
		issues: List[str]
	):
		"""Log faithfulness check results."""
		metrics = {
			'query_id': query_id,
			'is_faithful': is_faithful,
			'faithfulness_score': faithfulness_score,
			'issues': issues,
			'timestamp': datetime.now().isoformat()
		}
		
		self.metrics['faithfulness'].append(metrics)
		self.logger.info(
			f"Faithfulness - {query_id}: faithful={is_faithful}, "
			f"score={faithfulness_score:.4f}, issues={len(issues)}"
		)
	
	def log_user_feedback(
		self,
		query_id: str,
		rating: int,
		feedback_text: Optional[str] = None,
		correct_answer: Optional[str] = None
	):
		"""
		Log user feedback for continuous improvement.
		
		Args:
			query_id: Query identifier
			rating: User rating (1-5)
			feedback_text: Optional text feedback
			correct_answer: Optional correct answer from user
		"""
		feedback = {
			'query_id': query_id,
			'rating': rating,
			'feedback_text': feedback_text,
			'correct_answer': correct_answer,
			'timestamp': datetime.now().isoformat()
		}
		
		self.metrics['user_feedback'].append(feedback)
		self.logger.info(
			f"User Feedback - {query_id}: rating={rating}/5"
		)
		
		# Save feedback to separate file for training
		feedback_file = self.log_dir / 'user_feedback.jsonl'
		with open(feedback_file, 'a', encoding='utf-8') as f:
			f.write(json.dumps(feedback) + '\n')
	
	def get_summary_statistics(self) -> Dict[str, Any]:
		"""Get summary statistics for all metrics."""
		summary = {}
		
		# Success rate
		if self.metrics['success_rate']:
			success_rate = statistics.mean(self.metrics['success_rate']) * 100
			summary['success_rate'] = f"{success_rate:.1f}%"
		
		# Retrieval metrics
		if self.metrics['retrieval']:
			retrieval_latencies = [m['latency_ms'] for m in self.metrics['retrieval']]
			top_scores = [m['top_score'] for m in self.metrics['retrieval']]
			
			summary['retrieval'] = {
				'avg_latency_ms': statistics.mean(retrieval_latencies),
				'median_latency_ms': statistics.median(retrieval_latencies),
				'avg_top_score': statistics.mean(top_scores),
				'num_queries': len(self.metrics['retrieval'])
			}
		
		# Generation metrics
		if self.metrics['generation']:
			gen_latencies = [m['latency_ms'] for m in self.metrics['generation']]
			tokens = [m['tokens_used'] for m in self.metrics['generation'] if m['tokens_used']]
			
			summary['generation'] = {
				'avg_latency_ms': statistics.mean(gen_latencies),
				'median_latency_ms': statistics.median(gen_latencies),
				'total_tokens': sum(tokens) if tokens else 0,
				'num_queries': len(self.metrics['generation'])
			}
		
		# Faithfulness metrics
		if self.metrics['faithfulness']:
			faithful_count = sum(1 for m in self.metrics['faithfulness'] if m['is_faithful'])
			scores = [m['faithfulness_score'] for m in self.metrics['faithfulness']]
			
			summary['faithfulness'] = {
				'faithful_rate': f"{(faithful_count / len(self.metrics['faithfulness']) * 100):.1f}%",
				'avg_score': statistics.mean(scores),
				'num_checks': len(self.metrics['faithfulness'])
			}
		
		# User feedback
		if self.metrics['user_feedback']:
			ratings = [f['rating'] for f in self.metrics['user_feedback']]
			summary['user_feedback'] = {
				'avg_rating': statistics.mean(ratings),
				'num_feedback': len(self.metrics['user_feedback'])
			}
		
		# Total queries
		summary['total_queries'] = len(self.query_history)
		
		return summary
	
	def print_summary(self):
		"""Print summary statistics to console."""
		summary = self.get_summary_statistics()
		
		print("\n" + "="*80)
		print("RAG PIPELINE METRICS SUMMARY")
		print("="*80)
		
		print(f"\nTotal Queries: {summary.get('total_queries', 0)}")
		print(f"Success Rate: {summary.get('success_rate', 'N/A')}")
		
		if 'retrieval' in summary:
			print("\nRetrieval Metrics:")
			print(f"  - Average Latency: {summary['retrieval']['avg_latency_ms']:.2f} ms")
			print(f"  - Median Latency: {summary['retrieval']['median_latency_ms']:.2f} ms")
			print(f"  - Average Top Score: {summary['retrieval']['avg_top_score']:.4f}")
		
		if 'generation' in summary:
			print("\nGeneration Metrics:")
			print(f"  - Average Latency: {summary['generation']['avg_latency_ms']:.2f} ms")
			print(f"  - Total Tokens: {summary['generation']['total_tokens']}")
		
		if 'faithfulness' in summary:
			print("\nFaithfulness Metrics:")
			print(f"  - Faithful Rate: {summary['faithfulness']['faithful_rate']}")
			print(f"  - Average Score: {summary['faithfulness']['avg_score']:.4f}")
		
		if 'user_feedback' in summary:
			print("\nUser Feedback:")
			print(f"  - Average Rating: {summary['user_feedback']['avg_rating']:.2f}/5")
			print(f"  - Total Feedback: {summary['user_feedback']['num_feedback']}")
		
		print("="*80 + "\n")
	
	def export_metrics(self, filepath: Optional[str] = None) -> str:
		"""
		Export all metrics to JSON file.
		
		Args:
			filepath: Optional custom filepath
			
		Returns:
			Path to exported file
		"""
		if filepath is None:
			filepath = self.log_dir / f'metrics_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
		
		export_data = {
			'summary': self.get_summary_statistics(),
			'detailed_metrics': dict(self.metrics),
			'query_history': self.query_history,
			'export_time': datetime.now().isoformat()
		}
		
		with open(filepath, 'w', encoding='utf-8') as f:
			json.dump(export_data, f, indent=2)
		
		print(f"MetricsTracker: Exported metrics to '{filepath}'")
		return str(filepath)
