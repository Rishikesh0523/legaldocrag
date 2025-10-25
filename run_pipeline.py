import os
import torch
from legaldocrag import PipelineConfig, LegalRAGPipeline


def main():
	print("=" * 80)
	print("Legal RAG Pipeline v2.0 - Production-Ready Legal Document Q&A")
	print("=" * 80)
	
	# Display GPU information
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(0)
		gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
		print(f"\n✓ GPU Detected: {gpu_name}")
		print(f"✓ Total VRAM: {gpu_memory:.2f} GB")
	else:
		print("\n⚠ No GPU detected. Running on CPU (slower).")
	
	# =========================================================================
	# Configuration Display
	# =========================================================================
	print(f"\n📚 Model Configuration:")
	print(f"   • Generator Type: {PipelineConfig.GENERATOR_TYPE.upper()}")
	
	if PipelineConfig.GENERATOR_TYPE == 'local':
		print(f"   • LLM Model: {PipelineConfig.BASE_LLM_MODEL}")
		print(f"   • Embedding Model: {PipelineConfig.EMBEDDING_MODEL}")
		print(f"   • Reranker Model: {PipelineConfig.RERANKER_MODEL}")
		
		# Show which model is active
		if 'TinyLlama' in PipelineConfig.BASE_LLM_MODEL:
			print(f"\n   ℹ️  Using TinyLlama-1.1B (Optimized for 6GB GPU)")
			print(f"   ℹ️  For larger models, see MODEL_SELECTION.md")
		elif 'Llama-2-7b' in PipelineConfig.BASE_LLM_MODEL:
			print(f"\n   ⚠️  Using LLaMA-2-7B (Requires 16GB+ GPU)")
		elif 'Llama-2-13b' in PipelineConfig.BASE_LLM_MODEL:
			print(f"\n   ⚠️  Using LLaMA-2-13B (Requires 24GB+ GPU)")
	
	elif PipelineConfig.GENERATOR_TYPE == 'openai':
		print(f"   • OpenAI Model: {PipelineConfig.OPENAI_MODEL}")
		api_key_status = "✓ Set" if PipelineConfig.OPENAI_API_KEY else "✗ Not Set"
		print(f"   • API Key: {api_key_status}")
	
	elif PipelineConfig.GENERATOR_TYPE == 'claude':
		print(f"   • Claude Model: {PipelineConfig.CLAUDE_MODEL}")
		api_key_status = "✓ Set" if PipelineConfig.ANTHROPIC_API_KEY else "✗ Not Set"
		print(f"   • API Key: {api_key_status}")
	
	print(f"\n🔒 Production Features:")
	print(f"   • Answerability Checking: {PipelineConfig.ENABLE_ANSWERABILITY_CHECK}")
	print(f"   • Faithfulness Validation: {PipelineConfig.ENABLE_FAITHFULNESS_CHECK}")
	print(f"   • Jurisdiction Detection: {PipelineConfig.ENABLE_JURISDICTION_DETECTION}")
	print(f"   • Fallback Strategy: {PipelineConfig.FALLBACK_STRATEGY}")
	print(f"   • Citation Required: {PipelineConfig.REQUIRE_CITATIONS}")
	
	print("=" * 80)
	
	# Create a dummy LoRA adapter directory for demonstration if it doesn't exist
	if PipelineConfig.GENERATOR_TYPE == 'local' and not os.path.exists(PipelineConfig.LORA_ADAPTER_PATH):
		os.makedirs(PipelineConfig.LORA_ADAPTER_PATH, exist_ok=True)
		print(f"Created dummy LoRA adapter directory at '{PipelineConfig.LORA_ADAPTER_PATH}'.")
		print("NOTE: The generator will use the base LLaMA model as no real adapters are present.")

	# =========================================================================
	# Initialize Pipeline
	# =========================================================================
	config = PipelineConfig()
	pipeline = LegalRAGPipeline(config)

	# =========================================================================
	# Load and Process Documents
	# =========================================================================
	# Option 1: Load from datasets (recommended)
	pipeline.process_documents(load_from_datasets=True)
	
	# Option 2: Use custom documents (alternative)
	# custom_documents = {
	# 	"doc_id_1": "Article 90 of the Japanese Civil Law states...",
	# 	"doc_id_2": "In the case of Smith v. Jones...",
	# }
	# pipeline.process_documents(documents=custom_documents, load_from_datasets=False)

	# =========================================================================
	# Test Queries - Demonstrating Different Features
	# =========================================================================
	
	test_queries = [
		# Query 1: Mexico jurisdiction (from COLIEE dataset)
		{
			"query": "What does Mexican Civil Law say about juridical acts against public policy?",
			"description": "Testing Mexico jurisdiction detection and article retrieval"
		},
		
		# Query 2: Contract termination (general)
		{
			"query": "What is required for contract termination according to the Federal Court of Canada?",
			"description": "Testing Canadian jurisdiction and general contract law"
		},
		
		# Query 3: Unanswerable query (tests reliability)
		{
			"query": "What is the weather like in Tokyo?",
			"description": "Testing answerability check - should decline to answer"
		},
		
		# Query 4: Nepal constitution (if available)
		{
			"query": "What rights are guaranteed in the Constitution of Nepal?",
			"description": "Testing Nepal jurisdiction detection"
		}
	]
	
	# Run queries
	for i, test in enumerate(test_queries, 1):
		print(f"\n{'#'*80}")
		print(f"TEST QUERY {i}/{len(test_queries)}")
		print(f"Description: {test['description']}")
		print(f"{'#'*80}")
		
		result = pipeline.answer_query(test['query'])
		
		# Display Results
		print("\n" + "="*80)
		print("RESULT")
		print("="*80)
		print(f"Status: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}")
		print(f"Query: {result['query']}")
		
		if result.get('declined'):
			print(f"\n⚠ Answer Declined")
			print(f"Reason: {result.get('reason', 'Unknown')}")
		
		print(f"\nAnswer:")
		print(result['answer'])
		
		if result.get('sources'):
			print(f"\nSources ({len(result['sources'])} documents):")
			for source in result['sources'][:3]:  # Show top 3
				jurisdiction = pipeline.document_loader.get_metadata(source['id']).get('jurisdiction', 'unknown')
				print(f"  - ID: {source['id']}, Score: {source['score']:.4f}, Jurisdiction: {jurisdiction}")
		
		if result.get('jurisdictions_detected'):
			print(f"\nDetected Jurisdictions: {result['jurisdictions_detected']}")
		
		if result.get('metrics'):
			metrics = result['metrics']
			print(f"\nPerformance Metrics:")
			print(f"  - Retrieval Time: {metrics.get('retrieval_time_ms', 0):.2f} ms")
			print(f"  - Generation Time: {metrics.get('generation_time_ms', 0):.2f} ms")
			print(f"  - Total Time: {metrics.get('total_time_ms', 0):.2f} ms")
			if metrics.get('tokens_used'):
				print(f"  - Tokens Used: {metrics['tokens_used']}")
		
		if result.get('quality_checks', {}).get('faithfulness'):
			faith = result['quality_checks']['faithfulness']
			print(f"\nQuality Checks:")
			print(f"  - Faithful: {faith['is_faithful']}")
			print(f"  - Faithfulness Score: {faith['faithfulness_score']:.4f}")
			if faith.get('issues'):
				print(f"  - Issues: {faith['issues']}")
		
		print("="*80)
	
	# =========================================================================
	# Display Overall Metrics
	# =========================================================================
	print("\n" + "#"*80)
	print("OVERALL PIPELINE METRICS")
	print("#"*80)
	pipeline.print_metrics()
	
	# Export metrics to file
	metrics_file = pipeline.export_metrics()
	print(f"\n✓ Detailed metrics exported to: {metrics_file}")
	
	print("\n" + "="*80)
	print("Pipeline Demo Complete")
	print("="*80)


if __name__ == "__main__":
	main()
