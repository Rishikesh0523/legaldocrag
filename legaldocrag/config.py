import torch
import os
from typing import Optional, Literal

# Try to import BitsAndBytesConfig, but don't fail if unavailable (Windows compatibility)
try:
	from transformers import BitsAndBytesConfig
	BITSANDBYTES_AVAILABLE = True
except:
	BITSANDBYTES_AVAILABLE = False


class PipelineConfig:
	"""Configuration class for the RAG pipeline.
	
	This configuration supports both local models and API-based models (Claude, ChatGPT).
	It includes comprehensive settings for production-ready RAG systems with failure detection,
	jurisdiction handling, and adaptive retrieval strategies.
	"""
	
	# ============================================================================
	# API KEYS FOR EXTERNAL LLM PROVIDERS (Optional)
	# ============================================================================
	# Set these via environment variables for security:
	# - For Claude: export ANTHROPIC_API_KEY="your-key"
	# - For ChatGPT: export OPENAI_API_KEY="your-key"
	
	OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY', None)
	ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY', '<YOUR_ANTHROPIC_API_KEY>')
	
	# Generator model selection: 'local', 'openai', or 'claude'
	# - 'local': Use locally loaded Hugging Face models (TinyLlama, LLaMA-2, etc.)
	# - 'openai': Use OpenAI's ChatGPT API (gpt-4, gpt-3.5-turbo)
	# - 'claude': Use Anthropic's Claude API (claude-3-opus, claude-3-sonnet)
	GENERATOR_TYPE: Literal['local', 'openai', 'claude'] = 'claude'
	
	# Specific model names for API providers
	OPENAI_MODEL: str = 'gpt-4o'  # Options: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
	CLAUDE_MODEL: str = 'claude-3-5-sonnet-20241022'  # Options: claude-3-opus-20240229, claude-3-5-sonnet-20241022, claude-3-haiku-20240307
	
	# ============================================================================
	# EMBEDDING AND RERANKING MODELS
	# ============================================================================
	EMBEDDING_MODEL = 'BAAI/bge-m3'  # ~2.3GB - High quality embeddings
	RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # ~90MB
	
	# ============================================================================
	# GENERATOR LLM (Language Model)
	# ============================================================================
	
	# CURRENTLY USING: TinyLlama-1.1B (Optimized for 6GB GPU)
	# This is a smaller, efficient model that works well on consumer hardware
	# Model size: ~2.2GB | Memory usage: ~4GB with FP16
	BASE_LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
	
	# ---- PLACEHOLDER: Larger Models (for future use with 16GB+ GPU) ----
	# Uncomment one of these when you have sufficient GPU memory:
	
	# Option 1: LLaMA-2-7B (Recommended for production)
	# Requires: 16GB+ GPU VRAM with 4-bit quantization OR 32GB+ for FP16
	# Model size: ~13.5GB | Memory usage: ~8GB (4-bit) or ~28GB (FP16)
	# BASE_LLM_MODEL = 'meta-llama/Llama-2-7b-hf'
	
	# Option 2: LLaMA-2-7B Chat (Better for conversational tasks)
	# BASE_LLM_MODEL = 'meta-llama/Llama-2-7b-chat-hf'
	
	# Option 3: Mistral-7B (High performance, good for legal text)
	# Requires: 16GB+ GPU VRAM
	# Model size: ~14GB | Memory usage: ~9GB (4-bit) or ~28GB (FP16)
	# BASE_LLM_MODEL = 'mistralai/Mistral-7B-v0.1'
	
	# Option 4: LLaMA-2-13B (Best quality, requires powerful GPU)
	# Requires: 24GB+ GPU VRAM with 4-bit quantization OR 48GB+ for FP16
	# Model size: ~26GB | Memory usage: ~16GB (4-bit) or ~52GB (FP16)
	# BASE_LLM_MODEL = 'meta-llama/Llama-2-13b-hf'
	
	# LoRA Adapters (for fine-tuning)
	# Assumes LoRA adapters are trained and saved in this directory
	LORA_ADAPTER_PATH = './lora_adapters/llama2-legal-tuned'
	
	# ============================================================================
	# INDEXING AND RETRIEVAL PARAMETERS
	# ============================================================================
	FAISS_INDEX_PATH = 'legal_faiss.index'
	TOP_K_BM25 = 5      # Number of documents retrieved by BM25 (sparse retrieval)
	TOP_K_FAISS = 5     # Number of documents retrieved by FAISS (dense retrieval)
	TOP_K_RERANKED = 3  # Number of documents after reranking
	
	# Chunk optimization settings
	CHUNK_SIZE = 400  # Target chunk size in tokens (200-500 for optimal context)
	CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context
	
	# ============================================================================
	# DYNAMIC RETRIEVAL LOOP
	# ============================================================================
	RERANKER_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for context
	MAX_RETRIEVAL_LOOPS = 2               # Maximum number of retrieval attempts
	MIN_SIMILARITY_THRESHOLD = 0.3        # Minimum similarity for accepting results
	
	# ============================================================================
	# PRODUCTION RELIABILITY SETTINGS
	# ============================================================================
	# Answerability detection: refuse to answer when confidence is too low
	ENABLE_ANSWERABILITY_CHECK = True
	ANSWERABILITY_THRESHOLD = 0.4  # Below this, decline to answer
	
	# Fallback behavior settings
	FALLBACK_STRATEGY: Literal['decline', 'retry', 'base_model'] = 'retry'
	# - 'decline': Refuse to answer when retrieval fails
	# - 'retry': Reformulate query and try again
	# - 'base_model': Use LLM without retrieved context (uses general knowledge)
	
	MAX_RETRY_ATTEMPTS = 2  # Maximum reformulation attempts
	
	# Faithfulness checking: ensure answer is grounded in context
	ENABLE_FAITHFULNESS_CHECK = True
	FAITHFULNESS_THRESHOLD = 0.7  # Answer must be this grounded in context
	
	# Citation requirements
	REQUIRE_CITATIONS = True  # Force model to cite sources
	
	# ============================================================================
	# JURISDICTION DETECTION
	# ============================================================================
	# Supported jurisdictions for legal document filtering
	SUPPORTED_JURISDICTIONS = ['mexico', 'nepal', 'japan', 'canada', 'international']
	
	# Enable automatic jurisdiction detection from query
	ENABLE_JURISDICTION_DETECTION = True
	
	# Jurisdiction-specific metadata filtering
	ENABLE_METADATA_FILTERING = True
	
	# ============================================================================
	# MONITORING AND LOGGING
	# ============================================================================
	ENABLE_DETAILED_LOGGING = True
	LOG_RETRIEVAL_METRICS = True
	LOG_GENERATION_METRICS = True
	LOG_USER_FEEDBACK = True
	
	# Metrics tracking
	TRACK_LATENCY = True
	TRACK_TOKEN_USAGE = True
	
	# ============================================================================
	# CACHING SETTINGS
	# ============================================================================
	ENABLE_EMBEDDING_CACHE = True
	ENABLE_QUERY_CACHE = True
	CACHE_TTL_SECONDS = 3600  # Cache time-to-live (1 hour)
	
	# ============================================================================
	# GPU OPTIMIZATION SETTINGS
	# ============================================================================
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	USE_FP16 = True  # Use half precision (FP16) to reduce memory usage by 50%
	MAX_LENGTH = 2048  # Maximum context length for generation
	
	# API rate limiting and timeouts
	API_TIMEOUT_SECONDS = 30
	API_MAX_RETRIES = 3
	
	# ============================================================================
	# QUANTIZATION CONFIGURATION (for larger models)
	# ============================================================================
	# NOTE: Quantization is only used when loading larger models (7B+)
	# For TinyLlama, we use FP16 instead which is more efficient
	
	if BITSANDBYTES_AVAILABLE:
		# 4-bit quantization configuration (for models 7B+)
		# This reduces memory usage significantly (~8GB for 7B model vs ~28GB FP16)
		BNB_CONFIG = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16,
			bnb_4bit_use_double_quant=False,
		)
	else:
		# BitsAndBytes not available (common on Windows)
		# Will use FP16 or FP32 instead
		BNB_CONFIG = None
	
	# ============================================================================
	# DATA SOURCES CONFIGURATION
	# ============================================================================
	# Paths to legal document datasets
	MEXICO_CIVIL_LAW_PATH = './data/articlesFull.json'
	MEXICO_TRAINING_DATA_PATH = './data/TrainingData.json'
	NEPAL_CONSTITUTION_PATH = './data/nepal_constitution.json'  # To be added
	
	# Document versioning
	ENABLE_DOCUMENT_VERSIONING = True
	TRACK_DOCUMENT_UPDATES = True
