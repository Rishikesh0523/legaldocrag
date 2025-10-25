"""
API Generator Module

Supports OpenAI (ChatGPT) and Anthropic (Claude) APIs for generation.
Provides unified interface with local models for seamless switching.
"""

import os
import time
from typing import Optional, Dict, Any
import logging

# Optional API dependencies - install as needed
try:
	import openai
	OPENAI_AVAILABLE = True
except ImportError:
	OPENAI_AVAILABLE = False

try:
	import anthropic
	ANTHROPIC_AVAILABLE = True
except ImportError:
	ANTHROPIC_AVAILABLE = False


class APIGenerator:
	"""
	Generator that uses external API services (OpenAI or Anthropic).
	
	Handles rate limiting, retries, and error handling for production reliability.
	"""
	
	def __init__(
		self,
		provider: str,
		model_name: str,
		api_key: Optional[str] = None,
		timeout: int = 30,
		max_retries: int = 3
	):
		"""
		Initialize API generator.
		
		Args:
			provider: 'openai' or 'claude'
			model_name: Specific model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
			api_key: API key (if not provided, reads from environment)
			timeout: Request timeout in seconds
			max_retries: Maximum number of retry attempts
		"""
		self.provider = provider.lower()
		self.model_name = model_name
		self.timeout = timeout
		self.max_retries = max_retries
		
		# Setup logging
		self.logger = logging.getLogger(__name__)
		
		# Initialize API clients
		if self.provider == 'openai':
			if not OPENAI_AVAILABLE:
				raise ImportError(
					"OpenAI library not installed. Install with: pip install openai"
				)
			
			self.api_key = api_key or os.getenv('OPENAI_API_KEY')
			if not self.api_key:
				raise ValueError(
					"OpenAI API key not found. Set OPENAI_API_KEY environment variable "
					"or pass api_key parameter."
				)
			
			self.client = openai.OpenAI(api_key=self.api_key, timeout=timeout)
			print(f"APIGenerator: Initialized OpenAI client with model '{model_name}'")
			
		elif self.provider == 'claude':
			if not ANTHROPIC_AVAILABLE:
				raise ImportError(
					"Anthropic library not installed. Install with: pip install anthropic"
				)
			
			# self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
			self.api_key = "<YOUR_ANTHROPIC_API_KEY>"
			if not self.api_key:
				raise ValueError(
					"Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
					"or pass api_key parameter."
				)
			
			self.client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)
			print(f"APIGenerator: Initialized Anthropic client with model '{model_name}'")
			
		else:
			raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'claude'")
		
		# Metrics tracking
		self.total_requests = 0
		self.total_tokens = 0
		self.total_errors = 0
	
	def generate(self, query: str, context: str, **kwargs) -> str:
		"""
		Generate answer using API.
		
		Args:
			query: User's question
			context: Retrieved context from documents
			**kwargs: Additional generation parameters (temperature, max_tokens, etc.)
			
		Returns:
			Generated answer
		"""
		prompt = self._build_prompt(query, context)
		
		# Attempt generation with retries
		for attempt in range(self.max_retries):
			try:
				self.logger.info(f"APIGenerator: Attempt {attempt + 1}/{self.max_retries}")
				
				if self.provider == 'openai':
					response = self._generate_openai(prompt, **kwargs)
				elif self.provider == 'claude':
					response = self._generate_claude(prompt, **kwargs)
				else:
					raise ValueError(f"Unknown provider: {self.provider}")
				
				self.total_requests += 1
				self.logger.info("APIGenerator: Generation successful")
				return response
				
			except Exception as e:
				self.total_errors += 1
				self.logger.error(f"APIGenerator: Error on attempt {attempt + 1}: {e}")
				
				if attempt < self.max_retries - 1:
					# Exponential backoff
					wait_time = 2 ** attempt
					self.logger.info(f"APIGenerator: Retrying in {wait_time} seconds...")
					time.sleep(wait_time)
				else:
					raise RuntimeError(
						f"APIGenerator: Failed after {self.max_retries} attempts. Last error: {e}"
					)
	
	def _generate_openai(self, prompt: str, **kwargs) -> str:
		"""Generate using OpenAI API."""
		
		# Default parameters
		temperature = kwargs.get('temperature', 0.3)
		max_tokens = kwargs.get('max_tokens', 1024)
		
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[
				{
					"role": "system",
					"content": (
						"You are an expert legal assistant. Answer questions based solely on "
						"the provided legal context. Always cite sources using document IDs. "
						"If the context doesn't contain enough information, explicitly state that."
					)
				},
				{
					"role": "user",
					"content": prompt
				}
			],
			temperature=temperature,
			max_tokens=max_tokens,
			timeout=self.timeout
		)
		
		# Track token usage
		if hasattr(response, 'usage'):
			self.total_tokens += response.usage.total_tokens
			self.logger.info(f"APIGenerator: Used {response.usage.total_tokens} tokens")
		
		return response.choices[0].message.content.strip()
	
	def _generate_claude(self, prompt: str, **kwargs) -> str:
		"""Generate using Anthropic Claude API."""
		
		# Default parameters
		temperature = kwargs.get('temperature', 0.3)
		max_tokens = kwargs.get('max_tokens', 1024)
		
		response = self.client.messages.create(
			model=self.model_name,
			max_tokens=max_tokens,
			temperature=temperature,
			system=(
				"You are an expert legal assistant. Answer questions based solely on "
				"the provided legal context. Always cite sources using document IDs. "
				"If the context doesn't contain enough information, explicitly state that."
			),
			messages=[
				{
					"role": "user",
					"content": prompt
				}
			]
		)
		
		# Track token usage
		if hasattr(response, 'usage'):
			self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
			self.logger.info(
				f"APIGenerator: Used {response.usage.input_tokens} input + "
				f"{response.usage.output_tokens} output tokens"
			)
		
		return response.content[0].text.strip()
	
	def _build_prompt(self, query: str, context: str) -> str:
		"""Build the prompt for API generation."""
		return f"""--- LEGAL CONTEXT START ---
{context}
--- LEGAL CONTEXT END ---

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based EXCLUSIVELY on the provided legal context above.
2. For every statement in your answer, provide an inline citation using document IDs (e.g., [doc_id_1]).
3. If the context does not contain sufficient information to answer the question, explicitly state: "I cannot provide a complete answer based on the available legal documents."
4. Do NOT use any external knowledge or prior information not present in the context.
5. Be precise, accurate, and cite relevant articles/sections.

ANSWER:"""
	
	def get_metrics(self) -> Dict[str, Any]:
		"""Get usage metrics."""
		return {
			'total_requests': self.total_requests,
			'total_tokens': self.total_tokens,
			'total_errors': self.total_errors,
			'provider': self.provider,
			'model': self.model_name
		}
