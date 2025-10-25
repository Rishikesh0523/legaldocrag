from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Optional


class Generator:
	"""Loads a LoRA-fine-tuned LLM and generates text.
	
	Supports both small models (TinyLlama, GPT-2) and large models (LLaMA-2-7B+).
	Automatically optimizes loading based on model size and available resources.
	
	This class handles LOCAL models only. For API-based models (OpenAI, Claude),
	use the APIGenerator class instead.
	"""
	
	def __init__(self, base_model_name: str, lora_adapter_path: str, bnb_config):
		print(f"Generator: Loading base model '{base_model_name}'...")
		
		# Detect if GPU is available
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model_name = base_model_name
		print(f"Generator: Using device '{self.device}'")
		
		# Detect model size based on name to choose optimal loading strategy
		is_small_model = any(x in base_model_name.lower() for x in ['tinyllama', 'gpt2', 'distilgpt2', 'opt-125m', 'pythia-160m'])
		is_large_model = any(x in base_model_name.lower() for x in ['llama-2-7b', 'llama-2-13b', 'mistral-7b'])
		
		try:
			# ========================================================================
			# LOADING STRATEGY: Choose based on model size and available hardware
			# ========================================================================
			
			if is_small_model:
				# ---- SMALL MODELS (TinyLlama, GPT-2, etc.) ----
				# Use FP16 on GPU for efficiency, FP32 on CPU
				print(f"Generator: Loading small model with FP16 optimization...")
				
				if self.device == "cuda":
					self.model = AutoModelForCausalLM.from_pretrained(
						base_model_name,
						torch_dtype=torch.float16,  # Half precision for memory efficiency
						device_map="auto",           # Automatically map to GPU
						low_cpu_mem_usage=True,
						trust_remote_code=True       # Some models require this
					)
				else:
					self.model = AutoModelForCausalLM.from_pretrained(
						base_model_name,
						torch_dtype=torch.float32,   # Full precision on CPU
						low_cpu_mem_usage=True,
					)
					self.model.to(self.device)
				
			elif is_large_model and bnb_config is not None:
				# ---- LARGE MODELS with Quantization (7B+) ----
				# Use 4-bit quantization to fit in limited GPU memory
				print(f"Generator: Loading large model with 4-bit quantization...")
				print(f"Generator: This requires 16GB+ GPU VRAM")
				
				self.model = AutoModelForCausalLM.from_pretrained(
					base_model_name,
					quantization_config=bnb_config,  # 4-bit quantization
					device_map="auto",                # Automatically distribute across devices
					low_cpu_mem_usage=True,
				)
				
			else:
				# ---- LARGE MODELS without Quantization ----
				# Fallback: Use FP16 on GPU or FP32 on CPU
				print(f"Generator: Loading model without quantization...")
				
				if self.device == "cuda":
					self.model = AutoModelForCausalLM.from_pretrained(
						base_model_name,
						torch_dtype=torch.float16,
						device_map="auto",
						low_cpu_mem_usage=True,
					)
				else:
					self.model = AutoModelForCausalLM.from_pretrained(
						base_model_name,
						torch_dtype=torch.float32,
						low_cpu_mem_usage=True,
					)
					self.model.to(self.device)
			
			print(f"Generator: Model loaded successfully on {self.device}.")
			
			# Show GPU memory usage if available
			if torch.cuda.is_available():
				allocated = torch.cuda.memory_allocated(0) / 1024**3
				reserved = torch.cuda.memory_reserved(0) / 1024**3
				print(f"Generator: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
			
		except Exception as e:
			print(f"Generator: ERROR loading model: {e}")
			raise
		
		# Load tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# Try to load LoRA adapters if available
		try:
			print(f"Generator: Loading LoRA adapter from '{lora_adapter_path}'...")
			self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
			print("Generator: LoRA adapter loaded successfully.")
		except Exception as e:
			print(f"Generator: WARNING - Could not load LoRA adapter. Using base model. Error: {e}")

	def generate(self, query: str, context: str) -> str:
		"""Generates an answer based on the query and context."""
		prompt = self._build_prompt(query, context)
		
		# Tokenize with proper truncation
		inputs = self.tokenizer(
			prompt, 
			return_tensors="pt",
			truncation=True,
			max_length=2048  # Limit context length to fit in memory
		)
		
		# Move inputs to the same device as model (device_map handles this automatically)
		# But for models not using device_map, we need to move inputs
		if hasattr(self.model, 'device'):
			inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
		
		print("Generator: Generating response...")
		
		# Generate with appropriate parameters
		with torch.no_grad():  # Disable gradient calculation for inference
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=512,
				temperature=0.7,      # Slightly higher for more diverse outputs
				do_sample=True,
				top_p=0.9,            # Nucleus sampling
				top_k=50,             # Top-k sampling
				repetition_penalty=1.1,  # Reduce repetition
				pad_token_id=self.tokenizer.eos_token_id,
			)
		
		response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		
		# Extract answer based on model type
		if "ANSWER:" in response:
			answer = response.split("ANSWER:")[-1].strip()
		elif "<|assistant|>" in response:  # TinyLlama format
			answer = response.split("<|assistant|>")[-1].strip()
		else:
			# Fallback: take everything after the prompt
			answer = response[len(prompt):].strip()
		
		print("Generator: Response generated.")
		return answer

	def _build_prompt(self, query: str, context: str) -> str:
		"""Builds the prompt for the LLM.
		
		Uses different formats for different models:
		- TinyLlama: Chat format with special tokens
		- LLaMA-2: Standard instruction format
		"""
		model_name = self.model.config._name_or_path if hasattr(self.model, 'config') else ""
		
		# TinyLlama uses a specific chat format
		if 'tinyllama' in model_name.lower():
			return f"""<|system|>
You are an expert legal assistant. Answer the user's question based exclusively on the provided legal texts. Provide inline citations using document IDs (e.g., [doc_id_1]).</s>
<|user|>
Context:
{context}

Question: {query}</s>
<|assistant|>
"""
		
		# Standard format for LLaMA-2 and similar models
		else:
			return f"""
You are an expert legal assistant. Your task is to answer the user's question based exclusively on the provided legal texts. Do not use any prior knowledge. For every statement in your answer, you MUST provide an inline citation referencing the exact source document ID (e.g., [doc_id_1]).

--- CONTEXT START ---
{context}
--- CONTEXT END ---

USER QUESTION:
{query}

ANSWER:
"""
