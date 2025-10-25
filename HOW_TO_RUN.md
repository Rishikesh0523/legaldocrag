# How to Run the Legal RAG Pipeline 🚀

This guide provides step-by-step instructions to get the Legal RAG Pipeline up and running on your system.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (NVIDIA GPU with CUDA for faster processing)
  - Without GPU: Uses CPU (slower but works)
  - With GPU: 6GB VRAM minimum for TinyLlama, 16GB+ for larger models
- **Disk Space**: 10-20GB for models and cache

### Software Requirements
- Python 3.8+
- pip (Python package manager)
- Git (optional, for cloning)

---

## 🔧 Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd "d:\Rishikesh\BCT\VII\Fusemachines\Major Project\LegalDoc-RAG-Legal-Document-Drafting-Summarizing"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch (deep learning framework)
# - Transformers (Hugging Face models)
# - sentence-transformers (embeddings)
# - FAISS (vector search)
# - spaCy (NLP)
# - OpenAI and Anthropic APIs (optional)
# - PDF processing libraries (pypdf2, pdfplumber)
# - And more...
```

**Installation time**: 5-10 minutes (depending on internet speed)

### Step 4: Download spaCy Language Model

```powershell
# Download English language model for NER
python -m spacy download en_core_web_sm
```

### Step 5: Verify Installation

```powershell
# Check Python version
python --version

# Check installed packages
pip list | findstr "torch transformers"
```

---

## 🎮 Running the Pipeline

### Option 1: Run Demo Script (Recommended for First Time)

This is the **easiest way** to test the pipeline with sample queries:

```powershell
python run_pipeline.py
```

**What it does:**
- Loads all datasets (Mexico Civil Law, Nepal Constitution, etc.)
- Runs 4 test queries demonstrating different features
- Shows performance metrics
- Displays results with citations

**Expected output:**
```
================================================================================
Legal RAG Pipeline v2.0 - Production-Ready Legal Document Q&A
================================================================================

✓ GPU Detected: NVIDIA GeForce RTX 3060
✓ Total VRAM: 12.00 GB

📚 Model Configuration:
   • Generator Type: LOCAL
   • LLM Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ...

Loading All Legal Datasets
================================================================================
LegalDocumentLoader: Loading Mexico Civil Law from './data/articlesFull.json'...
LegalDocumentLoader: Loaded 283 Mexico Civil Law articles
...
Total Documents Loaded: 1,631
================================================================================

TEST QUERY 1/4
...
```

**First run**: Takes 5-15 minutes (downloads models, processes PDFs)  
**Subsequent runs**: Takes 1-2 minutes (uses cached data)

### Option 2: Test PDF Processing

Test the new PDF processing feature with caching:

```powershell
python test_pdf_processing.py
```

**What it does:**
- Loads Nepal Constitution PDF
- Shows first-time processing (slow)
- Shows cached loading (fast - 50-100x faster!)
- Displays performance comparison

### Option 3: Use as Python Module

Create your own script:

```python
# my_query.py
from legaldocrag import PipelineConfig, LegalRAGPipeline

# Initialize
config = PipelineConfig()
pipeline = LegalRAGPipeline(config)

# Load documents
pipeline.process_documents(load_from_datasets=True)

# Ask a question
result = pipeline.answer_query(
    "What does Mexican Civil Law say about contracts?"
)

print(result['answer'])
print(f"\nSources: {len(result['sources'])} documents")
```

Run it:
```powershell
python my_query.py
```

### Option 4: Interactive Mode

```python
# interactive.py
from legaldocrag import PipelineConfig, LegalRAGPipeline

config = PipelineConfig()
pipeline = LegalRAGPipeline(config)
pipeline.process_documents(load_from_datasets=True)

while True:
    query = input("\nEnter your legal query (or 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    result = pipeline.answer_query(query)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    if result.get('sources'):
        print(f"Sources used: {len(result['sources'])}")
```

---

## ⚙️ Configuration Options

### Using Local Models (Default)

The pipeline uses local models by default (no API keys needed):

**Edit `legaldocrag/config.py`:**
```python
class PipelineConfig:
    GENERATOR_TYPE = 'local'  # Default
    BASE_LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # 6GB VRAM
```

**Model Options:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - 6GB VRAM (default, fastest)
- `meta-llama/Llama-2-7b-chat-hf` - 16GB VRAM (better quality)
- `meta-llama/Llama-2-13b-chat-hf` - 24GB VRAM (best quality)

### Using OpenAI ChatGPT API

**Set API key:**
```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key-here"

# Or set permanently in Windows Environment Variables
```

**Edit `legaldocrag/config.py`:**
```python
class PipelineConfig:
    GENERATOR_TYPE = 'openai'
    OPENAI_MODEL = 'gpt-4o'  # or gpt-3.5-turbo
```

### Using Anthropic Claude API

**Set API key:**
```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-api-key-here"
```

**Edit `legaldocrag/config.py`:**
```python
class PipelineConfig:
    GENERATOR_TYPE = 'claude'
    CLAUDE_MODEL = 'claude-3-5-sonnet-20241022'
```

---

## 📊 Understanding the Output

### Sample Output from `run_pipeline.py`

```
================================================================================
RESULT
================================================================================
Status: ✓ SUCCESS
Query: What does Mexican Civil Law say about juridical acts against public policy?

Answer:
According to Article 8 of the Mexican Civil Code, juridical acts executed 
against the provisions of laws of public interest shall be void, except in 
cases where the law prescribes a different consequence for their violation. 
This means that contracts or legal acts that violate public policy are 
generally considered null and void.

Sources (3 documents):
  - ID: mexico_article_8, Score: 0.9234, Jurisdiction: mexico
  - ID: mexico_article_1795, Score: 0.8567, Jurisdiction: mexico
  - ID: mexico_case_42, Score: 0.7891, Jurisdiction: mexico

Detected Jurisdictions: ['mexico']

Performance Metrics:
  - Retrieval Time: 245.32 ms
  - Generation Time: 1823.45 ms
  - Total Time: 2068.77 ms
  - Tokens Used: 387

Quality Checks:
  - Faithful: True
  - Faithfulness Score: 0.8934
================================================================================
```

### Understanding Metrics

- **Retrieval Time**: Time to find relevant documents
- **Generation Time**: Time for LLM to generate answer
- **Total Time**: End-to-end query processing time
- **Tokens Used**: API tokens consumed (for ChatGPT/Claude)
- **Faithfulness Score**: How well answer matches source documents (0-1)
- **Sources**: Retrieved documents used for answer

---

## 🐛 Troubleshooting

### Problem 1: "No module named 'torch'"

**Solution:**
```powershell
pip install torch torchvision torchaudio
```

### Problem 2: "CUDA out of memory"

**Solutions:**
1. Use smaller model:
   ```python
   BASE_LLM_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
   ```

2. Use API-based model instead:
   ```python
   GENERATOR_TYPE = 'openai'  # or 'claude'
   ```

3. Close other applications using GPU

### Problem 3: "Import PyPDF2 could not be resolved"

**Solution:**
```powershell
pip install pypdf2 pdfplumber
```

### Problem 4: "spaCy model not found"

**Solution:**
```powershell
python -m spacy download en_core_web_sm
```

### Problem 5: Slow Performance (No GPU)

**Solutions:**
1. Use API-based model (OpenAI/Claude) - much faster
2. Use smaller local model (TinyLlama)
3. Reduce number of retrieved documents in config:
   ```python
   TOP_K_RETRIEVAL = 5  # Reduce from 10
   ```

### Problem 6: "Module not found: legaldocrag"

**Solution:**
Make sure you're in the project root directory:
```powershell
cd "d:\Rishikesh\BCT\VII\Fusemachines\Major Project\LegalDoc-RAG-Legal-Document-Drafting-Summarizing"
python run_pipeline.py
```

### Problem 7: Models downloading slowly

**Solution:**
Models download automatically on first run. Be patient:
- TinyLlama: ~2GB (5-10 minutes)
- Embedding model: ~2.3GB (5-10 minutes)
- Reranker: ~90MB (1-2 minutes)

Check download progress in console output.

---

## 📁 Data Files

The pipeline uses these datasets:

### Included in Repository
- `data/articlesFull.json` - Mexico Civil Law (283 articles)
- `data/TrainingData.json` - Mexico training data (1,181 cases)
- `data/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf` - Nepal Constitution (167 pages)

### Generated on First Run
- `data/.cache/` - Cached PDF data (auto-generated)
- `logs/` - Pipeline execution logs
- Model cache (Hugging Face models downloaded automatically)

---

## 🔍 Testing Different Features

### Test 1: Basic Query
```python
result = pipeline.answer_query(
    "What does Mexican Civil Law say about contracts?"
)
```

### Test 2: Jurisdiction-Specific Query
```python
result = pipeline.answer_query(
    "What fundamental rights are guaranteed in Nepal Constitution?"
)
```

### Test 3: Unanswerable Query (Tests Reliability)
```python
result = pipeline.answer_query(
    "What is the weather in Tokyo?"
)
# Expected: Pipeline declines to answer (not in scope)
```

### Test 4: Multi-Jurisdiction Query
```python
result = pipeline.answer_query(
    "Compare property rights in Mexico and Nepal"
)
```

---

## 📈 Performance Expectations

### First Run (Cold Start)
- **Time**: 10-20 minutes
- **Reason**: Downloads models, processes PDFs, builds indices
- **Downloads**: ~5-7GB of models

### Subsequent Runs
- **Time**: 1-2 minutes
- **Reason**: Uses cached models and data
- **Per Query**: 2-5 seconds (local) or 0.5-2 seconds (API)

### Query Processing Times (Approximate)

| Configuration | Retrieval | Generation | Total |
|--------------|-----------|------------|-------|
| **TinyLlama (GPU)** | 0.2-0.5s | 1-3s | 1.5-3.5s |
| **TinyLlama (CPU)** | 0.5-1s | 5-15s | 6-16s |
| **OpenAI GPT-4** | 0.2-0.5s | 0.5-2s | 1-2.5s |
| **Claude Sonnet** | 0.2-0.5s | 0.5-1.5s | 1-2s |

---

## 🎯 Quick Start Commands

```powershell
# 1. Navigate to project
cd "d:\Rishikesh\BCT\VII\Fusemachines\Major Project\LegalDoc-RAG-Legal-Document-Drafting-Summarizing"

# 2. Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Run demo
python run_pipeline.py

# 5. Test PDF processing
python test_pdf_processing.py
```

---

## 📚 Additional Resources

- **PDF Processing Guide**: `PDF_PROCESSING_GUIDE.md`
- **Quick Reference**: `PDF_QUICK_REFERENCE.md`
- **Configuration Guide**: `CONFIGURATION_GUIDE.md` (if exists)
- **Production Guide**: `PRODUCTION_GUIDE.md` (if exists)

---

## 💡 Tips for Best Results

1. **First time users**: Start with `python run_pipeline.py` to see demo
2. **Limited GPU**: Use API-based models (OpenAI/Claude)
3. **Production use**: Enable caching, monitoring, and logging
4. **Custom data**: Add your PDFs to `data/` directory
5. **Performance**: Use GPU if available, otherwise use API models

---

## 🔐 Security Notes

- **API Keys**: Never commit API keys to Git
- **Use Environment Variables**: Set keys via environment variables
- **API Costs**: Be aware of API usage costs for OpenAI/Claude

---

## ✅ Verification Checklist

Before running, ensure:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded (`python -m spacy download en_core_web_sm`)
- [ ] PDF libraries installed (`pip install pypdf2 pdfplumber`)
- [ ] In correct directory (project root)
- [ ] Virtual environment activated (if using)
- [ ] Data files present in `data/` directory

---

## 🎓 Learning Path

1. **Day 1**: Run `run_pipeline.py` to understand the system
2. **Day 2**: Test PDF processing with `test_pdf_processing.py`
3. **Day 3**: Create custom queries with your own script
4. **Day 4**: Add your own PDFs and customize configuration
5. **Day 5**: Explore API-based models (OpenAI/Claude)

---

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check that you're using Python 3.8+
5. Ensure sufficient disk space and RAM

---

**Ready to start?** Run:
```powershell
python run_pipeline.py
```

Good luck! 🚀
