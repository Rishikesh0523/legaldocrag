from anthropic import Anthropic
from typing import List, Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ClaudeService:
    def __init__(self):
        self.client = Anthropic(api_key=settings.CLAUDE_API_KEY)
        self.model = settings.CLAUDE_MODEL
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate answer using Claude with context"""
        logger.info("Generating answer with Claude")
        
        if not context_chunks:
            return {
                'answer': "I don't have enough context to answer this question.",
                'is_context_based': False
            }
        
        # Build context from chunks
        context = "\n\n".join([
            f"[Source {i+1} - Page {chunk['metadata']['page_number']}]:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create prompt
        system_prompt = """You are a legal document assistant specialized in the Constitution of Nepal and other legal documents. 

Your responsibilities:
1. Answer questions ONLY based on the provided context from the documents
2. If the answer is not in the context, clearly state: "I cannot answer this question as it is not covered in the available documents."
3. Cite specific page numbers when providing answers
4. Be precise and accurate
5. Do not make assumptions or provide information outside the given context

Always maintain a professional and helpful tone."""

        user_prompt = f"""Context from legal documents:

{context}

Question: {question}

Please provide a precise answer based only on the context above. If the context doesn't contain information to answer the question, clearly state that you cannot answer."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.content[0].text
            
            # Check if answer indicates lack of context
            no_context_indicators = [
                "cannot answer",
                "not covered",
                "not mentioned",
                "no information",
                "not in the context"
            ]
            
            is_context_based = not any(indicator in answer.lower() for indicator in no_context_indicators)
            
            logger.info(f"Answer generated successfully (context-based: {is_context_based})")
            
            return {
                'answer': answer,
                'is_context_based': is_context_based
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise