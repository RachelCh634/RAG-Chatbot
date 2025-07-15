from openai import OpenAI
from typing import List, Dict, Any
from config import Config

class AIService:
    """AI service for generating answers"""
    
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config.OPENROUTER_API_KEY
        )
    
    def generate_answer_from_context(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks[:self.config.MAX_CONTEXT_CHUNKS])
        prompt = f"""Based on the following context from a PDF document, answer the user's question clearly and concisely.

Context:
{context}

Question: {query}

Instructions:
- Give a direct, short answer (2-3 sentences maximum)
- Only use information from the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Be specific and factual
- Answer in the same language as the question

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.AI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Give short, direct answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"AI Error: {e}")
            return f"Based on the document: {context_chunks[0][:150]}..."
    
    def chat_with_context(self, query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Chat with context and history"""
        
        if conversation_history is None:
            conversation_history = []
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant answering questions about documents. Give short, direct answers."}
        ]
        
        for msg in conversation_history[-self.config.CONVERSATION_HISTORY_LIMIT:]:
            messages.append({"role": "user", "content": msg.get("question", "")})
            messages.append({"role": "assistant", "content": msg.get("answer", "")})
        
        current_prompt = f"""Based on this context from the document:

{context}

Question: {query}

Give a short, direct answer (1-2 sentences)."""

        messages.append({"role": "user", "content": current_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.AI_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Chat AI Error: {e}")
            return "I'm having trouble generating a response right now."
    
    def determine_confidence(self, best_score: float) -> str:
        """Determining the level of confidence in the answer"""
        if best_score > 0.7:
            return "high"
        elif best_score > 0.3:
            return "medium"
        else:
            return "low"