from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional
import time
from config import Config

class LangChainVectorStore(VectorStore):
    """Custom VectorStore wrapper for our VectorService"""
    
    def __init__(self, vector_service):
        self.vector_service = vector_service
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.vector_service.search_vectors(query, k)
        documents = []
        for result in results:
            doc = Document(
                page_content=result['text'],
                metadata={
                    'filename': result['filename'],
                    'chunk_index': result['chunk_index'],
                    'score': result['score']
                }
            )
            documents.append(doc)
        return documents
    
    @classmethod
    def from_texts(cls, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> "LangChainVectorStore":
        pass  

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        pass

class AIService:
    def __init__(self, vector_service=None):
        self.config = Config()
        
        self.llm = ChatOpenAI(
            openai_api_key=self.config.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            model="qwen/qwen-2.5-72b-instruct",  
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS,
            request_timeout=120,
            max_retries=3,
        )
        
        self.vector_service = vector_service
        if vector_service:
            self.vector_store = LangChainVectorStore(vector_service)
        else:
            self.vector_store = None
        
        self.conversation_history = []
        self.max_history_length = 6
        
        self._setup_chains()
    
    def _setup_chains(self):
        self.qa_template = """You are a helpful assistant specialized in construction and architectural documents.
Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Give a direct, clear answer (2-3 sentences maximum)
- Only use information from the provided context
- Be specific about doors, windows, measurements, and costs
- If you don't know the answer from the context, say so clearly
- Use simple punctuation and avoid special characters like ** or \n

Answer:"""
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.qa_template
        )
        
        self.qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        
        self.conversation_template = """You are a helpful assistant specialized in construction and architectural documents.
You help users understand door and window schedules, calculate areas, and provide cost estimates.

Previous conversation:
{chat_history}

Context from document:
{context}

Human: {input}
Assistant:"""
        
        self.conversation_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "input"],
            template=self.conversation_template
        )
        
        self.conversation_chain = self.conversation_prompt | self.llm | StrOutputParser()

    def chat_with_context(self, query: str, context: str = None) -> str:
        """Chat with context using Qwen model with vector search"""
        try:
            chat_history = ""
            for entry in self.conversation_history[-4:]: 
                chat_history += f"Human: {entry['question']}\nAssistant: {entry['answer']}\n\n"
            
            retrieved_context = ""
            if self.vector_store:
                try:
                    relevant_docs = self.vector_store.similarity_search(query, k=4)
                    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                except Exception as e:
                    print(f"Error retrieving context: {e}")
            
            combined_context = ""
            if context:
                combined_context += context
            if retrieved_context:
                if combined_context:
                    combined_context += "\n\n" + retrieved_context
                else:
                    combined_context = retrieved_context
            
            response = self.conversation_chain.invoke({
                "chat_history": chat_history,
                "context": combined_context,
                "input": query
            })
            
            self.conversation_history.append({
                "question": query,
                "answer": response
            })
            
            return response.strip()
        except Exception as e:
            print(f"Qwen Chat Error: {e}")
            return "I'm having trouble generating a response right now."

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        return self.conversation_history.copy()