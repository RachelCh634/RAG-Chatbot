import pytest
from unittest.mock import Mock, patch
import os

# Mock environment variables for testing
with patch.dict(os.environ, {
    'OPENROUTER_API_KEY': 'test_openrouter_key',
    'TAVILY_API_KEY': 'test_tavily_key'
}):

    from ai_service import AIService

class TestAIService:
    
    @patch('langchain_openai.ChatOpenAI')
    def test_init_basic(self, mock_chat_openai):
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        try:
            service = AIService()
            assert service is not None
            assert hasattr(service, 'llm')
            assert hasattr(service, 'conversation_history')
        except Exception as e:
            pytest.fail(f"AIService failed to initialize: {e}")
    
    @patch('langchain_openai.ChatOpenAI')
    def test_clear_memory(self, mock_chat_openai):
        service = AIService()
        service.conversation_history = [{"test": "data"}]
        service.clear_memory()
        assert len(service.conversation_history) == 0
    
    @patch('langchain_openai.ChatOpenAI')
    def test_get_conversation_history(self, mock_chat_openai):
        service = AIService()
        service.conversation_history = [{"question": "test", "answer": "response"}]
        history = service.get_conversation_history()
        assert isinstance(history, list)
        assert len(history) == 1