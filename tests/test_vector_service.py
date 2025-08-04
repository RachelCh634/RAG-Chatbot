import pytest
from unittest.mock import Mock, patch
import torch
import os

# Mock environment variables for testing
with patch.dict(os.environ, {
    'OPENROUTER_API_KEY': 'test_openrouter_key',
    'TAVILY_API_KEY': 'test_tavily_key'
}):

    from vector_service import VectorService

class TestVectorService:
    
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer') 
    def test_init_basic(self, mock_tokenizer, mock_create_model):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = Mock()
        
        with patch.object(mock_model, 'encode_text') as mock_encode:
            mock_encode.return_value = torch.randn(1, 512) 
            
            try:
                service = VectorService()
                assert service is not None
                assert hasattr(service, 'dimension')
            except Exception as e:
                pytest.fail(f"VectorService failed to initialize: {e}")
    
    def test_preprocess_text(self):
        with patch('open_clip.create_model_and_transforms') as mock_create_model, \
             patch('open_clip.get_tokenizer'):
            mock_model = Mock()
            mock_preprocess = Mock()
            mock_create_model.return_value = (mock_model, None, mock_preprocess)

            service = VectorService()
            result = service.preprocess_text("  hello   world  ")
            assert isinstance(result, str)
            assert "hello world" in result.lower()
    
    def test_split_text_into_chunks(self):
        with patch('open_clip.create_model_and_transforms') as mock_create_model, \
             patch('open_clip.get_tokenizer'):
            mock_model = Mock()
            mock_preprocess = Mock()
            mock_create_model.return_value = (mock_model, None, mock_preprocess)

            service = VectorService()
            text = " ".join(["word"] * 100) 
            chunks = service.split_text_into_chunks(text, chunk_size=10, overlap=2)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)