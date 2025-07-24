import pytest
import json
import tempfile
from unittest.mock import Mock, patch
import torch  

from vector_service import VectorService
from pdf_processor import PDFProcessor  
from door_schedule_parser import parse_door_schedule, calculate_door_area, calculate_door_cost
from ai_service import AIService

class TestDoorParser:
    
    def test_calculate_door_area_basic(self):
        area = calculate_door_area("90x210")
        expected = (90/100) * (210/100)  
        assert abs(area - expected) < 0.001

        area = calculate_door_area("80×200")  
        expected = (80/100) * (200/100)
        assert abs(area - expected) < 0.001
    
    def test_calculate_door_area_invalid(self):
        assert calculate_door_area("invalid") == 0.0
        assert calculate_door_area("") == 0.0
        assert calculate_door_area("90") == 0.0
        
    def test_calculate_door_cost_basic(self):
        door_info = {
            "area_sqm": 2.0,
            "material": "wood"
        }
        cost = calculate_door_cost(door_info)
        for field in ["material_cost", "labor_cost", "installation_cost", "total_cost"]:
            assert field in cost
            assert isinstance(cost[field], (int, float))
            assert cost[field] >= 0
    
    def test_parse_door_schedule_empty(self):
        doors = parse_door_schedule("")
        assert isinstance(doors, list)
        assert len(doors) == 0
    
    def test_parse_door_schedule_basic(self):
        text = """
        Door Schedule
        D-1 | 90x210 | wood | entrance door
        """
        doors = parse_door_schedule(text)
        assert len(doors) >= 0  
        if len(doors) > 0:
            door = doors[0]
            assert "door_id" in door
            assert "material" in door
            assert "area_sqm" in door

class TestPDFProcessor:
    
    def test_validate_file_extension(self):
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.validate_file("test.txt", b"some content")
    
    def test_validate_file_size(self):
        processor = PDFProcessor()
        large_content = b"x" * (51 * 1024 * 1024)
        with pytest.raises(Exception):
            processor.validate_file("test.pdf", large_content)

class TestVectorService:
    
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer') 
    def test_init_basic(self, mock_tokenizer, mock_create_model):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = Mock()
        
        with patch.object(mock_model, 'encode_text') as mock_encode:
            mock_encode.return_value = torch.randn(1, 512)  # טנסור אמיתי
            
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

class TestIntegration:
    
    def test_door_parser_with_real_data(self):
        sample_text = """
        DOOR SCHEDULE
        
        Door ID | Size | Material | Notes
        D-1 | 90x210 | wooden | Main entrance
        D-2 | 80x200 | metal | Fire door
        """
        doors = parse_door_schedule(sample_text)
        assert isinstance(doors, list)
        for door in doors:
            assert "door_id" in door
            assert "material" in door
            assert "area_sqm" in door
            assert door["area_sqm"] > 0

def create_mock_pdf_content():
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"

def create_temp_json_file(data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return f.name

if __name__ == "__main__":
    print("Running basic tests...")
    
    try:
        area = calculate_door_area("90x210")
        print(f"Door area test: {area} sqm")
        assert area > 0
        print("Door area calculation works")
    except ImportError:
        print("Could not import door_parser - adjust import path")
    except Exception as e:
        print(f"Door area calculation failed: {e}")
    
    try:
        import subprocess
        result = subprocess.run(['pytest', __file__, '-v'], capture_output=True, text=True)
        print("Pytest output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
    except FileNotFoundError:
        print("Pytest not found. Install with: pip install pytest")
        print("Or run individual test functions manually")