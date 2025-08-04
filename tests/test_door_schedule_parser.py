import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import torch
import io
import numpy as np
from PIL import Image
import os

# Mock environment variables for testing
with patch.dict(os.environ, {
    'OPENROUTER_API_KEY': 'test_openrouter_key',
    'TAVILY_API_KEY': 'test_tavily_key'
}):

    from door_schedule_parser import calculate_area_sqm, extract_door_schedule_json, calculate_costs_and_augment, TavilyPriceSearcher

class TestDoorScheduleParser:
    
    def test_calculate_area_sqm_basic(self):
        """Test basic area calculation"""
        area = calculate_area_sqm(90, 210)
        expected = (90/100) * (210/100)  
        assert abs(area - expected) < 0.001

        area = calculate_area_sqm(80, 200)  
        expected = (80/100) * (200/100)
        assert abs(area - expected) < 0.001
    
    def test_calculate_area_sqm_small_numbers(self):
        """Test area calculation with small dimensions"""
        area = calculate_area_sqm(50, 100)
        expected = 0.5
        assert abs(area - expected) < 0.001
    
    def test_calculate_area_sqm_rounding(self):
        """Test that area is properly rounded to 3 decimal places"""
        area = calculate_area_sqm(33.333, 66.666)
        assert isinstance(area, float)
        # Check that it's rounded to 3 decimal places
        assert len(str(area).split('.')[-1]) <= 3

    @patch('door_schedule_parser.TavilyClient')
    def test_tavily_price_searcher_init(self, mock_tavily_client):
        """Test TavilyPriceSearcher initialization"""
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        assert searcher.region == "usa"
        assert searcher.client == mock_client_instance
        assert isinstance(searcher.price_cache, dict)

    @patch('door_schedule_parser.TavilyClient')
    def test_search_material_prices_basic(self, mock_tavily_client):
        """Test basic material price search"""
        # Mock Tavily client response
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {
            "results": [
                {
                    "content": "Wood door price is 200$ per square meter with installation: $80"
                },
                {
                    "content": "Premium wood doors cost 250$ per sqm, installation $100"
                }
            ]
        }
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        result = searcher.search_material_prices("wood")
        
        assert isinstance(result, dict)
        assert "price_per_sqm" in result
        assert "installation" in result
        assert "timestamp" in result
        assert isinstance(result["price_per_sqm"], (int, float))
        assert isinstance(result["installation"], (int, float))

    @patch('door_schedule_parser.TavilyClient')
    def test_search_material_prices_cache(self, mock_tavily_client):
        """Test that price search results are cached"""
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        
        # First call
        result1 = searcher.search_material_prices("wood")
        # Second call should use cache
        result2 = searcher.search_material_prices("wood")
        
        assert result1 == result2
        # Search should only be called once due to caching
        assert mock_client_instance.search.call_count == 1

    @patch('door_schedule_parser.TavilyClient')
    def test_search_material_prices_no_results(self, mock_tavily_client):
        """Test material price search with no results (default values)"""
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        result = searcher.search_material_prices("exotic_material")
        
        # Should return default values
        assert result["price_per_sqm"] == 150
        assert result["installation"] == 60

    @patch('requests.post')
    def test_extract_door_schedule_json_success(self, mock_post):
        """Test successful JSON extraction from OCR text"""
        # Mock OpenRouter API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """```json
[
    {
        "door_id": "D-1",
        "count": 2,
        "width_cm": 90,
        "height_cm": 210,
        "operation": "swing",
        "finish": "wood",
        "remarks": "main entrance"
    }
]
```"""
                }
            }]
        }
        mock_post.return_value = mock_response
        
        ocr_text = "Door Schedule\nD-1 | 90x210 | wood | main entrance"
        result = extract_door_schedule_json(ocr_text)
        
        assert isinstance(result, list)
        assert len(result) == 1
        door = result[0]
        assert door["door_id"] == "D-1"
        assert door["count"] == 2
        assert door["width_cm"] == 90
        assert door["height_cm"] == 210

    @patch('requests.post')
    def test_extract_door_schedule_json_api_error(self, mock_post):
        """Test JSON extraction with API error"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        result = extract_door_schedule_json("some text")
        assert result == []

    @patch('requests.post')
    def test_extract_door_schedule_json_invalid_json(self, mock_post):
        """Test JSON extraction with invalid JSON response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is not valid JSON format"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = extract_door_schedule_json("some text")
        assert result == []

    @patch('requests.post')
    def test_extract_door_schedule_json_array_format(self, mock_post):
        """Test JSON extraction with array format (no code block)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """[{"door_id": "D-1", "count": 1, "width_cm": 80, "height_cm": 200, "operation": "swing", "finish": "metal", "remarks": "fire door"}]"""
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = extract_door_schedule_json("Door schedule text")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["door_id"] == "D-1"

    @patch('door_schedule_parser.TavilyClient')
    def test_calculate_costs_and_augment_basic(self, mock_tavily_client):
        """Test cost calculation and data augmentation"""
        # Mock Tavily client
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        searcher.search_material_prices = Mock(return_value={
            "price_per_sqm": 200,
            "installation": 80
        })
        
        doors_data = [{
            "door_id": "D-1",
            "count": 2,
            "width_cm": 90,
            "height_cm": 210,
            "operation": "swing",
            "finish": "wood",
            "remarks": "entrance"
        }]
        
        result = calculate_costs_and_augment(doors_data, searcher)
        
        assert len(result) == 1
        door = result[0]
        assert "area_sqm" in door
        assert "price_per_sqm" in door
        assert "installation_cost" in door
        assert "total_cost" in door
        assert door["area_sqm"] == calculate_area_sqm(90, 210)
        assert door["price_per_sqm"] == 200
        assert door["installation_cost"] == 80

    @patch('door_schedule_parser.TavilyClient')
    def test_calculate_costs_and_augment_error_handling(self, mock_tavily_client):
        """Test cost calculation with invalid data"""
        mock_client_instance = Mock()
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        
        # Invalid door data (missing required fields)
        doors_data = [{
            "door_id": "D-1",
            "finish": "wood"
            # Missing width_cm, height_cm, count
        }]
        
        result = calculate_costs_and_augment(doors_data, searcher)
        
        assert len(result) == 1
        door = result[0]
        assert door["area_sqm"] is None
        assert door["total_cost"] is None

    @patch('door_schedule_parser.TavilyClient')
    def test_calculate_costs_and_augment_multiple_doors(self, mock_tavily_client):
        """Test cost calculation for multiple doors"""
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance
        
        searcher = TavilyPriceSearcher("usa")
        searcher.search_material_prices = Mock(return_value={
            "price_per_sqm": 150,
            "installation": 60
        })
        
        doors_data = [
            {
                "door_id": "D-1",
                "count": 1,
                "width_cm": 90,
                "height_cm": 210,
                "finish": "wood"
            },
            {
                "door_id": "D-2", 
                "count": 2,
                "width_cm": 80,
                "height_cm": 200,
                "finish": "metal"
            }
        ]
        
        result = calculate_costs_and_augment(doors_data, searcher)
        
        assert len(result) == 2
        for door in result:
            assert "area_sqm" in door
            assert "total_cost" in door
            assert isinstance(door["total_cost"], (int, float))