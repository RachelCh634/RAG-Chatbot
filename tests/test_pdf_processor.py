import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import os

# Mock environment variables for testing
with patch.dict(os.environ, {
    'OPENROUTER_API_KEY': 'test_openrouter_key',
    'TAVILY_API_KEY': 'test_tavily_key'
}):

    from pdf_processor import PDFProcessor  
class TestPDFProcessor:
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_init(self, mock_paddle_ocr, mock_config):
        """Test PDFProcessor initialization"""
        mock_config_instance = Mock()
        mock_config_instance.MAX_FILE_SIZE = 50 * 1024 * 1024
        mock_config.return_value = mock_config_instance
        
        mock_ocr_instance = Mock()
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        processor = PDFProcessor()
        assert processor is not None
        assert processor.config == mock_config_instance
        assert processor.ocr == mock_ocr_instance
        mock_paddle_ocr.assert_called_once_with(use_angle_cls=True, lang='en')
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_validate_file_extension(self, mock_paddle_ocr, mock_config):
        mock_config_instance = Mock()
        mock_config_instance.MAX_FILE_SIZE = 50 * 1024 * 1024
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.validate_file("test.txt", b"some content")
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_validate_file_size(self, mock_paddle_ocr, mock_config):
        mock_config_instance = Mock()
        mock_config_instance.MAX_FILE_SIZE = 50 * 1024 * 1024
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        large_content = b"x" * (51 * 1024 * 1024)
        with pytest.raises(Exception):
            processor.validate_file("test.pdf", large_content)
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_validate_file_empty(self, mock_paddle_ocr, mock_config):
        mock_config_instance = Mock()
        mock_config_instance.MAX_FILE_SIZE = 50 * 1024 * 1024
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.validate_file("test.pdf", b"")
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf_basic(self, mock_pdf_reader, mock_paddle_ocr, mock_config):
        """Test basic PDF text extraction"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample text from page"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page, mock_page]
        mock_reader_instance.is_encrypted = False
        mock_pdf_reader.return_value = mock_reader_instance
        
        processor = PDFProcessor()
        text, num_pages = processor.extract_text_from_pdf(b"fake pdf content")
        
        assert isinstance(text, str)
        assert num_pages == 2
        assert "Sample text from page" in text
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf_encrypted(self, mock_pdf_reader, mock_paddle_ocr, mock_config):
        """Test handling of encrypted PDF"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [Mock()]
        mock_reader_instance.is_encrypted = True
        mock_pdf_reader.return_value = mock_reader_instance
        
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.extract_text_from_pdf(b"encrypted pdf content")
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_pdf_has_images_no_fitz(self, mock_paddle_ocr, mock_config):
        """Test pdf_has_images when PyMuPDF is not available"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'fitz'")):
            result = processor.pdf_has_images(b"fake pdf content")
            assert result is False
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    @patch('fitz.open')
    def test_pdf_has_images_with_images(self, mock_fitz_open, mock_paddle_ocr, mock_config):
        """Test pdf_has_images when images are present"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        # Mock fitz document with images
        mock_page = Mock()
        mock_page.get_images.return_value = [{"image": "data"}]  # Has images
        mock_page.get_drawings.return_value = []
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        processor = PDFProcessor()
        result = processor.pdf_has_images(b"fake pdf content")
        assert result is True
        mock_doc.close.assert_called_once()
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    @patch('fitz.open')
    def test_pdf_has_images_no_images(self, mock_fitz_open, mock_paddle_ocr, mock_config):
        """Test pdf_has_images when no images are present"""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # Mock fitz document without images
        mock_page = Mock()
        mock_page.get_images.return_value = []  # No images
        mock_page.get_drawings.return_value = []  # No significant drawings
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        processor = PDFProcessor()
        result = processor.pdf_has_images(b"fake pdf content")
        assert result is False
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_check_text_quality_good(self, mock_paddle_ocr, mock_config):
        """Test text quality check with good quality text"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        good_text = "This is a well-formatted document with proper text content. " * 10
        result = processor.check_text_quality(good_text, 1)
        
        assert result['quality'] == 'good'
        assert result['needs_ocr'] is False
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_check_text_quality_poor_insufficient(self, mock_paddle_ocr, mock_config):
        """Test text quality check with insufficient text"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        result = processor.check_text_quality("short", 1)
        
        assert result['quality'] == 'poor'
        assert result['reason'] == 'insufficient_text'
        assert result['needs_ocr'] is True
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_check_text_quality_poor_few_words(self, mock_paddle_ocr, mock_config):
        """Test text quality check with too few words per page"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        text = "Few words only here in this document that should be longer"
        result = processor.check_text_quality(text, 5)  # 5 pages, few words per page
        
        assert result['quality'] == 'poor'
        assert result['reason'] == 'too_few_words_per_page'
        assert result['needs_ocr'] is True
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_should_use_ocr_force(self, mock_paddle_ocr, mock_config):
        """Test should_use_ocr with force_ocr=True"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        result = processor.should_use_ocr("good text", 1, False, force_ocr=True)
        
        assert result['use_ocr'] is True
        assert result['reason'] == 'forced_by_user'
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_should_use_ocr_has_images(self, mock_paddle_ocr, mock_config):
        """Test should_use_ocr when PDF has images"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        result = processor.should_use_ocr("good text", 1, True, force_ocr=False)
        
        assert result['use_ocr'] is True
        assert result['reason'] == 'has_images_always_run_ocr'
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_resize_image_if_needed(self, mock_paddle_ocr, mock_config):
        """Test image resizing functionality"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        
        # Create a mock image
        mock_img = Mock(spec=Image.Image)
        mock_img.size = (2000, 1500)  # Large image
        mock_resized = Mock(spec=Image.Image)
        mock_img.resize.return_value = mock_resized
        
        result = processor.resize_image_if_needed(mock_img, max_dimension=1200)
        
        # Should call resize since image is larger than max_dimension
        mock_img.resize.assert_called_once()
        assert result == mock_resized
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_process_ocr_result_empty(self, mock_paddle_ocr, mock_config):
        """Test _process_ocr_result with empty result"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        result = processor._process_ocr_result(None)
        assert result == ""
        
        result = processor._process_ocr_result([])
        assert result == ""
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_process_ocr_result_with_text(self, mock_paddle_ocr, mock_config):
        """Test _process_ocr_result with valid OCR data"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        
        # Mock OCR result structure
        ocr_result = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], ["Sample text", 0.9]],
                [[[0, 40], [100, 40], [100, 70], [0, 70]], ["More text", 0.8]]
            ]
        ]
        
        result = processor._process_ocr_result(ocr_result)
        assert "Sample text" in result
        assert "More text" in result
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')  
    @patch('fitz.open')
    def test_extract_text_with_ocr_basic(self, mock_fitz_open, mock_paddle_ocr, mock_config):
        """Test OCR text extraction"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        # Mock OCR instance
        mock_ocr_instance = MagicMock()
        mock_ocr_instance.ocr.return_value = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], ["OCR extracted text", 0.9]]
            ]
        ]
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        # Mock fitz document
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake image bytes"
        
        mock_page = Mock()
        mock_page.get_pixmap.return_value = mock_pixmap
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        # Mock PIL Image
        with patch('PIL.Image.open') as mock_image_open:
            mock_img = Mock()
            mock_img_array = np.array([[1, 2, 3]])
            
            with patch('numpy.array', return_value=mock_img_array):
                mock_image_open.return_value = mock_img
                processor = PDFProcessor()
                processor.resize_image_if_needed = Mock(return_value=mock_img)
                
                result = processor.extract_text_with_ocr(b"fake pdf content", max_pages=1)
                
                assert isinstance(result, str)
                mock_doc.close.assert_called_once()
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_extract_text_integration(self, mock_paddle_ocr, mock_config):
        """Test the main extract_text method integration"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        
        # Mock all the methods used in extract_text
        processor.extract_text_from_pdf = Mock(return_value=("PDF text content", 1))
        processor.pdf_has_images = Mock(return_value=False)
        processor.should_use_ocr = Mock(return_value={'use_ocr': False, 'reason': 'good_text'})
        processor.extract_text_with_ocr = Mock(return_value="OCR text content")
        
        text, num_pages = processor.extract_text(b"fake pdf content", force_ocr=False)
        
        assert isinstance(text, str)
        assert isinstance(num_pages, int)
        assert text == "PDF text content"
        processor.extract_text_from_pdf.assert_called_once()
        processor.pdf_has_images.assert_called_once()
        processor.should_use_ocr.assert_called_once()
    
    @patch('pdf_processor.Config')
    @patch('pdf_processor.PaddleOCR')
    def test_extract_text_ocr_fallback(self, mock_paddle_ocr, mock_config):
        """Test extract_text when OCR is needed"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        processor = PDFProcessor()
        
        # Mock scenario where OCR is needed
        processor.extract_text_from_pdf = Mock(return_value=("", 1))
        processor.pdf_has_images = Mock(return_value=True)
        processor.should_use_ocr = Mock(return_value={'use_ocr': True, 'reason': 'has_images'})
        processor.extract_text_with_ocr = Mock(return_value="OCR extracted text")
        
        text, num_pages = processor.extract_text(b"fake pdf content", force_ocr=False)
        
        assert text == "OCR extracted text"
        processor.extract_text_with_ocr.assert_called_once()
