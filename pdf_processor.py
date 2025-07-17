import PyPDF2
import io
from typing import Tuple, List
from fastapi import HTTPException
from config import Config

class PDFProcessor:
    """Process PDF files"""
    
    def __init__(self):
        self.config = Config()
    
    def validate_file(self, filename: str, file_content: bytes) -> None:
        """Checking PDF file integrity"""
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="The file extension must be PDF."
            )
        
        file_size = len(file_content)
        if file_size > self.config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {self.config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
    
    def extract_text_from_pdf(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            num_pages = len(pdf_reader.pages)
            if num_pages == 0:
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no pages"
                )
            
            if pdf_reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password protected PDF not supported"
                )
            
            full_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            if not full_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no extractable text"
                )
            
            return full_text, num_pages
            
        except PyPDF2.errors.PdfReadError:
            raise HTTPException(
                status_code=400,
                detail="PDF file is corrupt or invalid"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )