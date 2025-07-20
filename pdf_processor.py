import PyPDF2
import io
from typing import Tuple
from fastapi import HTTPException
from config import Config
from paddleocr import PaddleOCR

class PDFProcessor:
    """Process PDF files"""

    def __init__(self):
        self.config = Config()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en') 

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
        """Extract text from PDF pages using PyPDF2"""
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

    def extract_text_with_ocr(self, file_content: bytes) -> str:
        """Run OCR on each page image extracted from PDF"""
        try:
            import fitz 
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PyMuPDF (fitz) is required for OCR image extraction"
            )

        pdf_file = io.BytesIO(file_content)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        ocr_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom_matrix = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)

            img_bytes = pix.tobytes("png")

            ocr_result = self.ocr.ocr(img_bytes, cls=True)

            for line in ocr_result:
                for word_info in line:
                    ocr_text += word_info[1][0] + " "
                ocr_text += "\n"

        return ocr_text

    def extract_text(self, file_content: bytes) -> Tuple[str, int]:
        """Combine PDF text extraction and OCR"""
        pdf_text, num_pages = self.extract_text_from_pdf(file_content)
        ocr_text = self.extract_text_with_ocr(file_content)

        combined_text = (pdf_text + "\n" + ocr_text).strip()
        if not combined_text:
            raise HTTPException(
                status_code=400,
                detail="PDF contains no extractable text even after OCR"
            )

        return combined_text, num_pages