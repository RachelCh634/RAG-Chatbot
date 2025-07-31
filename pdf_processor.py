import PyPDF2
import io
import numpy as np
from PIL import Image
from typing import Tuple
from fastapi import HTTPException
from config import Config
from paddleocr import PaddleOCR
import signal
from concurrent.futures import TimeoutError

class PDFProcessor:
    """Process PDF files with optimized OCR"""

    def __init__(self):
        self.config = Config()
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en',
        ) 

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

    def pdf_has_images(self, file_content: bytes) -> bool:
        """Check if PDF contains images using PyMuPDF"""
        try:
            import fitz 
        except ImportError:
            print("PyMuPDF not available - assuming NO images exist (will check text quality)")
            return False 

        try:
            pdf_file = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            has_images = False
            max_pages_to_check = min(5, len(doc))
            print(f"Checking {max_pages_to_check} pages for images...")
            
            for page_num in range(max_pages_to_check):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=False)
                
                if image_list:
                    print(f"Found {len(image_list)} images on page {page_num + 1}")
                    has_images = True
                    break
                else:
                    print(f"No images found on page {page_num + 1}")
                    
                drawings = page.get_drawings()
                if drawings:
                    significant_drawings = [d for d in drawings if len(d.get('items', [])) > 5]
                    if significant_drawings:
                        print(f"Found {len(significant_drawings)} significant drawings on page {page_num + 1}")
                        has_images = True
                        break
                    else:
                        print(f"Found {len(drawings)} simple drawings (likely borders/lines) on page {page_num + 1}")

            doc.close()
            print(f"[IMAGE CHECK] PDF contains images: {has_images}")
            return has_images
            
        except Exception as e:
            print(f"Error checking for images: {e}")
            import traceback
            traceback.print_exc()
            print("Assuming NO images exist due to error (will rely on text quality check)")
            return False  

    def check_text_quality(self, text: str, num_pages: int) -> dict:
        """Analyze text quality to determine if OCR is needed"""
        if not text or len(text.strip()) < 50:
            return {
                'quality': 'poor',
                'reason': 'insufficient_text',
                'needs_ocr': True
            }
        
        words = text.split()
        words_per_page = len(words) / num_pages if num_pages > 0 else 0
        
        if words_per_page < 20:
            return {
                'quality': 'poor',
                'reason': 'too_few_words_per_page',
                'needs_ocr': True
            }
        
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?-()[]{}":;\'')
        special_char_ratio = special_chars / len(text) if len(text) > 0 else 0
        
        if special_char_ratio > 0.3:
            return {
                'quality': 'poor',
                'reason': 'too_many_special_chars',
                'needs_ocr': True
            }
        
        if len(set(text.replace(' ', '').replace('\n', ''))) < 10:
            return {
                'quality': 'poor',
                'reason': 'repetitive_chars',
                'needs_ocr': True
            }
        
        return {
            'quality': 'good',
            'reason': 'text_extracted_successfully',
            'needs_ocr': False
        }

    def should_use_ocr(self, pdf_text: str, num_pages: int, has_images: bool, force_ocr: bool = False) -> dict:
        """Enhanced logic to determine if OCR is needed"""
        
        if force_ocr:
            return {
                'use_ocr': True,
                'reason': 'forced_by_user'
            }
        
        if has_images:
            return {
                'use_ocr': True,
                'reason': 'has_images_always_run_ocr'
            }
        
        text_quality = self.check_text_quality(pdf_text, num_pages)
        if not text_quality['needs_ocr']:
            return {
                'use_ocr': False,
                'reason': f'no_images_and_good_text_quality: {text_quality["reason"]}'
            }
        else:
            return {
                'use_ocr': True,
                'reason': f'no_images_but_poor_text: {text_quality["reason"]}'
            }

    def resize_image_if_needed(self, img: Image.Image, max_dimension: int = 1200) -> Image.Image:
        """Resize image if it's too large while maintaining aspect ratio"""
        width, height = img.size
        
        # if width <= max_dimension and height <= max_dimension:
        #     return img
                # גם אם התמונה קטנה – נגדיל אותה למימדים החדשים
        if width > height:
            new_width = max_dimension
            new_height = int((height * max_dimension) / width)
        else:
            new_height = max_dimension
            new_width = int((width * max_dimension) / height)
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            
        if width > height:
            new_width = max_dimension
            new_height = int((height * max_dimension) / width)
        else:
            new_height = max_dimension
            new_width = int((width * max_dimension) / height)
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def process_single_page_ocr(self, page_data):
        """Process a single page with OCR - for parallel processing"""
        page_num, img_array = page_data
        
        try:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"OCR timeout for page {page_num + 1}")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  
            
            try:
                ocr_result = self.ocr.ocr(img_array)
                signal.alarm(0)  
            except Exception as e:
                signal.alarm(0)
                print(f"OCR failed for page {page_num + 1}: {e}")
                return ""
            
            page_text = ""
            if ocr_result:
                if isinstance(ocr_result, list) and len(ocr_result) > 0:
                    for result_block in ocr_result:
                        if result_block is None:
                            continue
                            
                        if isinstance(result_block, list):
                            for line in result_block:
                                if line and len(line) >= 2:
                                    if isinstance(line[1], list) and len(line[1]) >= 2:
                                        text = line[1][0]
                                        confidence = line[1][1]
                                        if confidence > 0.6 and text.strip():  
                                            page_text += text + " "
                            page_text += "\n"
                        
                        elif isinstance(result_block, dict):
                            if 'text' in result_block:
                                page_text += result_block['text'] + " "
            
            return page_text.strip()
            
        except TimeoutError:
            print(f"Page {page_num + 1} timed out")
            return ""
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            return ""

    def extract_text_with_ocr(self, file_content: bytes, max_pages: int = 10) -> str:
        """Run OCR on PDF pages with optimizations for speed"""
        try:
            import fitz 
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PyMuPDF (fitz) is required for OCR image extraction"
            )

        try:
            pdf_file = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            num_pages = min(len(doc), max_pages)
            print(f"Processing {num_pages} pages with OCR (max {max_pages} for speed)")
            
            ocr_text = ""
            for page_num in range(num_pages):
                print(f"Starting OCR on page {page_num + 1}/{num_pages}")
                page = doc.load_page(page_num)
                
                zoom_matrix = fitz.Matrix(1.0, 1.0)  
                pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)

                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                
                img = self.resize_image_if_needed(img, max_dimension=1900)
                img_array = np.array(img)
                
                try:
                    print(f"Running OCR on page {page_num + 1}, image size: {img_array.shape}")
                    
                    ocr_result = self.ocr.ocr(img_array)  
                    
                    page_text = self._process_ocr_result(ocr_result)
                    ocr_text += page_text
                    
                    print(f"Page {page_num + 1} completed: {len(page_text)} chars extracted")
                    
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {e}")
                    continue

            doc.close()
            print(f"Total OCR text extracted: {len(ocr_text)} characters")
            return ocr_text.strip()
            
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            return ""

    def _process_ocr_result(self, ocr_result) -> str:
        """Process OCR result efficiently without debug prints"""
        page_text = ""
        
        if not ocr_result:
            return page_text
            
        try:
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                for result_block in ocr_result:
                    if result_block is None:
                        continue
                        
                    if isinstance(result_block, list):
                        for line in result_block:
                            if line and len(line) >= 2 and isinstance(line[1], list) and len(line[1]) >= 2:
                                text = line[1][0]
                                confidence = line[1][1]
                                if confidence > 0.4 and text.strip():  
                                    page_text += text + " "
                        page_text += "\n"
                    
                    elif isinstance(result_block, dict):
                        if 'rec_texts' in result_block and 'rec_scores' in result_block:
                            texts = result_block['rec_texts']
                            scores = result_block['rec_scores']
                            for text, score in zip(texts, scores):
                                if score > 0.3 and text.strip():
                                    page_text += text + " "
                            page_text += "\n"
                        elif 'text' in result_block:
                            page_text += result_block['text'] + " "
            
            elif isinstance(ocr_result, dict):
                if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                    texts = ocr_result['rec_texts']
                    scores = ocr_result['rec_scores']
                    for text, score in zip(texts, scores):
                        if score > 0.3 and text.strip():
                            page_text += text + " "
                    page_text += "\n"
                    
        except Exception as e:
            print(f"Error processing OCR result: {e}")
            
        return page_text

    def extract_text(self, file_content: bytes, force_ocr: bool = False) -> Tuple[str, int]:
        """Smart text extraction - OCR only when necessary"""
        try:
            pdf_text, num_pages = self.extract_text_from_pdf(file_content)
            print(f"[PDF] Extracted {len(pdf_text)} characters from {num_pages} pages")
        except Exception as e:
            print(f"[PDF] Text extraction failed: {e}")
            pdf_text, num_pages = "", 0

        has_images = self.pdf_has_images(file_content)
        
        ocr_decision = self.should_use_ocr(pdf_text, num_pages, has_images, force_ocr)
        print(f"[DECISION] OCR needed: {ocr_decision['use_ocr']}, Reason: {ocr_decision['reason']}")

        ocr_text = ""
        if ocr_decision['use_ocr']:
            print("[OCR] Starting OCR extraction...")
            try:
                ocr_text = self.extract_text_with_ocr(file_content)
                print(f"[OCR] Extracted {len(ocr_text)} characters")
            except Exception as e:
                print(f"[OCR] OCR extraction failed: {e}")
        else:
            print("[OCR] Skipping OCR - not needed")

        if ocr_text and len(ocr_text) > len(pdf_text) * 1.5: 
            combined_text = ocr_text
            print("[COMBINED] Using OCR text as primary")
        elif pdf_text and ocr_text:  
            combined_text = pdf_text + "\n\n--- OCR SUPPLEMENT ---\n\n" + ocr_text
            print("[COMBINED] Using both PDF and OCR text")
        elif pdf_text:
            combined_text = pdf_text
            print("[COMBINED] Using PDF text only")
        elif ocr_text:
            combined_text = ocr_text
            print("[COMBINED] Using OCR text only")
        else:
            raise HTTPException(
                status_code=400,
                detail="PDF contains no extractable text"
            )

        combined_text = combined_text.strip()
        print(f"[COMBINED] Final text: {len(combined_text)} characters")
        
        return combined_text, num_pages if num_pages > 0 else 1