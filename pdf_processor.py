import PyPDF2
import io
import numpy as np
from PIL import Image
from typing import Tuple
from fastapi import HTTPException
from config import Config
from paddleocr import PaddleOCR
import signal
import multiprocessing as mp
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

    def resize_image_if_needed(self, img: Image.Image, max_dimension: int = 1200) -> Image.Image:
        """Resize image if it's too large while maintaining aspect ratio"""
        width, height = img.size
        
        if width <= max_dimension and height <= max_dimension:
            return img
            
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
                ocr_result = self.ocr.ocr(img_array, cls=True)
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

    def extract_text_with_ocr(self, file_content: bytes, max_pages: int = 20) -> str:
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
            print(f"Processing {num_pages} pages with OCR (limited for speed)")
            
            ocr_text = ""
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                zoom_matrix = fitz.Matrix(1.2, 1.2)  
                pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)

                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                img = self.resize_image_if_needed(img, max_dimension=800)
                
                img_array = np.array(img)
                
                try:
                    print(f"Running OCR on image size: {img_array.shape}")
                    ocr_result = self.ocr.ocr(img_array)
                    print(f"OCR result type: {type(ocr_result)}")
                    
                    page_text = ""
                    if ocr_result:
                        if isinstance(ocr_result, list) and len(ocr_result) > 0:
                            print(f"OCR result length: {len(ocr_result)}")
                            
                            if len(ocr_result) == 1 and isinstance(ocr_result[0], dict):
                                result_dict = ocr_result[0]
                                print(f"Dictionary keys: {result_dict.keys()}")
                                
                                if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
                                    texts = result_dict['rec_texts']
                                    scores = result_dict['rec_scores']
                                    print(f"Found {len(texts)} text items")
                                    
                                    for text, score in zip(texts, scores):
                                        if score > 0.3 and text.strip(): 
                                            print(f"Adding text: '{text}' (score: {score:.3f})")
                                            page_text += text + " "
                                        else:
                                            print(f"Skipping: '{text}' (score: {score:.3f})")
                                    page_text += "\n"
                            
                            else:
                                for block_idx, result_block in enumerate(ocr_result):
                                    print(f"Block {block_idx}: {type(result_block)}")
                                    if result_block is None:
                                        print(f"Block {block_idx} is None")
                                        continue
                                        
                                    if isinstance(result_block, dict):
                                        print(f"Block {block_idx} is dict: {result_block.keys()}")
                                        if 'rec_texts' in result_block and 'rec_scores' in result_block:
                                            texts = result_block['rec_texts']
                                            scores = result_block['rec_scores']
                                            for text, score in zip(texts, scores):
                                                if score > 0.3 and text.strip():
                                                    page_text += text + " "
                                            page_text += "\n"
                                        elif 'text' in result_block:
                                            page_text += result_block['text'] + " "
                                    
                                    elif isinstance(result_block, list):
                                        print(f"Block {block_idx} has {len(result_block)} lines")
                                        for line_idx, line in enumerate(result_block):
                                            if line and len(line) >= 2:
                                                if isinstance(line[1], list) and len(line[1]) >= 2:
                                                    text = line[1][0]
                                                    confidence = line[1][1]
                                                    if confidence > 0.3 and text.strip():
                                                        page_text += text + " "
                                                elif isinstance(line[1], str):
                                                    page_text += line[1] + " "
                                        page_text += "\n"
                        
                        elif isinstance(ocr_result, dict):
                            print("OCR result is direct dictionary")
                            if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                                texts = ocr_result['rec_texts']
                                scores = ocr_result['rec_scores']
                                print(f"Found {len(texts)} text items")
                                
                                for text, score in zip(texts, scores):
                                    if score > 0.3 and text.strip():
                                        print(f"Adding text: '{text}' (score: {score:.3f})")
                                        page_text += text + " "
                                    else:
                                        print(f"Skipping: '{text}' (score: {score:.3f})")
                                page_text += "\n"
                    
                    print(f"Final page_text length: {len(page_text)}")
                    print(f"Page text preview: '{page_text[:200]}'")
                    ocr_text += page_text
                    print(f"Page {page_num + 1}: {len(page_text)} chars")
                    
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            doc.close()
            print(f"Total OCR text extracted: {len(ocr_text)} characters")
            return ocr_text.strip()
            
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            return ""

    def should_use_ocr(self, pdf_text: str, num_pages: int) -> bool:
        """Determine if OCR is needed based on extracted text quality"""
        if not pdf_text or len(pdf_text.strip()) < 50:
            return True
        
        words = pdf_text.split()
        if len(words) < num_pages * 10: 
            return True
        
        special_chars = sum(1 for c in pdf_text if not c.isalnum() and c not in ' \n\t.,!?-()[]{}":;')
        if special_chars > len(pdf_text) * 0.3:  
            return True
            
        return False

    def extract_text(self, file_content: bytes, force_ocr: bool = True) -> Tuple[str, int]:
        """Combine PDF text extraction and OCR - always OCR by default"""
        try:
            pdf_text, num_pages = self.extract_text_from_pdf(file_content)
            print(f"[PDF] Extracted {len(pdf_text)} characters from {num_pages} pages")
        except Exception as e:
            print(f"[PDF] Text extraction failed: {e}")
            pdf_text, num_pages = "", 0

        ocr_text = ""
        if force_ocr:
            print("[OCR] Starting OCR extraction (forced)...")
            try:
                ocr_text = self.extract_text_with_ocr(file_content)
                print(f"[OCR] Extracted {len(ocr_text)} characters")
            except Exception as e:
                print(f"[OCR] OCR extraction failed: {e}")

        if len(ocr_text) > len(pdf_text) * 1.5: 
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
                detail="PDF contains no extractable text even after OCR"
            )

        combined_text = combined_text.strip()
        print(f"[COMBINED] Final text: {len(combined_text)} characters")
        
        return combined_text, num_pages if num_pages > 0 else 1