import PyPDF2
import io
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List
from fastapi import HTTPException
from config import Config
import pandas as pd
import re

class PDFProcessor:
    """Process PDF files with optimized OCR and table extraction"""

    def __init__(self):
        self.config = Config()
        self.ocr_engine = None

        self.table_extractor = None
        self._init_ocr_engine()
        self._init_table_extractor()

    def _init_ocr_engine(self):
        """Initialize OCR engine - prefer Tesseract over PaddleOCR"""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  
            self.ocr_engine = 'tesseract'
            print("[OCR] Using Tesseract OCR engine")
        except ImportError:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                self.ocr_engine = 'paddle'
                print("[OCR] Using PaddleOCR engine")
            except ImportError:
                print("[OCR] No OCR engine available")
                self.ocr_engine = None

    def _init_table_extractor(self):
        """Initialize table extraction engines"""
        self.table_engines = {}
        
        try:
            import camelot
            self.table_engines['camelot'] = camelot
            print("[TABLE] Camelot available")
        except ImportError:
            print("[TABLE] Camelot not available")
        
        try:
            import tabula
            self.table_engines['tabula'] = tabula
            print("[TABLE] Tabula available")
        except ImportError:
            print("[TABLE] Tabula not available")
        
        try:
            import pdfplumber
            self.table_engines['pdfplumber'] = pdfplumber
            print("[TABLE] pdfplumber available")
        except ImportError:
            print("[TABLE] pdfplumber not available")

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

    def pdf_has_images_or_tables(self, file_content: bytes) -> Dict[str, bool]:
        """Check if PDF contains images or table-like structures - IMPROVED"""
        try:
            import fitz 
        except ImportError:
            print("PyMuPDF not available - assuming NO images exist")
            return {'has_images': False, 'likely_has_tables': False}

        try:
            pdf_file = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            has_images = False
            likely_has_tables = False
            
            for page_num in range(min(len(doc), 5)): 
                page = doc.load_page(page_num)
                
                image_list = page.get_images(full=False)
                if image_list:
                    has_images = True
                
                text = page.get_text()
                
                drawings = page.get_drawings()
                table_score = self._analyze_table_structure(drawings, text)
                
                if table_score > 0.5:
                    likely_has_tables = True
                    print(f"Found table structure on page {page_num + 1} (score: {table_score:.2f})")

            doc.close()
            
            result = {
                'has_images': has_images,
                'likely_has_tables': likely_has_tables
            }
            print(f"[ANALYSIS] Images: {has_images}, Tables: {likely_has_tables}")
            return result
            
        except Exception as e:
            print(f"Error checking for images/tables: {e}")
            return {'has_images': False, 'likely_has_tables': False}

    def _analyze_table_structure(self, drawings: list, text: str) -> float:
        """Analyze if page contains table structure - NEW IMPROVED METHOD"""
        score = 0.0
        
        if drawings:
            horizontal_lines = 0
            vertical_lines = 0
            
            for drawing in drawings:
                items = drawing.get('items', [])
                for item in items:
                    if item[0] == 'l' and len(item) >= 5:  
                        try:
                            x1, y1, x2, y2 = item[1:5]
                            if abs(y1 - y2) < 3:  
                                horizontal_lines += 1
                            elif abs(x1 - x2) < 3:  
                                vertical_lines += 1
                        except (ValueError, IndexError):
                            continue
            
            if horizontal_lines >= 3 and vertical_lines >= 2:
                score += 0.4
                print(f"Found grid structure: {horizontal_lines}H, {vertical_lines}V")

        if text:
            text_score = self._analyze_text_for_tables(text)
            score += text_score
        
        return min(score, 1.0)

    def _analyze_text_for_tables(self, text: str) -> float:
        """Analyze text patterns for table indicators - IMPROVED"""
        if not text:
            return 0.0
        
        score = 0.0
        lines = text.split('\n')
        
        table_keywords = [
            'schedule', 'table', 'total', 'amount', 'quantity', 'price',
            'description', 'item', 'date', 'name', 'type', 'category',
            'code', 'id', 'number', 'sum', 'balance', 'count'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in table_keywords if keyword in text_lower)
        
        if keyword_matches >= 3:
            score += 0.2
            print(f"Found {keyword_matches} table keywords")

        aligned_lines = 0
        numeric_columns = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if re.search(r'\s{3,}', line) or '\t' in line:
                aligned_lines += 1
                
                parts = re.split(r'\s{2,}|\t', line)
                if len(parts) >= 2:
                    numeric_parts = sum(1 for part in parts 
                                      if re.search(r'\d+[.,]?\d*', part.strip()))
                    if numeric_parts >= 1:
                        numeric_columns += 1

        if aligned_lines >= 3:
            score += 0.2
            print(f"Found {aligned_lines} aligned lines, {numeric_columns} with numbers")

        number_patterns = len(re.findall(r'\b\d{1,3}[,.]?\d{0,3}[,.]?\d{0,2}\b', text))
        if number_patterns >= 5:
            score += 0.1

        return score

    def extract_tables_with_camelot_improved(self, file_content: bytes) -> List[pd.DataFrame]:
        """Extract tables using Camelot with better settings"""
        if 'camelot' not in self.table_engines:
            return []
        
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            tables = []
            camelot = self.table_engines['camelot']
            
            lattice_settings = [
                {'flavor': 'lattice'},
                {'flavor': 'lattice', 'line_scale': 15},
                {'flavor': 'lattice', 'copy_text': ['v'], 'split_text': True},
            ]
            
            for settings in lattice_settings:
                try:
                    found_tables = camelot.read_pdf(tmp_file_path, **settings)
                    for table in found_tables:
                        if (not table.df.empty and 
                            table.accuracy > 70 and  
                            table.df.shape[0] > 1 and 
                            table.df.shape[1] > 1):
                            
                            cleaned_df = self._clean_extracted_table(table.df)
                            if not cleaned_df.empty:
                                tables.append(cleaned_df)
                                print(f"[CAMELOT-LATTICE] Found table: {cleaned_df.shape}, accuracy: {table.accuracy:.1f}%")
                    
                    if tables:
                        break
                        
                except Exception as e:
                    print(f"[CAMELOT-LATTICE] Settings {settings} failed: {e}")
                    continue

            if not tables:
                stream_settings = [
                    {'flavor': 'stream'},
                    {'flavor': 'stream', 'row_tol': 3},
                    {'flavor': 'stream', 'column_tol': 3},
                ]
                
                for settings in stream_settings:
                    try:
                        found_tables = camelot.read_pdf(tmp_file_path, **settings)
                        for table in found_tables:
                            if (not table.df.empty and 
                                table.accuracy > 50 and  
                                table.df.shape[0] > 1):
                                
                                cleaned_df = self._clean_extracted_table(table.df)
                                if not cleaned_df.empty:
                                    tables.append(cleaned_df)
                                    print(f"[CAMELOT-STREAM] Found table: {cleaned_df.shape}, accuracy: {table.accuracy:.1f}%")
                        if tables:
                            break
                            
                    except Exception as e:
                        print(f"[CAMELOT-STREAM] Settings {settings} failed: {e}")
                        continue
            os.unlink(tmp_file_path)
            return tables
            
        except Exception as e:
            print(f"[CAMELOT] Error: {e}")
            return []

    def extract_tables_with_pdfplumber_improved(self, file_content: bytes) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber with better settings - FIXED"""
        if 'pdfplumber' not in self.table_engines:
            return []
        
        try:
            pdfplumber = self.table_engines['pdfplumber']
            tables = []
            
            pdf_file = io.BytesIO(file_content)
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    strategies = [
                        {}, 
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"snap_tolerance": 3, "join_tolerance": 3},
                    ]
                    
                    for strategy in strategies:
                        try:
                            page_tables = page.extract_tables(table_settings=strategy)
                            
                            for table_data in page_tables:
                                if table_data and len(table_data) > 1:
                                    non_empty_rows = []
                                    for row in table_data:
                                        if row and any(cell and str(cell).strip() for cell in row):
                                            cleaned_row = []
                                            for cell in row:
                                                if cell is not None:
                                                    cleaned_row.append(str(cell).strip())
                                                else:
                                                    cleaned_row.append('')
                                            non_empty_rows.append(cleaned_row)
                                    
                                    if len(non_empty_rows) >= 2:
                                        try:
                                            max_cols = max(len(row) for row in non_empty_rows)
                                            normalized_rows = []
                                            
                                            for row in non_empty_rows:
                                                while len(row) < max_cols:
                                                    row.append('')
                                                normalized_rows.append(row[:max_cols])
                                            
                                            if len(normalized_rows) > 1:
                                                headers = normalized_rows[0]
                                                data_rows = normalized_rows[1:]
                                                
                                                clean_headers = []
                                                for i, header in enumerate(headers):
                                                    if header and str(header).strip():
                                                        clean_headers.append(str(header).strip())
                                                    else:
                                                        clean_headers.append(f'Column_{i+1}')
                                                
                                                df = pd.DataFrame(data_rows, columns=clean_headers)
                                                cleaned_df = self._clean_extracted_table(df)
                                                
                                                if not cleaned_df.empty and cleaned_df.shape[1] >= 2:
                                                    tables.append(cleaned_df)
                                                    print(f"[PDFPLUMBER] Found table on page {page_num + 1}: {cleaned_df.shape}")
                                                    break
                                                    
                                        except Exception as e:
                                            print(f"[PDFPLUMBER] Error creating DataFrame on page {page_num + 1}: {e}")
                                            continue
                            
                            if len(tables) > 0:  
                                break
                                
                        except Exception as e:
                            print(f"[PDFPLUMBER] Strategy {strategy} failed on page {page_num + 1}: {e}")
                            continue
            return tables
        except Exception as e:
            print(f"[PDFPLUMBER] General error: {e}")
            return []

    def extract_tables_with_tabula(self, file_content: bytes) -> List[pd.DataFrame]:
        """Extract tables using tabula-py as additional fallback"""
        if 'tabula' not in self.table_engines:
            return []
        
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            tables = []
            tabula = self.table_engines['tabula']
            
            try:
                strategies = [
                    {"lattice": True},
                    {"stream": True},
                    {"guess": True},
                ]
                
                for strategy in strategies:
                    try:
                        dfs = tabula.read_pdf(tmp_file_path, pages='all', **strategy)
                        
                        if isinstance(dfs, list):
                            for df in dfs:
                                if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
                                    cleaned_df = self._clean_extracted_table(df)
                                    if not cleaned_df.empty:
                                        tables.append(cleaned_df)
                                        print(f"[TABULA] Found table: {cleaned_df.shape}")
                        
                        if tables: 
                            break
                            
                    except Exception as e:
                        print(f"[TABULA] Strategy {strategy} failed: {e}")
                        continue
                        
            except Exception as e:
                print(f"[TABULA] All strategies failed: {e}")
            
            os.unlink(tmp_file_path)
            return tables
            
        except Exception as e:
            print(f"[TABULA] Error: {e}")
            return []

    def _clean_extracted_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted table - FIXED VERSION"""
        if df.empty:
            return df
        
        try:
            df = df.reset_index(drop=True)
            non_empty_rows = []
            for idx in df.index:
                row = df.iloc[idx]
                if not all(pd.isna(val) or str(val).strip() == '' for val in row):
                    non_empty_rows.append(idx)
            
            if not non_empty_rows:
                return pd.DataFrame()
            
            df = df.iloc[non_empty_rows]
            
            non_empty_cols = []
            for col in df.columns:
                if not all(pd.isna(val) or str(val).strip() == '' for val in df[col]):
                    non_empty_cols.append(col)
            
            if not non_empty_cols:
                return pd.DataFrame()
            
            df = df[non_empty_cols]
            
            new_columns = []
            for i, col in enumerate(df.columns):
                if col is not None and str(col).strip():
                    new_columns.append(str(col).strip())
                else:
                    new_columns.append(f'Col_{i}')
            df.columns = new_columns
            
            for col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else '')
            
            df = df.drop_duplicates().reset_index(drop=True)
            
            if df.shape[0] < 1 or df.shape[1] < 2:
                return pd.DataFrame()
            
            non_empty_cells = 0
            total_cells = df.shape[0] * df.shape[1]
            
            for col in df.columns:
                for val in df[col]:
                    if str(val).strip():
                        non_empty_cells += 1
            
            if total_cells > 0 and non_empty_cells / total_cells < 0.3:
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"[CLEAN_TABLE] Error cleaning table: {e}")
            try:
                return df.reset_index(drop=True)
            except:
                return pd.DataFrame()

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
        
        return {
            'quality': 'good',
            'reason': 'text_extracted_successfully',
            'needs_ocr': False
        }

    def should_use_ocr(self, pdf_text: str, num_pages: int, analysis: Dict[str, bool], force_ocr: bool = False) -> dict:
        """Determine if OCR is needed, always use OCR when images are present."""

        if force_ocr:
            return {
                'use_ocr': True,
                'reason': 'forced_by_user'
            }

        if analysis.get('has_images', False):
            return {
                'use_ocr': True,
                'reason': 'has_images_always_run_ocr'
            }

        if analysis.get('likely_has_tables', False):
            return {
                'use_ocr': True,
                'reason': 'likely_has_tables_need_ocr_for_structure'
            }

        text_quality = self.check_text_quality(pdf_text, num_pages)
        if not text_quality['needs_ocr']:
            return {
                'use_ocr': False,
                'reason': f'good_text_quality: {text_quality["reason"]}'
            }
        else:
            return {
                'use_ocr': True,
                'reason': f'poor_text_quality: {text_quality["reason"]}'
            }

    def extract_text_with_tesseract(self, file_content: bytes, max_pages: int = 10) -> str:
        """Extract text using Tesseract OCR"""
        if self.ocr_engine != 'tesseract':
            return ""
        
        try:
            import fitz
            import pytesseract
            
            pdf_file = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            num_pages = min(len(doc), max_pages)
            ocr_text = ""
            
            for page_num in range(num_pages):
                print(f"[TESSERACT] Processing page {page_num + 1}/{num_pages}")
                page = doc.load_page(page_num)
                
                zoom_matrix = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
                
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                
                try:
                    page_text = pytesseract.image_to_string(img, config=custom_config)
                    ocr_text += page_text + "\n\n"
                    print(f"[TESSERACT] Page {page_num + 1}: {len(page_text)} chars")
                except Exception as e:
                    print(f"[TESSERACT] Failed on page {page_num + 1}: {e}")
            
            doc.close()
            return ocr_text.strip()
            
        except Exception as e:
            print(f"[TESSERACT] Error: {e}")
            return ""

    def extract_text_with_paddle(self, file_content: bytes, max_pages: int = 10) -> str:
        """Extract text using PaddleOCR (fallback)"""
        if self.ocr_engine != 'paddle':
            return ""
        
        try:
            import fitz
            
            pdf_file = io.BytesIO(file_content)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            num_pages = min(len(doc), max_pages)
            ocr_text = ""
            
            for page_num in range(num_pages):
                print(f"[PADDLE] Processing page {page_num + 1}/{num_pages}")
                page = doc.load_page(page_num)
                
                zoom_matrix = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
                
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                
                try:
                    ocr_result = self.ocr.ocr(img_array)
                    page_text = self._process_paddle_result(ocr_result)
                    ocr_text += page_text + "\n\n"
                    print(f"[PADDLE] Page {page_num + 1}: {len(page_text)} chars")
                except Exception as e:
                    print(f"[PADDLE] Failed on page {page_num + 1}: {e}")
            
            doc.close()
            return ocr_text.strip()
            
        except Exception as e:
            print(f"[PADDLE] Error: {e}")
            return ""

    def _process_paddle_result(self, ocr_result) -> str:
        """Process PaddleOCR result"""
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
                                if confidence > 0.5 and text.strip():
                                    page_text += text + " "
                        page_text += "\n"
                    
        except Exception as e:
            print(f"Error processing PaddleOCR result: {e}")
            
        return page_text

    def extract_text_and_tables(self, file_content: bytes, force_ocr: bool = True) -> Tuple[str, List[pd.DataFrame]]:
        """Main method: Extract both text and tables - IMPROVED"""
        
        try:
            pdf_text, num_pages = self.extract_text_from_pdf(file_content)
            print(f"[PDF] Extracted {len(pdf_text)} characters from {num_pages} pages")
        except Exception as e:
            print(f"[PDF] Text extraction failed: {e}")
            pdf_text, num_pages = "", 0

        analysis = self.pdf_has_images_or_tables(file_content)
        
        ocr_decision = self.should_use_ocr(pdf_text, num_pages, analysis, force_ocr)
        print(f"[DECISION] OCR needed: {ocr_decision['use_ocr']}, Reason: {ocr_decision['reason']}")

        tables = []
        if analysis['likely_has_tables'] or force_ocr or len(pdf_text) < 1000:
            print("[TABLES] Attempting improved table extraction...")
            
            camelot_tables = self.extract_tables_with_camelot_improved(file_content)
            if camelot_tables:
                tables.extend(camelot_tables)
                print(f"[TABLES] Camelot found {len(camelot_tables)} high-quality tables")
            
            if len(tables) < 2:  
                plumber_tables = self.extract_tables_with_pdfplumber_improved(file_content)
                if plumber_tables:
                    new_tables = []
                    for new_table in plumber_tables:
                        is_duplicate = False
                        for existing_table in tables:
                            if (new_table.shape == existing_table.shape and 
                                new_table.iloc[0, 0] == existing_table.iloc[0, 0] if not new_table.empty and not existing_table.empty else False):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            new_tables.append(new_table)
                    
                    tables.extend(new_tables)
                    print(f"[TABLES] pdfplumber found {len(new_tables)} additional tables")
            
            if not tables:
                tabula_tables = self.extract_tables_with_tabula(file_content)
                if tabula_tables:
                    tables.extend(tabula_tables)
                    print(f"[TABLES] Tabula found {len(tabula_tables)} tables as fallback")

        ocr_text = ""
        if ocr_decision['use_ocr']:
            print("[OCR] Starting OCR extraction...")
            
            if self.ocr_engine == 'tesseract':
                ocr_text = self.extract_text_with_tesseract(file_content)
            elif self.ocr_engine == 'paddle':
                ocr_text = self.extract_text_with_paddle(file_content)
            
            print(f"[OCR] Extracted {len(ocr_text)} characters")

        if ocr_text and len(ocr_text) > len(pdf_text) * 1.2:
            final_text = ocr_text
            print("[FINAL] Using OCR text as primary")
        elif pdf_text and ocr_text:
            final_text = pdf_text + "\n\n--- OCR SUPPLEMENT ---\n\n" + ocr_text
            print("[FINAL] Using combined PDF + OCR text")
        elif pdf_text:
            final_text = pdf_text
            print("[FINAL] Using PDF text only")
        elif ocr_text:
            final_text = ocr_text
            print("[FINAL] Using OCR text only")
        else:
            raise HTTPException(
                status_code=400,
                detail="PDF contains no extractable text"
            )

        if tables:
            final_text += "\n\n--- EXTRACTED TABLES ---\n\n"
            for i, table in enumerate(tables):
                final_text += f"\n=== TABLE {i+1} ===\n"
                final_text += f"Dimensions: {table.shape[0]} rows × {table.shape[1]} columns\n"
                final_text += "-" * 60 + "\n"
                
                try:
                    table_str = table.to_string(index=False, max_cols=10, max_rows=50)
                    final_text += table_str
                except Exception as e:
                    final_text += str(table.head(20))
                    print(f"[TABLE] Formatting error for table {i+1}: {e}")
                
                final_text += "\n" + "="*60 + "\n"

        final_text = final_text.strip()
        print(f"[FINAL] Total text: {len(final_text)} characters, Tables: {len(tables)}")
        
        return final_text, tables

    def extract_text(self, file_content: bytes, force_ocr: bool = False) -> Tuple[str, int]:
        """Legacy method for backwards compatibility"""
        text, tables = self.extract_text_and_tables(file_content, force_ocr)
        
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
        except:
            num_pages = 1
        
        return text, num_pages

    def extract_tables_only(self, file_content: bytes) -> List[Dict]:
        """Extract only tables and return them with metadata"""
        _, tables = self.extract_text_and_tables(file_content, force_ocr=True)
        
        structured_tables = []
        for i, table in enumerate(tables):
            structured_tables.append({
                'table_id': i + 1,
                'shape': table.shape,
                'columns': list(table.columns),
                'data': table.to_dict('records'),
                'csv_string': table.to_csv(index=False),
                'preview': table.head(5).to_string(index=False)
            })
        
        return structured_tables