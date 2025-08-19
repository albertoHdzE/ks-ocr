#!/usr/bin/env python3
"""
Advanced OCR Document Extraction PoC
Supports PDF and image processing with intelligent content extraction
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Core libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd

# OCR libraries
import pytesseract
import easyocr

# PDF processing
import fitz  # PyMuPDF
from pdf2image import convert_from_path

# Text processing and NLP
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Structure for extraction results"""
    file_path: str
    file_type: str
    raw_text: str
    structured_data: Dict
    confidence_score: float
    processing_time: float
    metadata: Dict

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Apply image enhancement techniques"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using EAST detector or contours"""
        # Apply morphological operations to detect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # Filter small regions
                text_regions.append((x, y, w, h))
        
        return text_regions

class OCREngine:
    """Multi-engine OCR processor"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.easyocr_reader = easyocr.Reader(['en'])
    
    def extract_with_tesseract(self, image: np.ndarray, config: str = '--oem 3 --psm 6') -> Dict:
        """Extract text using Tesseract OCR"""
        try:
            # Get detailed data including confidence scores
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Filter out low-confidence results
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 30]
            texts = [data['text'][i] for i, conf in enumerate(data['conf']) if int(conf) > 30 and data['text'][i].strip()]
            
            text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'engine': 'tesseract',
                'word_data': data
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'tesseract', 'error': str(e)}
    
    def extract_with_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image)
            
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    texts.append(text)
                    confidences.append(confidence)
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence * 100,  # Convert to percentage
                'engine': 'easyocr',
                'raw_results': results
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'easyocr', 'error': str(e)}
    
    def process_image(self, image_path: str) -> Dict:
        """Process image with multiple OCR engines"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        enhanced_image = self.preprocessor.enhance_image(image)
        
        # Run both OCR engines
        tesseract_result = self.extract_with_tesseract(enhanced_image)
        easyocr_result = self.extract_with_easyocr(enhanced_image)
        
        # Choose best result based on confidence and text length
        if tesseract_result['confidence'] > easyocr_result['confidence']:
            best_result = tesseract_result
            alternative = easyocr_result
        else:
            best_result = easyocr_result
            alternative = tesseract_result
        
        return {
            'primary': best_result,
            'alternative': alternative,
            'text_regions': self.preprocessor.detect_text_regions(enhanced_image)
        }

class DocumentProcessor:
    """PDF document processor with text and image extraction"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text and images from PDF"""
        doc = fitz.open(pdf_path)
        results = {
            'text_content': [],
            'image_content': [],
            'metadata': {},
            'page_count': len(doc)
        }
        
        # Extract metadata
        results['metadata'] = doc.metadata
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text directly
            text = page.get_text()
            if text.strip():
                results['text_content'].append({
                    'page': page_num + 1,
                    'text': text,
                    'method': 'direct_extraction'
                })
            
            # Extract images and apply OCR
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Save temporary image for OCR
                        temp_path = f"temp_img_{page_num}_{img_index}.png"
                        with open(temp_path, "wb") as f:
                            f.write(img_data)
                        
                        # Apply OCR
                        ocr_result = self.ocr_engine.process_image(temp_path)
                        
                        results['image_content'].append({
                            'page': page_num + 1,
                            'image_index': img_index,
                            'ocr_result': ocr_result,
                            'dimensions': (pix.width, pix.height)
                        })
                        
                        # Clean up
                        os.remove(temp_path)
                    
                    pix = None
                except Exception as e:
                    logger.error(f"Error processing image on page {page_num}: {e}")
        
        doc.close()
        return results

class ContentAnalyzer:
    """Intelligent content analysis and structured data extraction"""
    
    def __init__(self):
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize transformer pipeline for classification
        try:
            self.classifier = pipeline("zero-shot-classification", 
                                     model="facebook/bart-large-mnli")
        except:
            logger.warning("Could not load classification model")
            self.classifier = None
    
    def classify_document_type(self, text: str) -> Dict:
        """Classify document type using content analysis"""
        if not self.classifier:
            return {'type': 'unknown', 'confidence': 0}
        
        candidate_labels = [
            'financial report', 'scientific paper', 'legal document', 
            'technical manual', 'invoice', 'receipt', 'medical report',
            'academic paper', 'business document', 'research paper'
        ]
        
        try:
            result = self.classifier(text[:1000], candidate_labels)  # Use first 1000 chars
            return {
                'type': result['labels'][0],
                'confidence': result['scores'][0],
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {'type': 'unknown', 'confidence': 0}
    
    def extract_financial_data(self, text: str) -> Dict:
        """Extract financial information from text"""
        financial_data = {
            'monetary_amounts': [],
            'percentages': [],
            'dates': [],
            'companies': [],
            'financial_terms': []
        }
        
        # Extract monetary amounts
        money_pattern = r'[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|dollars?|euros?|pounds?)'
        financial_data['monetary_amounts'] = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Extract percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        financial_data['percentages'] = re.findall(percentage_pattern, text)
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        ]
        for pattern in date_patterns:
            financial_data['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract financial terms
        financial_terms = [
            'revenue', 'profit', 'loss', 'assets', 'liabilities', 'equity',
            'cash flow', 'EBITDA', 'ROI', 'dividend', 'shares', 'market cap'
        ]
        for term in financial_terms:
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE):
                financial_data['financial_terms'].append(term)
        
        # Use NLP for entity extraction if available
        if self.nlp:
            doc = self.nlp(text[:5000])  # Limit text length for performance
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    financial_data['companies'].append(ent.text)
        
        return financial_data
    
    def extract_scientific_data(self, text: str) -> Dict:
        """Extract scientific information from text"""
        scientific_data = {
            'authors': [],
            'affiliations': [],
            'abstract': '',
            'keywords': [],
            'citations': [],
            'figures_tables': [],
            'methodology': [],
            'results': []
        }
        
        # Extract potential authors (names in title case)
        author_pattern = r'[A-Z][a-z]+\s+[A-Z][a-z]+'
        potential_authors = re.findall(author_pattern, text)
        scientific_data['authors'] = list(set(potential_authors[:10]))  # Limit to avoid noise
        
        # Extract abstract
        abstract_match = re.search(r'abstract\s*:?\s*(.*?)(?:\n\s*\n|\n\s*(?:introduction|keywords))', 
                                 text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            scientific_data['abstract'] = abstract_match.group(1).strip()
        
        # Extract keywords
        keywords_match = re.search(r'keywords?\s*:?\s*(.*?)(?:\n\s*\n|\n\s*[A-Z])', 
                                 text, re.IGNORECASE)
        if keywords_match:
            scientific_data['keywords'] = [kw.strip() for kw in keywords_match.group(1).split(',')]
        
        # Extract figure and table references
        fig_table_pattern = r'(?:Figure|Table|Fig\.)\s+\d+(?:\.\d+)?'
        scientific_data['figures_tables'] = re.findall(fig_table_pattern, text, re.IGNORECASE)
        
        # Extract citations (basic pattern)
        citation_pattern = r'\[[\d,\s-]+\]|\([A-Za-z]+,?\s+\d{4}\)'
        scientific_data['citations'] = re.findall(citation_pattern, text)
        
        return scientific_data
    
    def analyze_content(self, text: str) -> Dict:
        """Perform comprehensive content analysis"""
        doc_classification = self.classify_document_type(text)
        
        analysis = {
            'document_type': doc_classification,
            'text_statistics': {
                'word_count': len(text.split()),
                'sentence_count': len(sent_tokenize(text)),
                'character_count': len(text)
            },
            'structured_data': {}
        }
        
        # Extract structured data based on document type
        doc_type = doc_classification.get('type', '').lower()
        
        if 'financial' in doc_type:
            analysis['structured_data'] = self.extract_financial_data(text)
        elif 'scientific' in doc_type or 'research' in doc_type or 'academic' in doc_type:
            analysis['structured_data'] = self.extract_scientific_data(text)
        else:
            # Generic extraction
            analysis['structured_data'] = {
                'entities': [],
                'key_phrases': [],
                'summary': text[:500] + '...' if len(text) > 500 else text
            }
            
            # Extract entities if NLP is available
            if self.nlp:
                doc = self.nlp(text[:5000])
                analysis['structured_data']['entities'] = [
                    {'text': ent.text, 'label': ent.label_} for ent in doc.ents
                ]
        
        return analysis

class AdvancedOCRApp:
    """Main application class for advanced OCR processing"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.doc_processor = DocumentProcessor()
        self.content_analyzer = ContentAnalyzer()
    
    def process_file(self, file_path: str) -> ExtractionResult:
        """Process a file (PDF or image) and extract information"""
        start_time = datetime.now()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = file_path.suffix.lower()
        
        try:
            if file_type == '.pdf':
                # Process PDF
                pdf_result = self.doc_processor.extract_from_pdf(str(file_path))
                
                # Combine all text content
                all_text = []
                for text_content in pdf_result['text_content']:
                    all_text.append(text_content['text'])
                
                for img_content in pdf_result['image_content']:
                    if 'primary' in img_content['ocr_result']:
                        all_text.append(img_content['ocr_result']['primary']['text'])
                
                raw_text = '\n'.join(all_text)
                confidence_score = 85  # Default for PDF text extraction
                
            elif file_type in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                # Process image
                ocr_result = self.ocr_engine.process_image(str(file_path))
                raw_text = ocr_result['primary']['text']
                confidence_score = ocr_result['primary']['confidence']
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Analyze content
            content_analysis = self.content_analyzer.analyze_content(raw_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ExtractionResult(
                file_path=str(file_path),
                file_type=file_type,
                raw_text=raw_text,
                structured_data=content_analysis,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    'file_size': file_path.stat().st_size,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def batch_process(self, file_paths: List[str]) -> List[ExtractionResult]:
        """Process multiple files"""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        return results
    
    def export_results(self, results: List[ExtractionResult], output_format: str = 'json') -> str:
        """Export results to various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == 'json':
            output_file = f"ocr_results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(result) for result in results], f, indent=2, ensure_ascii=False)
        
        elif output_format.lower() == 'csv':
            output_file = f"ocr_results_{timestamp}.csv"
            data = []
            for result in results:
                data.append({
                    'file_path': result.file_path,
                    'file_type': result.file_type,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.processing_time,
                    'word_count': result.structured_data.get('text_statistics', {}).get('word_count', 0),
                    'document_type': result.structured_data.get('document_type', {}).get('type', 'unknown'),
                    'text_preview': result.raw_text[:200] + '...' if len(result.raw_text) > 200 else result.raw_text
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_file

def main():
    """CLI interface for the OCR application"""
    parser = argparse.ArgumentParser(description="Advanced OCR Document Extraction PoC")
    parser.add_argument('files', nargs='+', help='Input files (PDF or images)')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json',
                       help='Output format for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize the OCR application
    app = AdvancedOCRApp()
    
    try:
        # Process files
        logger.info(f"Processing {len(args.files)} files...")
        results = app.batch_process(args.files)
        
        if results:
            # Export results
            output_file = app.export_results(results, args.output_format)
            logger.info(f"Results exported to: {output_file}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Files processed: {len(results)}")
            print(f"Average confidence: {sum(r.confidence_score for r in results) / len(results):.2f}%")
            print(f"Total processing time: {sum(r.processing_time for r in results):.2f}s")
            print(f"Results saved to: {output_file}")
            
            # Show document types found
            doc_types = {}
            for result in results:
                doc_type = result.structured_data.get('document_type', {}).get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print(f"\nDocument types detected:")
            for doc_type, count in doc_types.items():
                print(f"  - {doc_type}: {count}")
            
        else:
            print("No files were successfully processed.")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

"""
Installation Requirements:
pip install opencv-python pillow pytesseract easyocr PyMuPDF pdf2image spacy transformers nltk pandas

>>> import nltk
>>> nltk.download('punkt_tab')

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Tesseract OCR system dependency:
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Usage Examples:
python ocr_app.py document.pdf image.jpg --output-format json --verbose
python ocr_app.py *.pdf --output-format csv
"""