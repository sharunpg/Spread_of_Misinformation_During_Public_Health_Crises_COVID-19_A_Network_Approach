"""
Preprocessing Module

Handles:
1. Language detection
2. Translation to English
3. Text normalization
4. OCR for images
"""
import re
from typing import Tuple, Optional
from dataclasses import dataclass

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

@dataclass
class PreprocessedText:
    """Container for preprocessing results"""
    original: str
    detected_language: str
    translated: str
    cleaned: str
    is_translated: bool

class TextPreprocessor:
    """
    Handles all text preprocessing steps.
    Stateless design - can process any text independently.
    """
    
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')
        
        self.preserve_terms = {
            'covid', 'covid-19', 'sars-cov-2', 'coronavirus',
            'vaccine', 'mrna', 'pfizer', 'moderna', 'astrazeneca',
            'who', 'cdc', 'nih', 'nhs', 'fda'
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            if len(text.strip()) < 10:
                return 'en'
            return detect(text)
        except LangDetectException:
            return 'en'
    
    def translate_to_english(self, text: str, source_lang: str) -> Tuple[str, bool]:
        """Translate non-English text to English."""
        if source_lang == 'en':
            return text, False
        
        try:
            translated = self.translator.translate(text)
            return translated, True
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
            return text, False
    
    def clean_text(self, text: str) -> str:
        """Normalize text for semantic comparison."""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess(self, text: str) -> PreprocessedText:
        """Complete preprocessing pipeline."""
        original = text.strip()
        detected_lang = self.detect_language(original)
        translated, was_translated = self.translate_to_english(original, detected_lang)
        cleaned = self.clean_text(translated)
        
        return PreprocessedText(
            original=original,
            detected_language=detected_lang,
            translated=translated,
            cleaned=cleaned,
            is_translated=was_translated
        )


class ImagePreprocessor:
    """
    OCR module for processing images.
    Uses EasyOCR for better accuracy on social media images.
    """
    
    def __init__(self):
        self._readers = {}
    
    def _get_reader(self, lang_list=None):
        """Lazy loading of EasyOCR with specific language support"""
        if lang_list is None:
            lang_list = ['en']
        
        # Create a key for caching readers
        lang_key = tuple(sorted(lang_list))
        
        if lang_key not in self._readers:
            try:
                import easyocr
                self._readers[lang_key] = easyocr.Reader(lang_list, gpu=False)
                print(f"✓ EasyOCR initialized with languages: {lang_list}")
            except ImportError:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr"
                )
        return self._readers[lang_key]
    
    def _try_ocr_with_languages(self, image_input, is_path=True):
        """
        Try OCR with different language combinations.
        Falls back to different configs if one fails.
        """
        import numpy as np
        from PIL import Image
        import io
        
        # Prepare image
        if is_path:
            image_data = image_input
        else:
            image = Image.open(io.BytesIO(image_input))
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_data = np.array(image)
        
        # Language combinations to try (in order of preference)
        lang_configs = [
            ['en'],                          # English only (most compatible)
            ['en', 'hi', 'mr', 'ne'],        # English + Devanagari scripts
            ['en', 'es', 'fr', 'de'],        # English + European languages
        ]
        
        last_error = None
        
        for lang_list in lang_configs:
            try:
                reader = self._get_reader(lang_list)
                results = reader.readtext(image_data)
                
                if results:
                    texts = [result[1] for result in results]
                    extracted = ' '.join(texts)
                    if extracted.strip():
                        return extracted
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # If it's a Devanagari compatibility error, try the Hindi-compatible config
                if 'devanagari' in error_str:
                    try:
                        reader = self._get_reader(['en', 'hi', 'mr', 'ne'])
                        results = reader.readtext(image_data)
                        if results:
                            texts = [result[1] for result in results]
                            return ' '.join(texts)
                    except Exception as e2:
                        last_error = e2
                        continue
                continue
        
        # If all attempts failed
        if last_error:
            print(f"⚠️ OCR failed after trying multiple language configs: {last_error}")
        return ""
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            return self._try_ocr_with_languages(image_path, is_path=True)
        except Exception as e:
            print(f"⚠️ OCR failed: {e}")
            return ""
    
    def extract_text_from_bytes(self, image_bytes: bytes) -> str:
        """Extract text from image bytes (for uploaded files)"""
        try:
            return self._try_ocr_with_languages(image_bytes, is_path=False)
        except Exception as e:
            print(f"⚠️ OCR failed: {e}")
            return ""


# Singleton instances
_text_preprocessor = None
_image_preprocessor = None

def get_text_preprocessor() -> TextPreprocessor:
    """Get or create text preprocessor"""
    global _text_preprocessor
    if _text_preprocessor is None:
        _text_preprocessor = TextPreprocessor()
    return _text_preprocessor

def get_image_preprocessor() -> ImagePreprocessor:
    """Get or create image preprocessor (lazy loaded)"""
    global _image_preprocessor
    if _image_preprocessor is None:
        _image_preprocessor = ImagePreprocessor()
    return _image_preprocessor

def preprocess_text(text: str) -> PreprocessedText:
    """Convenience function for quick preprocessing"""
    return get_text_preprocessor().preprocess(text)