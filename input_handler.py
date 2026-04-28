"""
Input Handler Module

Handles:
- Plain text inputs
- URL inputs (including Twitter/X)
- Image inputs (with OCR)

Key Principles:
- No auto-storage of scraped content
- Clear error handling for failed extractions
- Only extracts main textual content
"""
import re
import os
import base64
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

class InputType(Enum):
    TEXT = "text"
    URL = "url"
    IMAGE = "image"

@dataclass
class ProcessedInput:
    """Container for processed input"""
    original: str
    input_type: InputType
    text: str
    domain: Optional[str] = None
    title: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

class InputHandler:
    """
    Handles text, URL, and image inputs for the classifier.
    """
    
    URL_PATTERN = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    REMOVE_ELEMENTS = [
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'iframe', 'noscript', 'form', 'button', 'input',
        'advertisement', 'ad', 'sidebar', 'menu', 'popup',
        'cookie', 'banner', 'promo', 'newsletter', 'subscribe',
        'social', 'share', 'comment', 'related', 'recommended'
    ]
    
    REMOVE_PATTERNS = [
        r'nav', r'menu', r'footer', r'header', r'sidebar',
        r'comment', r'social', r'share', r'ad[-_]?', r'promo',
        r'related', r'recommend', r'popular', r'trending',
        r'newsletter', r'subscribe', r'cookie', r'banner'
    ]
    
    # Multiple user agents to rotate
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]
    
    # Supported image extensions
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._ocr_processor = None
    
    def _get_ocr_processor(self):
        """Lazy load OCR processor"""
        if self._ocr_processor is None:
            try:
                from preprocessing import get_image_preprocessor
                self._ocr_processor = get_image_preprocessor()
            except Exception as e:
                print(f"OCR not available: {e}")
        return self._ocr_processor
    
    def is_url(self, text: str) -> bool:
        """Check if input appears to be a URL"""
        text = text.strip()
        return bool(self.URL_PATTERN.match(text))
    
    def is_image_path(self, text: str) -> bool:
        """Check if input is an image file path"""
        text = text.strip().lower()
        return any(text.endswith(ext) for ext in self.IMAGE_EXTENSIONS)
    
    def is_twitter_url(self, url: str) -> bool:
        """Check if URL is from Twitter/X"""
        domain = self.extract_domain(url).lower()
        return any(t in domain for t in ['twitter.com', 'x.com', 't.co'])
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"
    
    def _get_session(self) -> requests.Session:
        """Create a session with rotating user agent"""
        import random
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        return session
    
    def fetch_twitter_content(self, url: str) -> Dict:
        """
        Fetch Twitter/X content using alternative methods.
        Twitter blocks direct requests, so we try multiple approaches.
        """
        # Extract tweet ID and username from URL
        tweet_id = None
        username = None
        
        # Extract tweet ID
        id_patterns = [r'/status/(\d+)', r'/statuses/(\d+)']
        for pattern in id_patterns:
            match = re.search(pattern, url)
            if match:
                tweet_id = match.group(1)
                break
        
        # Extract username
        user_match = re.search(r'(?:twitter\.com|x\.com)/([^/]+)', url)
        if user_match:
            username = user_match.group(1)
        
        if not tweet_id:
            return {
                'success': False,
                'error': "Could not extract tweet ID from URL."
            }
        
        # Method 1: Try Twitter's oEmbed API first (most reliable)
        try:
            oembed_url = f"https://publish.twitter.com/oembed?url={url}&omit_script=true"
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                html = data.get('html', '')
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    # Get all text and clean it
                    text = soup.get_text(separator=' ', strip=True)
                    # Remove the attribution part (usually after —)
                    if '—' in text:
                        text = text.split('—')[0].strip()
                    elif '-' in text and '@' in text:
                        # Try to extract just the tweet content
                        parts = text.split('-')
                        if len(parts) > 1:
                            text = parts[0].strip()
                    
                    # Clean URLs
                    text = re.sub(r'pic\.twitter\.com/\S+', '', text)
                    text = re.sub(r'https?://\S+', '', text)
                    text = text.strip()
                    
                    if len(text) > 10:
                        author = data.get('author_name', username or 'Unknown')
                        return {
                            'success': True,
                            'text': text,
                            'title': f"Tweet by {author}",
                            'method': 'Twitter oEmbed'
                        }
        except Exception as e:
            print(f"oEmbed failed: {e}")
        
        # Method 2: Try Nitter instances
        nitter_instances = [
            'nitter.privacydev.net',
            'nitter.poast.org',
            'nitter.woodland.cafe',
        ]
        
        if username and tweet_id:
            for instance in nitter_instances:
                try:
                    nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
                    session = self._get_session()
                    response = session.get(nitter_url, timeout=8)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find tweet content
                        tweet_content = soup.find('div', class_='tweet-content')
                        if not tweet_content:
                            tweet_content = soup.find('div', class_='tweet-body')
                        
                        if tweet_content:
                            text = tweet_content.get_text(strip=True)
                            if len(text) > 10:
                                return {
                                    'success': True,
                                    'text': text,
                                    'title': f"Tweet by @{username}",
                                    'method': f'Nitter ({instance})'
                                }
                except Exception as e:
                    print(f"Nitter {instance} failed: {e}")
                    continue
        
        # Method 3: Try FxTwitter API
        if username and tweet_id:
            try:
                fx_url = f"https://api.fxtwitter.com/{username}/status/{tweet_id}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(fx_url, timeout=10, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'tweet' in data and 'text' in data['tweet']:
                        text = data['tweet']['text']
                        author = data['tweet'].get('author', {}).get('name', username)
                        return {
                            'success': True,
                            'text': text,
                            'title': f"Tweet by {author}",
                            'method': 'FxTwitter API'
                        }
            except Exception as e:
                print(f"FxTwitter failed: {e}")
        
        # All methods failed
        return {
            'success': False,
            'error': (
                "Could not fetch tweet content. Twitter/X blocks automated access.\n\n"
                "Please try one of these alternatives:\n"
                "• Copy the tweet text and paste it directly\n"
                "• Take a screenshot and upload it in the Image tab"
            )
        }
    
    def fetch_url(self, url: str) -> Dict:
        """Fetch content from URL."""
        # Special handling for Twitter/X
        if self.is_twitter_url(url):
            return self.fetch_twitter_content(url)
        
        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return {
                    'success': False,
                    'error': f"Unsupported content type: {content_type}. Only HTML pages are supported."
                }
            
            return {
                'success': True,
                'html': response.text
            }
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': "Request timed out. Please try again."}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': "Could not connect to the URL. Please check the address."}
        except requests.exceptions.HTTPError as e:
            return {'success': False, 'error': f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': f"Failed to fetch URL: {str(e)}"}
    
    def extract_main_content(self, html: str) -> Dict:
        """Extract main textual content from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            for element in self.REMOVE_ELEMENTS:
                for tag in soup.find_all(element):
                    tag.decompose()
            
            for pattern in self.REMOVE_PATTERNS:
                regex = re.compile(pattern, re.IGNORECASE)
                for tag in soup.find_all(class_=regex):
                    tag.decompose()
                for tag in soup.find_all(id=regex):
                    tag.decompose()
            
            main_content = None
            content_selectors = [
                ('article', {}),
                ('main', {}),
                ('div', {'class': re.compile(r'(article|content|post|entry|story)', re.I)}),
                ('div', {'id': re.compile(r'(article|content|post|entry|story)', re.I)}),
                ('div', {'role': 'main'}),
            ]
            
            for tag, attrs in content_selectors:
                found = soup.find(tag, attrs)
                if found and len(found.get_text(strip=True)) > 200:
                    main_content = found
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            text = main_content.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n', text)
            text = text.strip()
            
            if len(text) < 50:
                return {
                    'success': False,
                    'error': "Could not extract meaningful content from this page."
                }
            
            if len(text) > 2000:
                truncated = text[:2000]
                last_period = truncated.rfind('.')
                if last_period > 1500:
                    truncated = truncated[:last_period + 1]
                text = truncated
            
            return {
                'success': True,
                'text': text,
                'title': title
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to parse page content: {str(e)}"
            }
    
    def process_image(self, image_path: str) -> Dict:
        """Process image file and extract text using OCR."""
        if not os.path.exists(image_path):
            return {
                'success': False,
                'input_type': InputType.IMAGE,
                'error': f"Image file not found: {image_path}"
            }
        
        ocr = self._get_ocr_processor()
        if not ocr:
            return {
                'success': False,
                'input_type': InputType.IMAGE,
                'error': "OCR not available. Install easyocr: pip install easyocr"
            }
        
        try:
            text = ocr.extract_text(image_path)
            if not text or len(text.strip()) < 10:
                return {
                    'success': False,
                    'input_type': InputType.IMAGE,
                    'error': "Could not extract meaningful text from image."
                }
            
            return {
                'success': True,
                'input_type': InputType.IMAGE,
                'text': text.strip(),
                'source': image_path
            }
        except Exception as e:
            return {
                'success': False,
                'input_type': InputType.IMAGE,
                'error': f"OCR failed: {str(e)}"
            }
    
    def process_image_bytes(self, image_bytes: bytes, filename: str = "uploaded_image") -> Dict:
        """Process image from bytes (for uploaded files)."""
        ocr = self._get_ocr_processor()
        if not ocr:
            return {
                'success': False,
                'input_type': InputType.IMAGE,
                'error': "OCR not available. Install easyocr: pip install easyocr"
            }
        
        try:
            text = ocr.extract_text_from_bytes(image_bytes)
            if not text or len(text.strip()) < 5:
                return {
                    'success': False,
                    'input_type': InputType.IMAGE,
                    'error': "Could not extract meaningful text from image. Try a clearer image or manually type the text."
                }
            
            return {
                'success': True,
                'input_type': InputType.IMAGE,
                'text': text.strip(),
                'source': filename
            }
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message
            if 'devanagari' in error_msg.lower():
                return {
                    'success': False,
                    'input_type': InputType.IMAGE,
                    'error': "Language detection issue. Please try again or manually type the text from the image."
                }
            return {
                'success': False,
                'input_type': InputType.IMAGE,
                'error': f"OCR failed: {error_msg}. Try a clearer image or manually type the text."
            }
    
    def process_input(self, user_input: str) -> Dict:
        """
        Main entry point for processing user input.
        Handles text, URLs, and image file paths.
        """
        user_input = user_input.strip()
        
        if not user_input:
            return {
                'success': False,
                'input_type': InputType.TEXT,
                'error': "Please enter a claim, URL, or image path to verify."
            }
        
        # Check if it's an image path
        if self.is_image_path(user_input):
            return self.process_image(user_input)
        
        # Check if URL
        if self.is_url(user_input):
            domain = self.extract_domain(user_input)
            
            # Special handling for Twitter
            if self.is_twitter_url(user_input):
                result = self.fetch_twitter_content(user_input)
                if result['success']:
                    return {
                        'success': True,
                        'input_type': InputType.URL,
                        'text': result['text'],
                        'domain': domain,
                        'title': result.get('title', ''),
                        'fetch_method': result.get('method', 'unknown')
                    }
                else:
                    return {
                        'success': False,
                        'input_type': InputType.URL,
                        'domain': domain,
                        'error': result['error']
                    }
            
            # Regular URL
            fetch_result = self.fetch_url(user_input)
            if not fetch_result['success']:
                return {
                    'success': False,
                    'input_type': InputType.URL,
                    'domain': domain,
                    'error': fetch_result['error']
                }
            
            extract_result = self.extract_main_content(fetch_result['html'])
            if not extract_result['success']:
                return {
                    'success': False,
                    'input_type': InputType.URL,
                    'domain': domain,
                    'error': extract_result['error']
                }
            
            return {
                'success': True,
                'input_type': InputType.URL,
                'text': extract_result['text'],
                'domain': domain,
                'title': extract_result.get('title', '')
            }
        
        # Plain text input
        return {
            'success': True,
            'input_type': InputType.TEXT,
            'text': user_input
        }


def process_user_input(user_input: str) -> Dict:
    """Process user input (text, URL, or image path)"""
    handler = InputHandler()
    return handler.process_input(user_input)