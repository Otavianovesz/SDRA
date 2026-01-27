"""
Deep Link Hunter Module
========================
Finds and downloads financial documents from email link buttons.

Many invoices don't come as attachments - they come as "Click here to view" links.
This module:
1. Parses email HTML to find candidate links
2. Uses Playwright to navigate (handles JavaScript)
3. Downloads or "prints" PDFs from web viewers

Part of Project Cyborg - SRDA Autonomous Treasury Agent
"""

import re
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin

# Lazy imports
BeautifulSoup = None
_playwright_available = False

logger = logging.getLogger('srda.downloads')


def _ensure_bs4():
    """Lazy import BeautifulSoup."""
    global BeautifulSoup
    if BeautifulSoup is None:
        try:
            from bs4 import BeautifulSoup as _BS
            BeautifulSoup = _BS
        except ImportError:
            raise ImportError(
                "BeautifulSoup required. Install with: pip install beautifulsoup4 lxml"
            )


def _check_playwright():
    """Check if Playwright is available."""
    global _playwright_available
    try:
        import playwright
        _playwright_available = True
    except ImportError:
        _playwright_available = False
    return _playwright_available


# Import config
try:
    import config
except ImportError:
    class config:
        from pathlib import Path
        BASE_DIR = Path(".")
        TEMP_DOWNLOADS_DIR = BASE_DIR / "temp_downloads"


# Keywords that suggest a financial document link
FINANCIAL_KEYWORDS_PT = [
    'boleto', 'nota fiscal', 'nfe', 'nfse', 'fatura', 'duplicata',
    'visualizar', 'imprimir', '2ª via', 'segunda via', 'download',
    'baixar', 'ver documento', 'acessar', 'clique aqui', 'abrir',
    'gerar pdf', 'emitir', 'danfe', 'dacte', 'guia', 'pagamento'
]

FINANCIAL_KEYWORDS_EN = [
    'invoice', 'receipt', 'payment', 'bill', 'download', 'view',
    'print', 'open document', 'click here', 'access'
]

FINANCIAL_KEYWORDS = FINANCIAL_KEYWORDS_PT + FINANCIAL_KEYWORDS_EN

# Domains to ignore (social media, marketing, etc.)
IGNORED_DOMAINS = [
    'facebook.com', 'fb.com', 'instagram.com', 'twitter.com', 'x.com',
    'linkedin.com', 'youtube.com', 'whatsapp.com', 'telegram.org',
    'mailto:', 'tel:', 'javascript:', '#',
    'unsubscribe', 'privacy', 'terms', 'help.', 'support.',
    'google.com/analytics', 'doubleclick.net', 'googleadservices.com'
]

# Known financial institution domains (higher confidence)
FINANCIAL_DOMAINS = [
    'itau', 'bradesco', 'santander', 'bb.com.br', 'caixa.gov.br',
    'sicoob', 'sicredi', 'nubank', 'inter.co', 'safra', 'banrisul',
    'nfe.', 'nfse.', 'sefaz', 'prefeitura', 'gov.br',
    'totvs', 'senior', 'sap.', 'linx', 'sydle'
]


@dataclass
class CandidateLink:
    """A potential financial document link."""
    url: str
    text: str
    confidence: float
    source_type: str = "anchor"  # anchor, button, form
    
    def __repr__(self):
        return f"Link({self.confidence:.2f}, {self.text[:30]}...)"


class LinkExtractor:
    """
    Extracts candidate document links from email HTML.
    
    Filtering strategy:
    1. Find all <a> tags and buttons
    2. Filter out known non-document domains
    3. Score remaining links by keyword presence
    4. Return links with confidence > threshold
    """
    
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self._keyword_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in FINANCIAL_KEYWORDS),
            re.IGNORECASE
        )
    
    def extract_links(self, html_content: str, base_url: str = "") -> List[CandidateLink]:
        """
        Extract candidate document links from HTML.
        
        Args:
            html_content: Email HTML body
            base_url: Base URL for resolving relative links
            
        Returns:
            List of CandidateLink objects sorted by confidence
        """
        if not html_content:
            return []
        
        _ensure_bs4()
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except Exception:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except Exception as e:
                logger.error(f"Failed to parse HTML: {e}")
                return []
        
        candidates = []
        
        # Extract from anchor tags
        for link in soup.find_all('a', href=True):
            url = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Resolve relative URLs
            if base_url and not url.startswith(('http://', 'https://')):
                url = urljoin(base_url, url)
            
            # Skip ignored domains
            if self._should_ignore(url):
                continue
            
            # Calculate confidence based on keywords
            confidence = self._calculate_confidence(url, text)
            
            if confidence >= self.min_confidence:
                candidates.append(CandidateLink(
                    url=url,
                    text=text or url[:50],
                    confidence=confidence,
                    source_type="anchor"
                ))
        
        # Also check for buttons with onclick or data-url attributes
        for button in soup.find_all(['button', 'input'], type=['button', 'submit']):
            onclick = button.get('onclick', '')
            data_url = button.get('data-url', '') or button.get('data-href', '')
            text = button.get('value', '') or button.get_text(strip=True)
            
            url = data_url or self._extract_url_from_onclick(onclick)
            if url and not self._should_ignore(url):
                confidence = self._calculate_confidence(url, text)
                if confidence >= self.min_confidence:
                    candidates.append(CandidateLink(
                        url=url,
                        text=text or "Button",
                        confidence=confidence,
                        source_type="button"
                    ))
        
        # Sort by confidence descending and deduplicate
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        seen_urls = set()
        unique_candidates = []
        for c in candidates:
            if c.url not in seen_urls:
                seen_urls.add(c.url)
                unique_candidates.append(c)
        
        logger.info(f"Found {len(unique_candidates)} candidate links")
        return unique_candidates
    
    def _should_ignore(self, url: str) -> bool:
        """Check if URL should be ignored."""
        if not url:
            return True
        url_lower = url.lower()
        return any(domain in url_lower for domain in IGNORED_DOMAINS)
    
    def _calculate_confidence(self, url: str, text: str) -> float:
        """
        Calculate confidence that this link leads to a financial document.
        
        Scoring:
        - Keyword in link text: +0.15 per keyword (max 0.5)
        - Keyword in URL: +0.10 per keyword (max 0.3)
        - PDF in URL: +0.3
        - Known financial domain: +0.25
        """
        confidence = 0.0
        
        # Keyword matches in text (more important)
        text_keywords = len(self._keyword_pattern.findall(text))
        confidence += min(text_keywords * 0.15, 0.5)
        
        # Keyword matches in URL
        url_keywords = len(self._keyword_pattern.findall(url))
        confidence += min(url_keywords * 0.10, 0.3)
        
        # PDF extension in URL
        if '.pdf' in url.lower():
            confidence += 0.3
        
        # Known financial domains
        url_lower = url.lower()
        if any(domain in url_lower for domain in FINANCIAL_DOMAINS):
            confidence += 0.25
        
        return min(confidence, 1.0)
    
    def _extract_url_from_onclick(self, onclick: str) -> Optional[str]:
        """Try to extract URL from onclick JavaScript."""
        if not onclick:
            return None
        
        # Common patterns: window.open('url'), location.href='url', etc.
        patterns = [
            r"window\.open\(['\"]([^'\"]+)['\"]",
            r"location\.href\s*=\s*['\"]([^'\"]+)['\"]",
            r"window\.location\s*=\s*['\"]([^'\"]+)['\"]",
            r"['\"]?(https?://[^'\"]+)['\"]?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, onclick)
            if match:
                return match.group(1)
        
        return None


class PlaywrightDownloader:
    """
    Downloads documents using headless browser.
    
    Handles:
    - JavaScript-rendered pages
    - Redirects
    - PDF direct downloads
    - Page-to-PDF conversion for web viewers
    
    Note: Requires `playwright install chromium` to be run first.
    """
    
    def __init__(self, headless: bool = True, timeout_ms: int = 30000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._browser = None
        self._playwright = None
        self._context = None
    
    async def initialize(self):
        """Initialize Playwright browser (async)."""
        if self._browser:
            return
        
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            logger.info("Playwright browser initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise RuntimeError(
                f"Playwright initialization failed: {e}. "
                "Run 'playwright install chromium' first."
            )
    
    async def close(self):
        """Close browser and release resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Playwright resources released")
    
    async def download_from_link(
        self,
        url: str,
        output_dir: Path = None
    ) -> Optional[Path]:
        """
        Download document from URL.
        
        Strategy:
        1. Navigate to URL
        2. Wait for network idle
        3. If response is PDF → save directly
        4. If HTML page → convert to PDF
        
        Args:
            url: Document URL
            output_dir: Directory for downloaded files
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        output_dir = output_dir or config.TEMP_DOWNLOADS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._browser:
            await self.initialize()
        
        context = None
        try:
            # Create new context for isolation
            context = await self._browser.new_context(
                accept_downloads=True,
                viewport={'width': 1280, 'height': 900}
            )
            page = await context.new_page()
            
            # Set up download handler
            download_path = None
            
            async def handle_download(download):
                nonlocal download_path
                suggested = download.suggested_filename or "download.pdf"
                download_path = output_dir / f"dl_{hashlib.md5(url.encode()).hexdigest()[:8]}_{suggested}"
                await download.save_as(str(download_path))
                logger.info(f"Downloaded via browser: {download_path}")
            
            page.on("download", handle_download)
            
            # Navigate with timeout
            logger.info(f"Navigating to: {url}")
            response = await page.goto(
                url, 
                wait_until='networkidle', 
                timeout=self.timeout_ms
            )
            
            if not response:
                logger.error(f"No response from {url}")
                return None
            
            # Check for download that was triggered
            if download_path and download_path.exists():
                return download_path
            
            content_type = response.headers.get('content-type', '')
            
            # Direct PDF download (from response)
            if 'application/pdf' in content_type:
                pdf_data = await response.body()
                if pdf_data and len(pdf_data) > 100:
                    output_path = output_dir / f"direct_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
                    output_path.write_bytes(pdf_data)
                    logger.info(f"Downloaded PDF directly: {output_path}")
                    return output_path
            
            # HTML page - try to find embedded PDF or convert page to PDF
            if 'text/html' in content_type:
                # Wait a bit more for any lazy-loaded content
                await page.wait_for_timeout(2000)
                
                # Check for embedded PDF viewer (iframe, embed, object)
                pdf_url = await self._find_embedded_pdf(page)
                if pdf_url:
                    logger.info(f"Found embedded PDF: {pdf_url}")
                    # Recursively download the embedded PDF
                    return await self.download_from_link(pdf_url, output_dir)
                
                # Convert visible page to PDF
                output_path = output_dir / f"printed_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
                await page.pdf(
                    path=str(output_path), 
                    format='A4',
                    print_background=True
                )
                
                # Verify PDF is not empty
                if output_path.exists() and output_path.stat().st_size > 1000:
                    logger.info(f"Printed page to PDF: {output_path}")
                    return output_path
                else:
                    if output_path.exists():
                        output_path.unlink()
                    logger.warning(f"Empty/invalid PDF generated from {url}")
                    return None
            
            logger.warning(f"Unexpected content type: {content_type}")
            return None
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return None
        finally:
            if context:
                await context.close()
    
    async def _find_embedded_pdf(self, page) -> Optional[str]:
        """Look for embedded PDF in page (iframe, embed, object)."""
        try:
            # Check iframes
            for selector in ['iframe[src*=".pdf"]', 'embed[src*=".pdf"]', 'object[data*=".pdf"]']:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    src = await elem.get_attribute('src') or await elem.get_attribute('data')
                    if src:
                        return src
            
            # Check for data URIs or blob URLs
            # (these are harder to extract, skip for now)
            
        except Exception as e:
            logger.debug(f"Error finding embedded PDF: {e}")
        
        return None


# =============================================================================
# Synchronous wrappers for non-async code
# =============================================================================

def download_link_sync(
    url: str, 
    output_dir: Path = None,
    timeout_ms: int = 30000
) -> Optional[Path]:
    """
    Synchronous wrapper for Playwright download.
    
    Args:
        url: URL to download from
        output_dir: Output directory (default: config.TEMP_DOWNLOADS_DIR)
        timeout_ms: Timeout in milliseconds
        
    Returns:
        Path to downloaded file or None
    """
    if not _check_playwright():
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None
    
    async def _download():
        downloader = PlaywrightDownloader(timeout_ms=timeout_ms)
        try:
            await downloader.initialize()
            return await downloader.download_from_link(url, output_dir)
        finally:
            await downloader.close()
    
    try:
        # Try to get existing event loop, create new one if needed
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            # In this case, caller should use async version directly
            logger.warning("Already in async context, use async version instead")
            return None
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_download())
            finally:
                loop.close()
                
    except Exception as e:
        logger.error(f"Sync download failed: {e}")
        return None


def extract_links_from_html(html: str, min_confidence: float = 0.3) -> List[CandidateLink]:
    """
    Convenience function to extract links from HTML.
    
    Args:
        html: HTML content
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of candidate links
    """
    extractor = LinkExtractor(min_confidence=min_confidence)
    return extractor.extract_links(html)


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test HTML parsing
    test_html = """
    <html>
    <body>
        <a href="https://example.com/boleto.pdf">Visualizar Boleto</a>
        <a href="https://facebook.com/share">Compartilhar</a>
        <a href="https://banco.itau.com.br/ver-fatura">Ver Fatura</a>
        <a href="https://example.com/newsletter">Newsletter</a>
        <button onclick="window.open('https://nfe.example.com/danfe')">Imprimir NF-e</button>
    </body>
    </html>
    """
    
    print("Link Extractor Test")
    print("=" * 50)
    
    links = extract_links_from_html(test_html)
    for link in links:
        print(f"  {link.confidence:.2f} | {link.text} | {link.url[:50]}...")
    
    # Test Playwright download (if URL provided)
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        print(f"\nDownloading: {test_url}")
        result = download_link_sync(test_url)
        if result:
            print(f"Downloaded to: {result}")
        else:
            print("Download failed")
