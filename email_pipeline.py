"""
Email Pipeline Orchestrator
============================
Coordinates the full email-to-document processing workflow.

This is the main entry point for Project Cyborg - the autonomous treasury agent.

Workflow:
1. Fetch pending emails from Gmail
2. Extract attachments (XML, PDF)
3. Hunt for links if no attachments
4. Process documents through extraction pipeline
5. Validate and rename files
6. Mark emails as processed

Part of Project Cyborg - SRDA Autonomous Treasury Agent
"""

import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

import config
from logging_config import get_logger

logger = get_logger('email_pipeline')


class ProcessingStatus(Enum):
    """Status of email processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"


@dataclass
class EmailProcessingResult:
    """Result of processing a single email."""
    email_id: str
    subject: str
    status: ProcessingStatus
    documents_extracted: int = 0
    files_created: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    gemini_tokens_used: int = 0


@dataclass
class PipelineStats:
    """Statistics for pipeline run."""
    emails_processed: int = 0
    emails_succeeded: int = 0
    emails_failed: int = 0
    emails_skipped: int = 0
    documents_extracted: int = 0
    gemini_tokens_used: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class EmailPipeline:
    """
    Main pipeline for email-to-document processing.
    
    Implements the full Project Cyborg workflow:
    1. Gmail API integration
    2. Attachment extraction
    3. Link hunting with Playwright
    4. Document classification and extraction
    5. Validation and deduplication
    6. Renaming and organization
    
    Usage:
        pipeline = EmailPipeline()
        results = pipeline.process_pending_emails()
    """
    
    def __init__(
        self,
        db=None,
        use_cloud_ai: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the email pipeline.
        
        Args:
            db: Database instance (will create if None)
            use_cloud_ai: Enable Gemini AI fallback
            progress_callback: Callback(message, current, total) for UI updates
        """
        self.db = db
        self.use_cloud_ai = use_cloud_ai
        self.progress_callback = progress_callback
        self.stats = PipelineStats()
        self._running = False
        self._stop_requested = False
        
        # Lazy-loaded components
        self._gmail_connector = None
        self._link_extractor = None
        self._scanner = None
        
    def _log_progress(self, message: str, current: int = 0, total: int = 0, level: str = "info"):
        """Log progress and optionally call callback."""
        getattr(logger, level)(message)
        if self.progress_callback:
            try:
                self.progress_callback(message, current, total)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    @property
    def gmail_connector(self):
        """Lazy load Gmail connector."""
        if self._gmail_connector is None:
            from integrations.gmail_connector import get_gmail_connector
            self._gmail_connector = get_gmail_connector()
        return self._gmail_connector
    
    @property
    def link_extractor(self):
        """Lazy load link extractor."""
        if self._link_extractor is None:
            from integrations.link_extractor import LinkExtractor
            self._link_extractor = LinkExtractor()
        return self._link_extractor
    
    @property
    def scanner(self):
        """Lazy load document scanner."""
        if self._scanner is None:
            from scanner import CognitiveScanner
            self._scanner = CognitiveScanner(db=self.db)
        return self._scanner
    
    def stop(self):
        """Request graceful stop of pipeline."""
        self._stop_requested = True
        logger.info("Pipeline stop requested")
    
    def process_pending_emails(
        self,
        days_lookback: int = None,
        max_emails: int = 50
    ) -> List[EmailProcessingResult]:
        """
        Process all pending emails.
        
        Args:
            days_lookback: Days to look back (default from config)
            max_emails: Maximum emails to process in one run
            
        Returns:
            List of processing results
        """
        self._running = True
        self._stop_requested = False
        self.stats = PipelineStats(start_time=datetime.now())
        results = []
        
        try:
            # Authenticate Gmail
            self._log_progress("Autenticando com Gmail...")
            if not self.gmail_connector.authenticate():
                raise RuntimeError("Gmail authentication failed")
            
            # Ensure labels exist
            self._log_progress("Verificando labels...")
            self.gmail_connector.ensure_labels_exist()
            
            # Fetch pending emails
            self._log_progress("Buscando e-mails pendentes...")
            emails = list(self.gmail_connector.fetch_pending_emails(
                days_lookback=days_lookback or config.GMAIL_LOOKBACK_DAYS
            ))
            
            total = min(len(emails), max_emails)
            self._log_progress(f"Encontrados {len(emails)} e-mails, processando {total}")
            
            # Process each email
            for i, email in enumerate(emails[:max_emails]):
                if self._stop_requested:
                    self._log_progress("Pipeline interrompido pelo usuário", level="warning")
                    break
                
                self._log_progress(
                    f"Processando: {email.subject[:50]}...",
                    current=i + 1,
                    total=total
                )
                
                result = self._process_single_email(email)
                results.append(result)
                
                # Update stats
                self.stats.emails_processed += 1
                if result.status == ProcessingStatus.SUCCESS:
                    self.stats.emails_succeeded += 1
                elif result.status == ProcessingStatus.FAILED:
                    self.stats.emails_failed += 1
                elif result.status in (ProcessingStatus.SKIPPED, ProcessingStatus.DUPLICATE):
                    self.stats.emails_skipped += 1
                
                self.stats.documents_extracted += result.documents_extracted
                self.stats.gemini_tokens_used += result.gemini_tokens_used
            
            self.stats.end_time = datetime.now()
            self._log_progress(
                f"Pipeline concluído: {self.stats.emails_succeeded}/{self.stats.emails_processed} "
                f"em {self.stats.duration_seconds:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._log_progress(f"Erro no pipeline: {e}", level="error")
            
        finally:
            self._running = False
        
        return results
    
    def _process_single_email(self, email) -> EmailProcessingResult:
        """
        Process a single email through the extraction pipeline.
        
        Workflow:
        1. Check for XML attachment (deterministic path)
        2. Check for PDF attachment (hybrid path)
        3. Hunt for links (slow path)
        4. Analyze text body (last resort)
        """
        import time
        start_time = time.time()
        
        result = EmailProcessingResult(
            email_id=email.id,
            subject=email.subject,
            status=ProcessingStatus.PROCESSING
        )
        
        try:
            # PATH 1: XML attachment (TRUTH - deterministic)
            xml_attachments = email.get_xml_attachments()
            if xml_attachments:
                self._log_progress(f"  → Encontrado XML: {xml_attachments[0].filename}")
                result = self._process_xml_path(email, xml_attachments)
                
            # PATH 2: PDF attachment (hybrid - OCR + Gemini if needed)
            elif email.has_pdf_attachment():
                pdf_attachments = email.get_pdf_attachments()
                self._log_progress(f"  → Encontrado PDF: {pdf_attachments[0].filename}")
                result = self._process_pdf_path(email, pdf_attachments)
                
            # PATH 3: Link hunting (slow - Playwright)
            elif email.html_body:
                self._log_progress(f"  → Buscando links no corpo do e-mail...")
                result = self._process_link_path(email)
                
            # PATH 4: Text body analysis (last resort)
            elif email.text_body:
                self._log_progress(f"  → Analisando texto do e-mail...")
                result = self._process_text_path(email)
                
            else:
                result.status = ProcessingStatus.SKIPPED
                result.error_message = "No processable content"
            
            # Apply Gmail labels based on result
            self._apply_labels(email.id, result)
            
        except Exception as e:
            logger.exception(f"Error processing email {email.id}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            self.gmail_connector.apply_label(email.id, 'failed')
        
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result
    
    def _process_xml_path(self, email, xml_attachments) -> EmailProcessingResult:
        """
        Process email with XML attachment (deterministic path).
        
        XML is the source of truth - no OCR needed.
        """
        result = EmailProcessingResult(
            email_id=email.id,
            subject=email.subject,
            status=ProcessingStatus.PROCESSING
        )
        
        for attachment in xml_attachments:
            # Check for duplicate
            if self._check_duplicate(attachment.md5_hash):
                result.status = ProcessingStatus.DUPLICATE
                return result
            
            # Save and parse XML
            temp_path = self._save_attachment_temp(attachment)
            
            try:
                # Parse XML (NFe/NFSe)
                from lxml import etree
                tree = etree.parse(str(temp_path))
                
                # Extract data from XML
                # (This would need specific XPath for NFe/NFSe structure)
                # For now, delegate to scanner
                data = self.scanner.process_file(str(temp_path))
                
                if data:
                    result.documents_extracted += 1
                    result.status = ProcessingStatus.SUCCESS
                    
            except Exception as e:
                logger.error(f"XML parsing error: {e}")
                result.error_message = str(e)
                result.status = ProcessingStatus.FAILED
            finally:
                temp_path.unlink(missing_ok=True)
        
        return result
    
    def _process_pdf_path(self, email, pdf_attachments) -> EmailProcessingResult:
        """
        Process email with PDF attachment (hybrid path).
        
        Uses OCR + Gemini fallback if needed.
        """
        result = EmailProcessingResult(
            email_id=email.id,
            subject=email.subject,
            status=ProcessingStatus.PROCESSING
        )
        
        for attachment in pdf_attachments:
            # Check for duplicate
            if self._check_duplicate(attachment.md5_hash):
                result.status = ProcessingStatus.DUPLICATE
                continue
            
            # Save PDF temporarily
            temp_path = self._save_attachment_temp(attachment)
            
            try:
                # Process through scanner (with Gemini fallback if enabled)
                data = self.scanner.process_file(str(temp_path))
                
                if data:
                    result.documents_extracted += 1
                    result.status = ProcessingStatus.SUCCESS
                    
                    # Track Gemini usage
                    if hasattr(data, 'gemini_tokens_used'):
                        result.gemini_tokens_used += data.gemini_tokens_used
                else:
                    result.status = ProcessingStatus.FAILED
                    result.error_message = "Extraction returned no data"
                    
            except Exception as e:
                logger.error(f"PDF processing error: {e}")
                result.error_message = str(e)
                result.status = ProcessingStatus.FAILED
            finally:
                temp_path.unlink(missing_ok=True)
        
        if result.documents_extracted > 0:
            result.status = ProcessingStatus.SUCCESS
            
        return result
    
    def _process_link_path(self, email) -> EmailProcessingResult:
        """
        Process email by finding and downloading from links.
        
        Uses multilevel approach:
        1. Try BeautifulSoup link extraction (fast)
        2. If no links found, ask Gemini to identify URLs (aggressive)
        3. Download with Playwright for JavaScript-heavy pages
        """
        result = EmailProcessingResult(
            email_id=email.id,
            subject=email.subject,
            status=ProcessingStatus.PROCESSING
        )
        
        # Level 1: Extract candidate links with BeautifulSoup
        links = self.link_extractor.extract_links(email.html_body)
        
        # Level 2: If no links found, use Gemini to identify URLs (AGGRESSIVE HUNT)
        if not links and self.use_cloud_ai:
            self._log_progress("    → Sem links detectados. Consultando Gemini...")
            links = self._gemini_hunt_links(email.html_body or email.text_body)
        
        if not links:
            result.status = ProcessingStatus.SKIPPED
            result.error_message = "No candidate links found"
            return result
        
        # Try top 3 links
        for link in links[:3]:
            try:
                link_url = getattr(link, 'url', link) if hasattr(link, 'url') else link
                link_text = getattr(link, 'text', link_url[:30]) if hasattr(link, 'text') else str(link_url)[:30]
                self._log_progress(f"    → Baixando: {link_text}...")
                
                from integrations.link_extractor import download_link_sync
                downloaded = download_link_sync(link_url)
                
                if downloaded and downloaded.exists():
                    # Process downloaded file
                    data = self.scanner.process_file(str(downloaded))
                    
                    if data:
                        result.documents_extracted += 1
                        result.status = ProcessingStatus.SUCCESS
                        break
                        
            except Exception as e:
                logger.warning(f"Link download failed: {e}")
                continue
        
        if result.documents_extracted == 0:
            result.status = ProcessingStatus.FAILED
            result.error_message = "No documents extracted from links"
        
        return result
    
    def _gemini_hunt_links(self, content: str) -> List[str]:
        """
        Use Gemini AI to find download URLs in email content.
        
        Aggressive hunt mode - the AI reads the entire email and 
        identifies any links that might lead to boletos or invoices.
        """
        if not content:
            return []
        
        try:
            from voters.gemini_voter import get_gemini_voter
            voter = get_gemini_voter()
            
            if not voter.is_available():
                return []
            
            # Specific prompt for link hunting
            prompt = f"""Analise este conteúdo de e-mail e identifique URLs que provavelmente levam a boletos, 
notas fiscais ou documentos financeiros para download.

Conteúdo do E-mail:
---
{content[:8000]}
---

Retorne APENAS um JSON válido no formato:
{{"urls": ["url1", "url2"]}}

Se não houver nenhuma URL relevante, retorne:
{{"urls": []}}

IMPORTANTE: Retorne APENAS o JSON, sem explicações."""

            import google.generativeai as genai
            import json
            
            # Use direct text generation (cheaper)
            result = voter._get_model(voter.default_model).generate_content(prompt)
            
            if result and result.text:
                # Parse JSON response
                import re
                cleaned = result.text.strip()
                if cleaned.startswith('```'):
                    cleaned = re.sub(r'^```\\w*\\n?', '', cleaned)
                    cleaned = re.sub(r'\\n?```$', '', cleaned)
                
                data = json.loads(cleaned)
                urls = data.get('urls', [])
                
                if urls:
                    logger.info(f"[GEMINI LINK HUNTER] Encontradas {len(urls)} URLs")
                return urls
                
        except Exception as e:
            logger.warning(f"Gemini link hunting failed: {e}")
        
        return []
    
    def _process_text_path(self, email) -> EmailProcessingResult:
        """
        Process email by analyzing text body with Gemini.
        
        Last resort when no attachments or links available.
        """
        result = EmailProcessingResult(
            email_id=email.id,
            subject=email.subject,
            status=ProcessingStatus.PROCESSING
        )
        
        if not self.use_cloud_ai:
            result.status = ProcessingStatus.SKIPPED
            result.error_message = "Cloud AI disabled, cannot analyze text"
            return result
        
        try:
            from voters.gemini_voter import get_gemini_voter
            voter = get_gemini_voter()
            
            text_content = f"Assunto: {email.subject}\n\n{email.text_body or email.html_body}"
            extraction = voter.extract_from_text(text_content)
            
            if extraction.success and extraction.data:
                result.documents_extracted += 1
                result.gemini_tokens_used = extraction.tokens_used
                result.status = ProcessingStatus.SUCCESS
                
                # Save extracted data as JSON
                output_path = config.OUTPUT_DIR / f"email_{email.id[:8]}_extracted.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                import json
                output_path.write_text(json.dumps(extraction.data, indent=2, ensure_ascii=False))
                result.files_created.append(str(output_path))
                
            else:
                result.status = ProcessingStatus.FAILED
                result.error_message = extraction.error or "No data extracted"
                
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
        
        return result
    
    def _check_duplicate(self, file_hash: str) -> bool:
        """Check if document with this hash already exists."""
        if not self.db:
            return False
        try:
            return self.db.document_exists(file_hash)
        except Exception:
            return False
    
    def _save_attachment_temp(self, attachment) -> Path:
        """Save attachment to temp directory."""
        config.TEMP_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = config.TEMP_DOWNLOADS_DIR / f"att_{attachment.md5_hash[:8]}_{attachment.filename}"
        temp_path.write_bytes(attachment.data)
        return temp_path
    
    def _apply_labels(self, email_id: str, result: EmailProcessingResult):
        """Apply appropriate Gmail labels based on processing result."""
        try:
            if result.status == ProcessingStatus.SUCCESS:
                self.gmail_connector.apply_label(email_id, 'processed')
                if result.gemini_tokens_used > 0:
                    self.gmail_connector.apply_label(email_id, 'ai_analyzed')
                    
            elif result.status == ProcessingStatus.DUPLICATE:
                self.gmail_connector.apply_label(email_id, 'processed')
                self.gmail_connector.apply_label(email_id, 'duplicate')
                
            elif result.status == ProcessingStatus.FAILED:
                self.gmail_connector.apply_label(email_id, 'failed')
                
        except Exception as e:
            logger.warning(f"Failed to apply labels: {e}")


# Convenience function
def run_email_pipeline(
    days_lookback: int = None,
    max_emails: int = 50,
    use_cloud_ai: bool = True,
    progress_callback: Callable = None
) -> List[EmailProcessingResult]:
    """
    Run the email pipeline with default settings.
    
    Args:
        days_lookback: Days to look back for emails
        max_emails: Maximum emails to process
        use_cloud_ai: Enable Gemini AI
        progress_callback: Optional progress callback
        
    Returns:
        List of processing results
    """
    pipeline = EmailPipeline(
        use_cloud_ai=use_cloud_ai,
        progress_callback=progress_callback
    )
    return pipeline.process_pending_emails(
        days_lookback=days_lookback,
        max_emails=max_emails
    )


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Email Pipeline Test")
    print("=" * 50)
    
    def progress(msg, current, total):
        if total > 0:
            print(f"[{current}/{total}] {msg}")
        else:
            print(f"[*] {msg}")
    
    # Parse args
    max_emails = 5
    if len(sys.argv) > 1:
        try:
            max_emails = int(sys.argv[1])
        except ValueError:
            pass
    
    print(f"\nProcessing up to {max_emails} emails...\n")
    
    results = run_email_pipeline(
        max_emails=max_emails,
        use_cloud_ai=True,
        progress_callback=progress
    )
    
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    
    for result in results:
        status_icon = {
            ProcessingStatus.SUCCESS: "✅",
            ProcessingStatus.FAILED: "❌",
            ProcessingStatus.SKIPPED: "⏭️",
            ProcessingStatus.DUPLICATE: "🔄"
        }.get(result.status, "❓")
        
        print(f"{status_icon} {result.subject[:40]}...")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        if result.documents_extracted > 0:
            print(f"   Docs: {result.documents_extracted}")
