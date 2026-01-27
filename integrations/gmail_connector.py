"""
Gmail Connector Module
======================
Handles Gmail API authentication, message fetching, and label management.

Key Features:
- OAuth 2.0 with automatic token refresh
- Label creation and management (colored labels)
- Paginated email fetching
- Attachment extraction with hash deduplication

Part of Project Cyborg - SRDA Autonomous Treasury Agent
"""

import os
import base64
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator, Callable
from dataclasses import dataclass, field

# Lazy imports to avoid loading heavy libs if not used
_google_libs_loaded = False
Request = None
Credentials = None
InstalledAppFlow = None
build = None
HttpError = None

logger = logging.getLogger('srda.gmail')


def _ensure_google_libs():
    """Lazy import Google API libraries."""
    global _google_libs_loaded, Request, Credentials, InstalledAppFlow, build, HttpError
    
    if _google_libs_loaded:
        return
    
    try:
        from google.auth.transport.requests import Request as _Request
        from google.oauth2.credentials import Credentials as _Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow as _InstalledAppFlow
        from googleapiclient.discovery import build as _build
        from googleapiclient.errors import HttpError as _HttpError
        
        Request = _Request
        Credentials = _Credentials
        InstalledAppFlow = _InstalledAppFlow
        build = _build
        HttpError = _HttpError
        _google_libs_loaded = True
        
    except ImportError as e:
        logger.error(f"Google API libraries not installed: {e}")
        raise ImportError(
            "Gmail integration requires: google-auth-oauthlib, google-api-python-client. "
            "Install with: pip install google-auth-oauthlib google-api-python-client"
        )


# Import config
try:
    import config
except ImportError:
    # Fallback for standalone testing
    class config:
        from pathlib import Path
        BASE_DIR = Path(".")
        GMAIL_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
        GMAIL_TOKEN_PATH = BASE_DIR / "token.json"
        GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
        GMAIL_MAX_RESULTS = 20
        GMAIL_LOOKBACK_DAYS = 5
        GMAIL_LABELS = {
            "processed": "SRDA_PROCESSADO",
            "failed": "SRDA_FALHA_EXTRAÇÃO",
            "ai_analyzed": "SRDA_IA_ANALISOU",
            "duplicate": "SRDA_DUPLICADO"
        }


# Label colors (Google's predefined palette)
LABEL_COLORS = {
    "processed": {"backgroundColor": "#16a765", "textColor": "#ffffff"},   # Green
    "failed": {"backgroundColor": "#cc3a21", "textColor": "#ffffff"},      # Red
    "ai_analyzed": {"backgroundColor": "#653e9b", "textColor": "#ffffff"}, # Purple
    "duplicate": {"backgroundColor": "#ffad47", "textColor": "#000000"}    # Orange
}


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    data: bytes
    mime_type: str
    md5_hash: str
    
    def save_to(self, directory: Path) -> Path:
        """Save attachment to directory, return path."""
        directory.mkdir(parents=True, exist_ok=True)
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in self.filename)
        output_path = directory / safe_name
        output_path.write_bytes(self.data)
        return output_path


@dataclass  
class EmailMessage:
    """Represents a parsed email message."""
    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    attachments: List[EmailAttachment] = field(default_factory=list)
    html_body: Optional[str] = None
    text_body: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    
    def has_pdf_attachment(self) -> bool:
        """Check if message has PDF attachment."""
        return any(a.filename.lower().endswith('.pdf') for a in self.attachments)
    
    def has_xml_attachment(self) -> bool:
        """Check if message has XML attachment."""
        return any(a.filename.lower().endswith('.xml') for a in self.attachments)
    
    def get_pdf_attachments(self) -> List[EmailAttachment]:
        """Get all PDF attachments."""
        return [a for a in self.attachments if a.filename.lower().endswith('.pdf')]
    
    def get_xml_attachments(self) -> List[EmailAttachment]:
        """Get all XML attachments."""
        return [a for a in self.attachments if a.filename.lower().endswith('.xml')]


class GmailConnector:
    """
    Gmail API connector with OAuth 2.0 authentication.
    
    Implements:
    - Lazy authentication with token persistence
    - Automatic token refresh
    - Label management (create, apply)
    - Paginated message fetching
    - Attachment extraction with MD5 hashing
    
    Usage:
        connector = GmailConnector()
        for email in connector.fetch_pending_emails():
            process(email)
            connector.apply_label(email.id, 'processed')
    """
    
    def __init__(
        self,
        credentials_path: Path = None,
        token_path: Path = None,
        scopes: List[str] = None
    ):
        self.credentials_path = credentials_path or config.GMAIL_CREDENTIALS_PATH
        self.token_path = token_path or config.GMAIL_TOKEN_PATH
        self.scopes = scopes or config.GMAIL_SCOPES
        self._service = None
        self._credentials = None
        self._label_ids: Dict[str, str] = {}
        
    def authenticate(self) -> bool:
        """
        Authenticate with Gmail API.
        
        Token lifecycle:
        1. If token.json exists and valid → use it
        2. If token.json exists and expired → refresh it
        3. If no token.json → open browser for OAuth flow
        
        Returns:
            True if authentication successful
        """
        _ensure_google_libs()
        
        try:
            # Try to load existing token
            if Path(self.token_path).exists():
                self._credentials = Credentials.from_authorized_user_file(
                    str(self.token_path), 
                    self.scopes
                )
            
            # Check if credentials need refresh or creation
            if not self._credentials or not self._credentials.valid:
                if self._credentials and self._credentials.expired and self._credentials.refresh_token:
                    logger.info("Refreshing expired Gmail token...")
                    self._credentials.refresh(Request())
                else:
                    logger.info("Starting OAuth flow for Gmail...")
                    if not Path(self.credentials_path).exists():
                        raise FileNotFoundError(
                            f"credentials.json not found at {self.credentials_path}. "
                            "Download from Google Cloud Console."
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path),
                        self.scopes
                    )
                    self._credentials = flow.run_local_server(port=0)
                
                # Save the credentials for future runs
                with open(self.token_path, 'w') as token:
                    token.write(self._credentials.to_json())
                logger.info("Gmail token saved successfully")
            
            # Build the service
            self._service = build('gmail', 'v1', credentials=self._credentials)
            logger.info("Gmail API authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Gmail authentication failed: {e}")
            return False
    
    @property
    def service(self):
        """Get authenticated Gmail service, authenticating if needed."""
        if not self._service:
            if not self.authenticate():
                raise RuntimeError("Gmail authentication failed")
        return self._service
    
    def ensure_labels_exist(self) -> Dict[str, str]:
        """
        Create SRDA labels if they don't exist.
        
        Returns:
            Dict mapping label keys to label IDs
        """
        if self._label_ids:
            return self._label_ids
        
        _ensure_google_libs()
        
        try:
            # Get existing labels
            results = self.service.users().labels().list(userId='me').execute()
            existing = {label['name']: label['id'] for label in results.get('labels', [])}
            
            for key, name in config.GMAIL_LABELS.items():
                if name in existing:
                    self._label_ids[key] = existing[name]
                    logger.debug(f"Label '{name}' already exists")
                else:
                    # Create new label with color
                    color = LABEL_COLORS.get(key, {})
                    label_body = {
                        'name': name,
                        'labelListVisibility': 'labelShow',
                        'messageListVisibility': 'show'
                    }
                    if color:
                        label_body['color'] = color
                    
                    created = self.service.users().labels().create(
                        userId='me',
                        body=label_body
                    ).execute()
                    self._label_ids[key] = created['id']
                    logger.info(f"Created Gmail label: {name} ({key})")
            
            return self._label_ids
            
        except HttpError as e:
            logger.error(f"Failed to manage labels: {e}")
            return {}
    
    def apply_label(self, message_id: str, label_key: str) -> bool:
        """
        Apply an SRDA label to a message.
        
        Args:
            message_id: Gmail message ID
            label_key: Key from config.GMAIL_LABELS (e.g., 'processed')
            
        Returns:
            True if successful
        """
        _ensure_google_libs()
        
        try:
            label_ids = self.ensure_labels_exist()
            if label_key not in label_ids:
                logger.error(f"Unknown label key: {label_key}")
                return False
            
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [label_ids[label_key]]}
            ).execute()
            
            logger.debug(f"Applied label '{label_key}' to message {message_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to apply label: {e}")
            return False
    
    def remove_label(self, message_id: str, label_key: str) -> bool:
        """
        Remove an SRDA label from a message.
        
        Args:
            message_id: Gmail message ID
            label_key: Key from config.GMAIL_LABELS
            
        Returns:
            True if successful
        """
        _ensure_google_libs()
        
        try:
            label_ids = self.ensure_labels_exist()
            if label_key not in label_ids:
                return False
            
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': [label_ids[label_key]]}
            ).execute()
            
            return True
            
        except HttpError as e:
            logger.error(f"Failed to remove label: {e}")
            return False
    
    def fetch_pending_emails(
        self,
        days_lookback: int = None,
        max_results: int = None,
        custom_query: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Generator[EmailMessage, None, None]:
        """
        Fetch emails pending processing.
        
        Query filters:
        - Has attachment OR filename:pdf OR filename:xml
        - NOT labeled SRDA_PROCESSADO
        - Within lookback period
        
        Args:
            days_lookback: Days to look back (default from config)
            max_results: Max results per page (default from config)
            custom_query: Override default query
            progress_callback: Called with (current, total) counts
            
        Yields:
            EmailMessage objects with parsed content
        """
        days_lookback = days_lookback or config.GMAIL_LOOKBACK_DAYS
        max_results = max_results or config.GMAIL_MAX_RESULTS
        
        # Calculate date threshold
        date_threshold = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y/%m/%d')
        
        # Build query
        if custom_query:
            query = custom_query
        else:
            query_parts = [
                "(has:attachment OR filename:pdf OR filename:xml)",
                f"-label:{config.GMAIL_LABELS['processed']}",
                f"after:{date_threshold}"
            ]
            query = " ".join(query_parts)
        
        logger.info(f"Fetching emails with query: {query}")
        
        _ensure_google_libs()
        
        try:
            page_token = None
            total_fetched = 0
            
            while True:
                # Fetch batch of messages with exponential backoff
                results = self._execute_with_backoff(
                    lambda: self.service.users().messages().list(
                        userId='me',
                        q=query,
                        maxResults=max_results,
                        pageToken=page_token
                    ).execute()
                )
                
                messages = results.get('messages', [])
                if not messages:
                    logger.info("No more messages to process")
                    break
                
                total = results.get('resultSizeEstimate', len(messages))
                
                # Process each message
                for i, msg_ref in enumerate(messages):
                    try:
                        email = self._parse_message(msg_ref['id'])
                        if email:
                            total_fetched += 1
                            if progress_callback:
                                progress_callback(total_fetched, total)
                            yield email
                    except Exception as e:
                        logger.error(f"Failed to parse message {msg_ref['id']}: {e}")
                        continue
                
                # Check for next page
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
                    
            logger.info(f"Total emails fetched: {total_fetched}")
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            raise
    
    def _execute_with_backoff(self, func, max_retries: int = 5):
        """Execute function with exponential backoff for rate limiting."""
        _ensure_google_libs()
        
        for attempt in range(max_retries):
            try:
                return func()
            except HttpError as e:
                if e.resp.status in (429, 500, 503):
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise RuntimeError(f"Max retries ({max_retries}) exceeded")
    
    def _parse_message(self, message_id: str) -> Optional[EmailMessage]:
        """
        Parse a Gmail message into EmailMessage dataclass.
        
        Extracts:
        - Headers (subject, from, date)
        - Attachments (PDF, XML with MD5 hash)
        - Body (HTML and text)
        """
        _ensure_google_libs()
        
        try:
            msg = self._execute_with_backoff(
                lambda: self.service.users().messages().get(
                    userId='me',
                    id=message_id,
                    format='full'
                ).execute()
            )
            
            # Extract headers
            headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}
            
            # Extract attachments and body
            attachments = []
            html_body = None
            text_body = None
            
            def set_html(h):
                nonlocal html_body
                html_body = h
            
            def set_text(t):
                nonlocal text_body
                text_body = t
            
            self._process_parts(
                msg['payload'],
                message_id,
                attachments,
                set_html,
                set_text
            )
            
            return EmailMessage(
                id=message_id,
                thread_id=msg.get('threadId', ''),
                subject=headers.get('subject', 'Sem Assunto'),
                sender=headers.get('from', 'Desconhecido'),
                date=headers.get('date', ''),
                attachments=attachments,
                html_body=html_body,
                text_body=text_body,
                labels=msg.get('labelIds', [])
            )
            
        except HttpError as e:
            logger.error(f"Failed to get message {message_id}: {e}")
            return None
    
    def _process_parts(
        self,
        payload: Dict,
        message_id: str,
        attachments: List[EmailAttachment],
        html_callback: Callable,
        text_callback: Callable
    ):
        """Recursively process message parts to extract attachments and body."""
        
        mime_type = payload.get('mimeType', '')
        filename = payload.get('filename', '')
        
        # Handle attachments
        if filename:
            filename_lower = filename.lower()
            if filename_lower.endswith(('.pdf', '.xml')):
                attachment_id = payload.get('body', {}).get('attachmentId', '')
                if attachment_id:
                    attachment = self._download_attachment(
                        message_id,
                        attachment_id,
                        filename,
                        mime_type
                    )
                    if attachment:
                        attachments.append(attachment)
        
        # Handle body
        elif 'body' in payload and 'data' in payload['body']:
            try:
                data = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
                if mime_type == 'text/html':
                    html_callback(data)
                elif mime_type == 'text/plain':
                    text_callback(data)
            except Exception as e:
                logger.warning(f"Failed to decode body: {e}")
        
        # Recurse into parts
        for part in payload.get('parts', []):
            self._process_parts(part, message_id, attachments, html_callback, text_callback)
    
    def _download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        filename: str,
        mime_type: str
    ) -> Optional[EmailAttachment]:
        """Download and hash an attachment."""
        _ensure_google_libs()
        
        try:
            attachment = self._execute_with_backoff(
                lambda: self.service.users().messages().attachments().get(
                    userId='me',
                    messageId=message_id,
                    id=attachment_id
                ).execute()
            )
            
            data = base64.urlsafe_b64decode(attachment['data'])
            
            # Check for empty/corrupted attachment
            if len(data) == 0:
                logger.warning(f"Empty attachment: {filename}")
                return None
            
            md5_hash = hashlib.md5(data).hexdigest()
            
            logger.debug(f"Downloaded attachment: {filename} ({len(data)} bytes, hash: {md5_hash[:8]})")
            
            return EmailAttachment(
                filename=filename,
                data=data,
                mime_type=mime_type,
                md5_hash=md5_hash
            )
            
        except HttpError as e:
            logger.error(f"Failed to download attachment {filename}: {e}")
            return None


# Singleton instance
_connector: Optional[GmailConnector] = None


def get_gmail_connector() -> GmailConnector:
    """Get or create Gmail connector singleton."""
    global _connector
    if _connector is None:
        _connector = GmailConnector()
    return _connector


# =============================================================================
# CLI for testing
# =============================================================================
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Gmail Connector Test")
    print("=" * 50)
    
    connector = get_gmail_connector()
    
    if not connector.authenticate():
        print("Authentication failed!")
        sys.exit(1)
    
    print("\nCreating labels...")
    labels = connector.ensure_labels_exist()
    print(f"Labels: {labels}")
    
    print("\nFetching pending emails...")
    count = 0
    for email in connector.fetch_pending_emails(days_lookback=2, max_results=5):
        count += 1
        print(f"\n{count}. {email.subject}")
        print(f"   From: {email.sender}")
        print(f"   Attachments: {[a.filename for a in email.attachments]}")
        if count >= 5:
            break
    
    print(f"\nTotal: {count} emails found")
