"""
SRDA Integrations Package
=========================
Contains external service integrations:
- Gmail API connector
- Link extractor with Playwright
"""

from .gmail_connector import GmailConnector, get_gmail_connector, EmailMessage, EmailAttachment
from .link_extractor import LinkExtractor, PlaywrightDownloader, CandidateLink

__all__ = [
    'GmailConnector',
    'get_gmail_connector', 
    'EmailMessage',
    'EmailAttachment',
    'LinkExtractor',
    'PlaywrightDownloader',
    'CandidateLink'
]
