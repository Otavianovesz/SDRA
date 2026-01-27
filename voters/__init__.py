"""
SRDA Voters Package
===================
Contains voting components for data extraction:
- Gemini AI voter (cloud-based semantic OCR)
"""

from .gemini_voter import GeminiVoter, get_gemini_voter, GeminiExtractionResult

__all__ = [
    'GeminiVoter',
    'get_gemini_voter',
    'GeminiExtractionResult'
]
