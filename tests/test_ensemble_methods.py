
import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_extractor import EnsembleExtractor, ExtractionResult, DocumentType, VoterResult, DateInfo, PaymentStatus

@pytest.fixture
def extractor():
    """Initializes EnsembleExtractor with mocked voters."""
    # We mock __init__ to avoid loading heavy models or files
    with patch("ensemble_extractor.EnsembleExtractor.__init__", return_value=None) as mock_init:
        ext = EnsembleExtractor()
        # Re-assign mocked voters manually since init was skipped
        ext.regex_voter = MagicMock()
        ext._gliner_voter = MagicMock() # Set backing field for property
        ext.use_gliner = True # Enable it
        ext.layout_voter = MagicMock()
        ext.ocr_voter = MagicMock()
        ext.classifier = MagicMock()
        ext.use_vlm = False
        
        # Add helper methods that are needed (or we can use partial mock)
        # But we want to test _resolve_* methods which are part of the class.
        # Since we skipped init, we need to ensure these methods are bound.
        # They are standard methods so they exist on instance.
        
        # We need _clean_supplier available (it is methods on class)
        # But since we skipped init, 'self' is the instance.
        # However, we need 'RegexVoter' patterns? No, we mock regex_voter.
        
        return ext

# We need to reimplement _clean_supplier or call the real one?
# Since we didn't mock the method, it runs the real code using 'self'.
# But 'self' usually has no state dep for _clean_supplier except regex usage?
# The real _clean_supplier uses 're' model which is imported.
# It uses no 'self' state. So it's fine.

class TestEnsembleMethods:
    
    def test_resolve_supplier_priority(self, extractor):
        # Setup inputs
        pdf_path = "dummy.pdf"
        full_text = "ignored"
        doc_type = DocumentType.NFE
        
        # Mock Voters
        # 1. GLiNER
        gliner_res = {"fornecedor": VoterResult("GLINER SUP", 0.8, "gliner")}
        
        # 2. Regex
        extractor.regex_voter.extract_supplier.return_value = VoterResult("REGEX SUP", 0.9, "regex")
        
        # 3. Layout Result
        layout_res = {"header_supplier": VoterResult("LAYOUT SUP", 0.7, "layout")}
        
        # 4. _is_valid_supplier mock
        extractor._is_valid_supplier = MagicMock(return_value=True)
        # 5. _clean_supplier mock (or rely on real one if simple)
        extractor._clean_supplier = MagicMock(return_value="REGEX SUP") 
        # Note: clean name logic might transform inputs. 
        # Let's mock it to return the input for simplicity here, except checking calls.
        
        def side_effect_clean(name): return name
        extractor._clean_supplier.side_effect = side_effect_clean

        # RUN
        supp, sources = extractor._resolve_supplier(pdf_path, full_text, doc_type, layout_res, gliner_res)
        
        # ASSERT
        # Should pick REGEX because confidence 0.9 > 0.8 > 0.7
        assert supp == "REGEX SUP"
        assert sources["fornecedor"] == "regex"
        
    def test_resolve_dates_nfse_fallback(self, extractor):
        # Test NFSE prioritizes filename date if no due date found
        pdf_path = "01.05.2025_DOC.pdf"
        full_text = "..."
        doc_type = DocumentType.NFSE
        layout_res = {}
        result = ExtractionResult()
        
        # Mock RegexVoter dates
        extractor.regex_voter.extract_dates.return_value = {
            "emission": VoterResult("2025-05-01", 0.9, "regex")
        }
        
        # Mock Filename Date
        extractor._extract_date_from_filename = MagicMock(return_value=DateInfo("2025-05-01", "filename", 1.0, "filename"))
        
        # RUN
        extractor._resolve_dates(pdf_path, full_text, doc_type, layout_res, result)
        
        # ASSERT
        # NFSE logic: if no due_date, check filename_date. 
        # Here we had emission and filename. Filename should be primary?
        # Logic: 
        # elif doc_type == DocumentType.NFSE:
        #    if result.due_date: ...
        #    elif result.filename_date: primary = filename ...
        
        assert result.due_date == "2025-05-01"
        assert result.date_selection_reason == "nfse_uses_filename_date(no_due)"
        
    def test_resolve_financials(self, extractor):
        full_text = "Valor R$ 100,00 Doc 123"
        doc_type = DocumentType.UNKNOWN
        result = ExtractionResult()
        
        # Mock regex
        extractor.regex_voter.extract_amount.return_value = VoterResult("10000", 0.9, "regex")
        extractor.regex_voter.extract_doc_number.return_value = VoterResult("123", 0.8, "regex")
        extractor.regex_voter.extract_sisbb.return_value = VoterResult(None, 0.0, "regex")
        extractor.regex_voter.check_scheduling.return_value = False
        
        # Mock confidence calc
        extractor._calculate_confidence = MagicMock(return_value=0.95)
        
        # RUN
        extractor._resolve_financials(full_text, doc_type, result)
        
        # ASSERT
        assert result.amount_cents == 10000
        assert result.doc_number == "123"
        assert result.payment_status == PaymentStatus.UNKNOWN
        assert result.confidence == 0.95
        assert not result.needs_review

