
import pytest
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_extractor import RegexVoter, DocumentType

@pytest.fixture(scope="module")
def regex_voter():
    """Initializes RegexVoter with default patterns."""
    return RegexVoter(patterns_file="extraction_patterns.yaml")

class TestRegexVoter:
    
    def test_init(self, regex_voter):
        """Verify patterns are loaded."""
        assert regex_voter.doc_number_patterns
        assert regex_voter.supplier_patterns
        assert "NFE" in regex_voter.doc_number_patterns
    
    def test_extract_doc_number_nfe(self, regex_voter):
        text = "NOTA FISCAL No 12345 SÉRIE 1"
        res = regex_voter.extract_doc_number(text, DocumentType.NFE)
        assert res.value == "12345"
        
        text2 = "NFe: 000.001.234"
        res2 = regex_voter.extract_doc_number(text2, DocumentType.NFE)
        assert res2.value == "1234" # Cleaning usually removes leading zeros? Or pattern extract group. 
        # Check pattern: NFe[:\s]*(\d{1,9}) -> 000 will be matched?
        # Actually pattern is usually group(1). 
        # Let's rely on standard NFE pattern.
    
    def test_extract_doc_number_nfse(self, regex_voter):
        text = "Numero da Nota: 2024"
        res = regex_voter.extract_doc_number(text, DocumentType.NFSE)
        assert res.value == "2024"
    
    def test_extract_doc_number_boleto(self, regex_voter):
        text = "Nosso Numero: 987654321"
        res = regex_voter.extract_doc_number(text, DocumentType.BOLETO)
        assert res.value == "987654321"

    def test_extract_supplier_nfe(self, regex_voter):
        text = "RAZAO SOCIAL: FORNECEDOR TESTE LTDA CNPJ: 12.345.678/0001-90"
        # NFE pattern usually looks for RAZAO SOCIAL
        res = regex_voter.extract_supplier(text, DocumentType.NFE)
        assert "FORNECEDOR TESTE" in res.value
    
    def test_extract_sisbb(self, regex_voter):
        text = "SISBB - SISTEMA DE INFORMACOES BANCO DO BRASIL\nJ.B.5.213.432"
        res = regex_voter.extract_sisbb(text)
        assert res.value == "J.B.5.213.432"
        
    def test_check_scheduling(self, regex_voter):
        text = "Nao houve agendamento."
        assert not regex_voter.check_scheduling(text)
        
        text2 = "Transacao agendada para 20/12/2025"
        assert regex_voter.check_scheduling(text2)
        
    def test_clean_name(self, regex_voter):
        # We need to access _clean_name or verify extraction result structure
        # RegexVoter helper _clean_name usage
        name = "   TESTE LTDA   "
        # Since _clean_name is used inside extract, we test effect
        # But _clean_name is now in EnsembleExtractor too? 
        # RegexVoter has its own `_clean_name` (added in Step 2083).
        cleaned = regex_voter._clean_name(name)
        assert cleaned == "TESTE"

