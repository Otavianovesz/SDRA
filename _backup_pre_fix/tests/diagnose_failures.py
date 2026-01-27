"""
Deep diagnosis of supplier extraction failures.
Looks at actual PDF text to understand why patterns fail.
"""
import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor, DocumentType
import fitz

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"


def diagnose_specific_failures():
    """Analyze specific failing files."""
    
    # Files that failed in simulation
    failing_files = [
        "28.11.2025_VG_MV PECAS AGRICOLAS_427,50_BOLETO_12524.pdf",
        "26.11.2025_VG_IGLIKOSKI E IGLIKOSKI_6.000,00_PARC 2-2_BOLETO_111712_111713.pdf",
        "21.11.2025_VG_MONTANA ASSISTEC_2.140,00_NFSE_353.pdf",
    ]
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    for filename in failing_files:
        # Find file (partial match)
        matches = list(NFE_DIR.glob(f"*{filename[:30]}*"))
        if not matches:
            # Try more flexible match
            parts = filename.split('_')
            if len(parts) >= 3:
                matches = list(NFE_DIR.glob(f"*{parts[2][:10]}*"))
        
        if not matches:
            print(f"Could not find file: {filename[:40]}")
            continue
        
        pdf_path = matches[0]
        print(f"\n{'='*80}")
        print(f"ANALYZING: {pdf_path.name}")
        print('='*80)
        
        # Get raw text
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"  ERROR reading: {e}")
            continue
        
        if len(text.strip()) < 50:
            print(f"  TEXT TOO SHORT: {len(text)} chars - likely scanned image PDF")
            continue
        
        # Run extraction
        result = extractor.extract_from_pdf(str(pdf_path))
        
        print(f"  Doc Type: {result.doc_type}")
        print(f"  Extracted Supplier: {result.fornecedor or 'NONE'}")
        print(f"  Amount: {result.amount_cents}")
        
        # Search for supplier patterns in text
        print("\n  -- SUPPLIER PATTERN SEARCH --")
        patterns = [
            (r'RECEBEMOS\s+DE\s+([A-Z][A-Z0-9\s\.,\-]+?)(?=\s+OS\s+PRODUTOS|\s+OS\s+SERVICOS)', 'RECEBEMOS DE'),
            (r'Raz[aã]o\s*[Ss]ocial[:\s\n]+([^\n]{5,50})', 'Razao Social'),
            (r'EMITENTE[:\s\n]+([^\n]{5,50})', 'EMITENTE'),
            (r'Benefici[aá]rio[:\s\n]+([^\n]{5,50})', 'Beneficiario'),
            (r'PRESTADOR[^\n]*\n([^\n]{5,50})', 'PRESTADOR'),
            (r'Nome[:\s]+([A-Z][A-Z\s]{8,40})', 'Nome'),
        ]
        
        found_any = False
        for pattern, name in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                found_any = True
                for m in matches[:2]:
                    clean = m.strip()[:50]
                    print(f"    {name}: '{clean}'")
        
        if not found_any:
            print("    NO PATTERNS FOUND!")
            print(f"\n  -- TEXT PREVIEW (first 800 chars) --")
            print(text[:800])


if __name__ == "__main__":
    diagnose_specific_failures()
