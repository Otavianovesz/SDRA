"""
Diagnostic Test - Identify specific extraction failures
"""

import os
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor, DocumentType

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"
COMPROVANTES_DIR = Path(__file__).parent.parent / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"


def diagnose_nfe_failures():
    """Diagnose NFE supplier extraction failures."""
    print("=" * 80)
    print("NFE SUPPLIER EXTRACTION DIAGNOSIS")
    print("=" * 80)
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    # Get NFE files
    pdf_files = list(NFE_DIR.glob("*_NFE_*.pdf"))[:10]
    
    failures = []
    
    for pdf in pdf_files:
        # Parse expected from filename
        parts = pdf.stem.split('_')
        expected_supplier = parts[2] if len(parts) > 2 else "UNKNOWN"
        
        # Extract
        result = extractor.extract_from_pdf(str(pdf))
        extracted = result.fornecedor or "NONE"
        
        # Compare
        match = expected_supplier.upper() in extracted.upper() or extracted.upper() in expected_supplier.upper()
        
        if not match:
            failures.append({
                'file': pdf.name[:60],
                'expected': expected_supplier,
                'extracted': extracted,
                'raw_text_preview': result.raw_text[:500] if result.raw_text else "NO TEXT"
            })
            print(f"\nFAILURE: {pdf.name[:50]}")
            print(f"  Expected: {expected_supplier}")
            print(f"  Extracted: {extracted}")
            
            # Show relevant text patterns
            if result.raw_text:
                text = result.raw_text
                # Look for key patterns
                patterns = [
                    (r'RECEBEMOS\s+DE\s+([^\n]+)', 'RECEBEMOS DE'),
                    (r'Raz[aã]o\s*[Ss]ocial[:\s]+([^\n]+)', 'Razao Social'),
                    (r'EMITENTE[:\s]+([^\n]+)', 'EMITENTE'),
                    (r'PRESTADOR[^\n]*\n([^\n]+)', 'PRESTADOR'),
                    (r'Benefici[aá]rio[:\s]+([^\n]+)', 'Beneficiario'),
                ]
                print("  Text patterns found:")
                for pat, name in patterns:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        print(f"    {name}: {m.group(1)[:50]}")
    
    print(f"\n\nTotal failures: {len(failures)}/{len(pdf_files)}")
    return failures


def diagnose_comprovante_extraction():
    """Diagnose comprovante extraction to see what's available."""
    print("\n" + "=" * 80)
    print("COMPROVANTE EXTRACTION DIAGNOSIS")
    print("=" * 80)
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    # Get new-format comprovantes (they have supplier in filename)
    new_format = [f for f in COMPROVANTES_DIR.glob("2025_*.pdf")][:5]
    
    for pdf in new_format:
        print(f"\n--- {pdf.name[:60]} ---")
        
        result = extractor.extract_from_pdf(str(pdf))
        
        print(f"  Type: {result.doc_type}")
        print(f"  Amount: {result.amount_cents} cents")
        print(f"  Supplier: {result.fornecedor or 'NONE'}")
        print(f"  Doc Number: {result.doc_number or 'NONE'}")
        print(f"  SISBB Auth: {result.sisbb_auth or 'NONE'}")
        
        # Show first part of text
        if result.raw_text:
            print(f"  Text preview: {result.raw_text[:200].replace(chr(10), ' ')}")
            
            # Look for key patterns in comprovantes
            text = result.raw_text
            patterns = [
                (r'Favorecido[:\s]+([^\n]+)', 'Favorecido'),
                (r'Nome[:\s]+([^\n]+)', 'Nome'),
                (r'Destinatário[:\s]+([^\n]+)', 'Destinatario'),
                (r'Beneficiário[:\s]+([^\n]+)', 'Beneficiario'),
            ]
            print("  Supplier patterns found:")
            for pat, name in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    print(f"    {name}: {m.group(1)[:40]}")


if __name__ == "__main__":
    diagnose_nfe_failures()
    diagnose_comprovante_extraction()
