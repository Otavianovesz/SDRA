"""Deep analysis of Boleto supplier extraction failures."""
import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import fitz

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"


def analyze_boleto_text():
    """Show full text of failing boletos to find patterns."""
    
    # Specific failing boletos from analysis
    failing_names = [
        "SANTA CLARA",
        "MV PECAS",
        "CASA DE CARNES"
    ]
    
    print("=" * 80)
    print("BOLETO TEXT ANALYSIS FOR PATTERN IMPROVEMENT")
    print("=" * 80)
    
    for search in failing_names:
        matches = list(NFE_DIR.glob(f"*{search}*BOLETO*.pdf"))[:1]
        if not matches:
            matches = list(NFE_DIR.glob(f"*{search}*BOLETO*.PDF"))[:1]
        
        if not matches:
            print(f"\nNo files found for: {search}")
            continue
        
        pdf_path = matches[0]
        print(f"\n{'='*80}")
        print(f"FILE: {pdf_path.name}")
        print('='*80)
        
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if len(text) < 100:
                print("  TEXT TOO SHORT - likely scanned")
                continue
            
            # Show first 2000 chars
            print(text[:2000])
            
            # Try to find key patterns
            print("\n--- PATTERN SEARCH ---")
            patterns = [
                (r'Cedente[\s\n:]+([^\n]+)', 'Cedente'),
                (r'Benefici.rio[\s\n:]+([^\n]+)', 'Beneficiario'),
                (r'([A-Z][A-Z\s,&]+(?:LTDA|CIA|EIRELI))', 'Company with LTDA'),
                (r'CNPJ[:\s]+[\d\./-]+[\s\n]+([A-Z][A-Z\s]+)', 'After CNPJ'),
            ]
            
            for pat, name in patterns:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    print(f"  {name}: {m.group(1)[:50]}")
                    
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    analyze_boleto_text()
