"""Show full Boleto text to understand structure."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import fitz

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"

# Find a Boleto file
boletos = list(NFE_DIR.glob("*BOLETO*.pdf"))[:3]

for pdf_path in boletos:
    print(f"\n{'='*80}")
    print(f"BOLETO: {pdf_path.name}")
    print('='*80)
    
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        print(text[:2000])  # First 2000 chars
    except Exception as e:
        print(f"Error: {e}")
