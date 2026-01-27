"""Show full text of a comprovante to understand structure."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import fitz

COMPROVANTES_DIR = Path(__file__).parent.parent / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"

# Get a PIX comprovante (new format)
pix_files = list(COMPROVANTES_DIR.glob("2025_12_18_*_PIX_*.pdf"))[:2]

for pdf_path in pix_files:
    print(f"\n{'='*80}")
    print(f"FILE: {pdf_path.name}")
    print('='*80)
    
    doc = fitz.open(str(pdf_path))
    for page in doc:
        text = page.get_text()
        print(text)
    doc.close()
