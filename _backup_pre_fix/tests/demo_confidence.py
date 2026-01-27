"""
Demo script to show confidence flagging in action.
Processes a few files and displays the new review fields.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"
COMP_DIR = Path(__file__).parent.parent / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"


def demo_confidence_flagging():
    """Show confidence flagging on sample documents."""
    
    print("=" * 80)
    print("CONFIDENCE FLAGGING DEMO (v3.1)")
    print("=" * 80)
    print("Philosophy: 'Extract with warning is better than empty'")
    print()
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    # Get mixed sample files
    nfe_files = list(NFE_DIR.glob("*.pdf"))[:5]
    comp_files = list(COMP_DIR.glob("*.pdf"))[:3]
    files = nfe_files + comp_files
    
    print(f"Processing {len(files)} sample files...\n")
    
    flagged_count = 0
    
    for pdf in files:
        result = extractor.extract_from_pdf(str(pdf))
        
        print(f"--- {pdf.name[:50]} ---")
        print(f"  Tipo: {result.doc_type.name}")
        print(f"  Fornecedor: {result.fornecedor or 'NONE'}")
        print(f"  Valor: R$ {result.amount_cents/100:.2f}")
        
        # Show confidence fields
        if result.field_confidence:
            print(f"  Confiancas: {result.field_confidence}")
        
        if result.needs_review:
            flagged_count += 1
            print(f"  ⚠️  NEEDS REVIEW: True")
            if result.review_reasons:
                for reason in result.review_reasons:
                    print(f"      Motivo: {reason}")
        else:
            print(f"  [OK] Review: Not needed")
        
        if result.low_confidence_extractions:
            print(f"  Low confidence: {result.low_confidence_extractions}")
        
        print()
    
    print("=" * 80)
    print(f"Documents flagged for review: {flagged_count}/{len(files)}")
    print("=" * 80)


if __name__ == "__main__":
    demo_confidence_flagging()
