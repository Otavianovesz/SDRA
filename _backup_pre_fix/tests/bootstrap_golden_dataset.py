"""
Bootstrap Golden Dataset Generator

Generates initial ground truth JSON from current extractor output.
User must manually verify and correct the values, then set is_verified=True.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
PDF_FOLDERS = [
    PROJECT_ROOT / "11.2025_NOVEMBRO_1.547",
    PROJECT_ROOT / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"
]
OUTPUT_FILE = Path(__file__).parent / "golden_dataset.json"
SAMPLES_PER_FOLDER = 25  # 50 total


def bootstrap_dataset():
    """Generate initial golden dataset from current extractor."""
    
    print("=" * 80)
    print("GOLDEN DATASET BOOTSTRAP")
    print("=" * 80)
    
    extractor = EnsembleExtractor(use_gliner=False, use_vlm=False, use_ocr=False)
    
    # Collect sample files
    files = []
    for folder in PDF_FOLDERS:
        if folder.exists():
            folder_files = list(folder.glob("*.pdf"))[:SAMPLES_PER_FOLDER]
            files.extend(folder_files)
            print(f"Found {len(folder_files)} files in {folder.name}")
    
    print(f"\nProcessing {len(files)} files for ground truth generation...\n")
    
    dataset = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "description": "Ground truth dataset for SDRA extraction validation",
        "instructions": "Verify each entry and set is_verified=true after manual validation",
        "documents": []
    }
    
    success = 0
    errors = 0
    
    for i, pdf in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] {pdf.name[:50]}...", end=" ")
            
            result = extractor.extract_from_pdf(str(pdf))
            
            entry = {
                "file": pdf.name,
                "folder": pdf.parent.name,
                "ground_truth": {
                    "doc_type": result.doc_type.name if result.doc_type else "UNKNOWN",
                    "fornecedor": result.fornecedor or "",
                    "valor_centavos": result.amount_cents or 0,
                    "numero_documento": result.doc_number or "",
                    "sisbb_auth": result.sisbb_auth or ""
                },
                "extraction_confidence": result.confidence or 0.0,
                "meta": {
                    "is_verified": False,
                    "verified_by": "",
                    "notes": ""
                }
            }
            
            dataset["documents"].append(entry)
            success += 1
            print("OK")
            
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            dataset["documents"].append({
                "file": pdf.name,
                "folder": pdf.parent.name,
                "error": str(e),
                "meta": {"is_verified": False}
            })
    
    # Write JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("BOOTSTRAP COMPLETE")
    print("=" * 80)
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("\n[IMPORTANTE] Agora abra o arquivo JSON e CORRIJA os valores manualmente.")
    print("Mude 'is_verified' para true apenas nos documentos que voce validou.")
    print("=" * 80)


if __name__ == "__main__":
    bootstrap_dataset()
