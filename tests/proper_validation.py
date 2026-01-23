"""
PROPER VALIDATION TEST v2 - Fixed Filename Parser

Filename format: DD.MM.YYYY_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
Example: 01.11.2025_VG_CADORE BIDOIA_3.108,00_BOLETO_961905.pdf
"""
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor, DocumentType

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"


@dataclass
class FilenameGroundTruth:
    """Parsed ground truth from filename."""
    date: str  # DD.MM.YYYY
    entity: str  # VG or MV
    supplier: str  # FORNECEDOR
    amount_cents: int  # valor em centavos
    doc_type: str  # NFE, BOLETO, NFSE, FATURA
    doc_number: Optional[str]  # numero do documento
    raw_filename: str


def parse_filename(filename: str) -> Optional[FilenameGroundTruth]:
    """
    Parse filename to extract ground truth.
    Format: DD.MM.YYYY_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
    Example: 01.11.2025_VG_CADORE BIDOIA_3.108,00_BOLETO_961905.pdf
    """
    name = Path(filename).stem
    parts = name.split('_')
    
    if len(parts) < 5:
        return None
    
    # First part: date DD.MM.YYYY
    if not re.match(r'\d{2}\.\d{2}\.\d{4}', parts[0]):
        return None
    date = parts[0]
    
    # Second part: entity (VG or MV)
    entity = parts[1].upper()
    if entity not in ('VG', 'MV'):
        return None
    
    # Find the type and number from the end
    # Last or second-to-last should be type (NFE, BOLETO, NFSE etc)
    doc_type = None
    doc_number = None
    type_idx = -1
    
    for i, p in enumerate(parts[2:], start=2):
        if p.upper() in ('NFE', 'BOLETO', 'NFSE', 'FATURA', 'COMPROVANTE', 'PARC'):
            type_idx = i
            doc_type = p.upper()
            # Next part might be number
            if i + 1 < len(parts):
                doc_number = parts[i + 1]
            break
    
    if not doc_type:
        return None
    
    # Everything between entity and type is supplier + value
    # Value is typically second-to-last before type (format: X.XXX,XX)
    middle_parts = parts[2:type_idx]
    
    if not middle_parts:
        return None
    
    # Find value - typically contains comma for cents
    value_idx = -1
    amount_cents = 0
    for i, p in enumerate(middle_parts):
        if re.match(r'^\d+[.,]\d{2}$', p) or re.match(r'^\d{1,3}(?:\.\d{3})*,\d{2}$', p):
            value_idx = i
            # Parse: 3.108,00 -> 310800
            val_clean = p.replace('.', '').replace(',', '.')
            try:
                amount_cents = int(float(val_clean) * 100)
            except:
                pass
            break
    
    if value_idx < 0:
        return None
    
    # Everything before value is supplier
    supplier_parts = middle_parts[:value_idx]
    supplier = ' '.join(supplier_parts).strip().upper()
    
    if not supplier:
        return None
    
    return FilenameGroundTruth(
        date=date,
        entity=entity,
        supplier=supplier,
        amount_cents=amount_cents,
        doc_type=doc_type,
        doc_number=doc_number,
        raw_filename=filename
    )


def validate_extraction(result, truth: FilenameGroundTruth) -> dict:
    """Compare extraction result against filename ground truth."""
    validation = {}
    
    # Supplier validation (fuzzy match)
    got_supplier = (result.fornecedor or "").upper()
    validation["supplier"] = {
        "expected": truth.supplier,
        "got": got_supplier or "NONE",
        "match": False
    }
    if got_supplier:
        # Match if first 6 chars match or one contains the other
        exp_words = truth.supplier.split()
        got_words = got_supplier.split()
        if exp_words and got_words:
            if exp_words[0][:5] in got_supplier or got_words[0][:5] in truth.supplier:
                validation["supplier"]["match"] = True
    
    # Amount validation (exact or within 1%)
    validation["amount"] = {
        "expected": truth.amount_cents,
        "got": result.amount_cents,
        "match": False
    }
    if truth.amount_cents > 0 and result.amount_cents > 0:
        diff = abs(truth.amount_cents - result.amount_cents)
        tolerance = max(truth.amount_cents * 0.01, 100)  # 1% or R$1
        if diff <= tolerance:
            validation["amount"]["match"] = True
    
    # Doc type validation
    got_type = result.doc_type.name.upper() if result.doc_type else "UNKNOWN"
    validation["doc_type"] = {
        "expected": truth.doc_type,
        "got": got_type,
        "match": got_type == truth.doc_type or (got_type in ("NFE", "NFSE") and truth.doc_type in ("NFE", "NFSE"))
    }
    
    # Doc number validation (if available in filename)
    if truth.doc_number and truth.doc_number.isdigit():
        got_num = result.doc_number or "NONE"
        validation["doc_number"] = {
            "expected": truth.doc_number,
            "got": got_num,
            "match": got_num.lstrip('0') == truth.doc_number.lstrip('0') if got_num != "NONE" else False
        }
    
    return validation


def run_validation(sample_size: int = 100):
    """Run proper validation using filename as ground truth."""
    
    print("=" * 80)
    print("PROPER VALIDATION v2 - Filename as Ground Truth")
    print("=" * 80)
    print("Format: DD.MM.YYYY_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf")
    print()
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=True, use_vlm=False)
    
    # Parse all files
    all_files = list(NFE_DIR.glob("*.pdf")) + list(NFE_DIR.glob("*.PDF"))
    validatable = []
    
    for f in all_files:
        truth = parse_filename(f.name)
        if truth:
            validatable.append((f, truth))
    
    print(f"Parseable files: {len(validatable)} / {len(all_files)}")
    
    # Test sample
    sample = validatable[:sample_size]
    print(f"Testing: {len(sample)} files\n")
    
    stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
    errors = defaultdict(list)
    
    for i, (pdf_path, truth) in enumerate(sample, 1):
        try:
            result = extractor.extract_from_pdf(str(pdf_path))
            validation = validate_extraction(result, truth)
            
            for field, v in validation.items():
                if v["match"]:
                    stats[field]["correct"] += 1
                else:
                    stats[field]["wrong"] += 1
                    if len(errors[field]) < 3:
                        errors[field].append({
                            "file": pdf_path.name[:45],
                            "expected": v["expected"],
                            "got": v["got"]
                        })
            
            if i % 20 == 0:
                print(f"  {i}/{len(sample)}...")
                
        except Exception as e:
            print(f"  Error: {pdf_path.name[:30]}: {e}")
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for field in ["supplier", "amount", "doc_type", "doc_number"]:
        if field not in stats:
            continue
        s = stats[field]
        total = s["correct"] + s["wrong"]
        acc = 100 * s["correct"] / total if total else 0
        print(f"\n{field.upper()}: {s['correct']}/{total} ({acc:.0f}%)")
        
        if errors[field]:
            for e in errors[field]:
                print(f"  X {e['file']}")
                print(f"      exp: {e['expected']}, got: {e['got']}")
    
    # Overall
    total_fields = sum(s["correct"] + s["wrong"] for s in stats.values())
    total_correct = sum(s["correct"] for s in stats.values())
    print(f"\nOVERALL: {total_correct}/{total_fields} ({100*total_correct/total_fields:.0f}%)")


if __name__ == "__main__":
    run_validation(100)
