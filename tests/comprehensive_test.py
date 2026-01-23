"""
Comprehensive extraction test to verify ALL fields are being extracted.
Shows: fornecedor, valor, data, numero_documento for each file.
"""
import sys
from pathlib import Path
from collections import defaultdict
sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor

NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"
COMP_DIR = Path(__file__).parent.parent / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"


def comprehensive_extraction_test(sample_size: int = 30):
    """Test all field extractions across file types."""
    
    print("=" * 80)
    print("COMPREHENSIVE FIELD EXTRACTION TEST")
    print("Checking: Fornecedor, Valor, Data, Numero Documento")
    print("=" * 80)
    
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    # Get sample files from each type
    nfe_files = [f for f in NFE_DIR.glob("*NFE*.pdf")][:sample_size//3]
    boleto_files = [f for f in NFE_DIR.glob("*BOLETO*.pdf")][:sample_size//3]
    comp_files = list(COMP_DIR.glob("*.pdf"))[:sample_size//3]
    
    all_files = nfe_files + boleto_files + comp_files
    
    # Track extraction stats
    stats = {
        "NFE": {"total": 0, "fornecedor": 0, "valor": 0, "data": 0, "doc_number": 0},
        "BOLETO": {"total": 0, "fornecedor": 0, "valor": 0, "data": 0, "doc_number": 0},
        "COMPROVANTE": {"total": 0, "fornecedor": 0, "valor": 0, "data": 0, "doc_number": 0},
        "NFSE": {"total": 0, "fornecedor": 0, "valor": 0, "data": 0, "doc_number": 0},
        "OTHER": {"total": 0, "fornecedor": 0, "valor": 0, "data": 0, "doc_number": 0},
    }
    
    missing_examples = defaultdict(list)
    
    print(f"\nProcessing {len(all_files)} files...\n")
    
    for i, pdf in enumerate(all_files, 1):
        try:
            result = extractor.extract_from_pdf(str(pdf))
            doc_type = result.doc_type.name
            
            if doc_type not in stats:
                doc_type = "OTHER"
            
            stats[doc_type]["total"] += 1
            
            # Check each field
            has_fornecedor = bool(result.fornecedor)
            has_valor = result.amount_cents > 0
            has_data = bool(result.due_date or result.emission_date or result.payment_date)
            has_doc_number = bool(result.doc_number)
            
            if has_fornecedor:
                stats[doc_type]["fornecedor"] += 1
            else:
                missing_examples["fornecedor"].append(pdf.name[:40])
                
            if has_valor:
                stats[doc_type]["valor"] += 1
            else:
                missing_examples["valor"].append(pdf.name[:40])
                
            if has_data:
                stats[doc_type]["data"] += 1
            else:
                missing_examples["data"].append(pdf.name[:40])
                
            if has_doc_number:
                stats[doc_type]["doc_number"] += 1
            else:
                missing_examples["doc_number"].append(pdf.name[:40])
            
            # Progress
            if i % 10 == 0:
                print(f"  Processed {i}/{len(all_files)}...")
                
        except Exception as e:
            print(f"  Error on {pdf.name}: {e}")
    
    # Print results
    print("\n" + "=" * 80)
    print("EXTRACTION RATES BY DOC TYPE")
    print("=" * 80)
    
    for doc_type, s in stats.items():
        if s["total"] == 0:
            continue
        print(f"\n{doc_type} ({s['total']} files):")
        print(f"  Fornecedor: {s['fornecedor']}/{s['total']} ({100*s['fornecedor']/s['total']:.0f}%)")
        print(f"  Valor:      {s['valor']}/{s['total']} ({100*s['valor']/s['total']:.0f}%)")
        print(f"  Data:       {s['data']}/{s['total']} ({100*s['data']/s['total']:.0f}%)")
        print(f"  Doc Number: {s['doc_number']}/{s['total']} ({100*s['doc_number']/s['total']:.0f}%)")
    
    # Show missing examples
    print("\n" + "=" * 80)
    print("MISSING FIELD EXAMPLES (first 5)")
    print("=" * 80)
    
    for field, examples in missing_examples.items():
        if examples:
            print(f"\n{field.upper()} missing in:")
            for ex in examples[:5]:
                print(f"  - {ex}")
    
    # Calculate total
    total = sum(s["total"] for s in stats.values())
    total_fornecedor = sum(s["fornecedor"] for s in stats.values())
    total_valor = sum(s["valor"] for s in stats.values())
    total_data = sum(s["data"] for s in stats.values())
    total_doc = sum(s["doc_number"] for s in stats.values())
    
    print("\n" + "=" * 80)
    print("OVERALL EXTRACTION RATES")
    print("=" * 80)
    print(f"Fornecedor:  {total_fornecedor}/{total} ({100*total_fornecedor/total:.0f}%)")
    print(f"Valor:       {total_valor}/{total} ({100*total_valor/total:.0f}%)")
    print(f"Data:        {total_data}/{total} ({100*total_data/total:.0f}%)")
    print(f"Doc Number:  {total_doc}/{total} ({100*total_doc/total:.0f}%)")


if __name__ == "__main__":
    comprehensive_extraction_test(30)
