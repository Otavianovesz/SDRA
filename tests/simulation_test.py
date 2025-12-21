"""
Realistic Large-Scale Extraction Simulation

Simulates realistic import of random PDF files from both folders,
analyzing extraction accuracy and identifying specific failures to fix.
"""

import os
import sys
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor, DocumentType
from supplier_validator import is_valid_supplier, validate_supplier

# Configuration
NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"
COMPROVANTES_DIR = Path(__file__).parent.parent / "NOVEMBRO_COMPROVANTES_N_CONCILIADO"
TOTAL_SAMPLES = 200  # Test 200 random files


def parse_nfe_filename(filename: str) -> Dict:
    """Parse expected values from NFE/Boleto filename."""
    parts = filename.rsplit('.', 1)[0].split('_')
    if len(parts) >= 5:
        return {
            'date': parts[0] if re.match(r'\d{2}\.\d{2}\.\d{4}', parts[0]) else None,
            'entity': parts[1] if len(parts) > 1 else None,
            'supplier': parts[2] if len(parts) > 2 else None,
            'expected_amount': parts[3] if len(parts) > 3 else None,
            'doc_type': parts[4] if len(parts) > 4 else None,
            'doc_number': parts[5] if len(parts) > 5 else None,
            'source': 'NFE_FOLDER'
        }
    return {'source': 'NFE_FOLDER'}


def parse_comprovante_filename(filename: str) -> Dict:
    """Parse expected values from comprovante filename."""
    base = filename.rsplit('.', 1)[0]
    
    # New format: 2025_12_18_SUPPLIER_BB_PIX_AMOUNT.pdf
    if re.match(r'^20\d{2}_', base):
        parts = base.split('_')
        if len(parts) >= 7:
            # Find amount at end
            amount = None
            type_hint = None
            for i in range(len(parts) - 1, 2, -1):
                if parts[i] in ['PIX', 'TED', 'DOC']:
                    type_hint = parts[i]
                    if i + 1 < len(parts):
                        amount = parts[i + 1]
                    break
            
            # Supplier is between date and type
            supplier_end = len(parts) - 2 if amount else len(parts)
            for i in range(3, supplier_end):
                if parts[i] in ['BB', 'PIX', 'TED', 'DOC', 'GENERICO']:
                    supplier_parts = parts[3:i]
                    break
            else:
                supplier_parts = parts[3:supplier_end-1]
            
            return {
                'date': f"{parts[2]}.{parts[1]}.{parts[0]}",
                'supplier': ' '.join(supplier_parts) if supplier_parts else None,
                'expected_amount': amount,
                'type_hint': type_hint,
                'source': 'COMPROVANTES_FOLDER',
                'format': 'new'
            }
    
    # Old format: SEQ - DDMMYYYY - TYPE - AMOUNT.pdf
    if re.match(r'^\d+ - ', base):
        parts = [p.strip() for p in base.split(' - ')]
        if len(parts) >= 4:
            date_raw = parts[1]
            if len(date_raw) == 8:
                date_fmt = f"{date_raw[:2]}.{date_raw[2:4]}.{date_raw[4:]}"
            else:
                date_fmt = date_raw
            return {
                'sequence': parts[0],
                'date': date_fmt,
                'trans_type': parts[2],
                'expected_amount': parts[3],
                'source': 'COMPROVANTES_FOLDER',
                'format': 'old'
            }
    
    return {'source': 'COMPROVANTES_FOLDER'}


def parse_amount(amount_str: str) -> Optional[int]:
    """Convert amount string to cents."""
    if not amount_str:
        return None
    clean = amount_str.replace('.', '').replace(',', '.')
    try:
        return int(float(clean) * 100)
    except:
        return None


def run_simulation():
    """Run realistic large-scale simulation."""
    print("=" * 80)
    print("REALISTIC LARGE-SCALE EXTRACTION SIMULATION")
    print(f"Testing {TOTAL_SAMPLES} random files from both folders")
    print("=" * 80)
    
    # Gather all PDFs
    nfe_files = list(NFE_DIR.glob("*.pdf"))
    comp_files = [f for f in COMPROVANTES_DIR.glob("*.pdf") if not f.name.endswith('.exe')]
    
    print(f"\nAvailable: {len(nfe_files)} NFE/Boleto, {len(comp_files)} Comprovantes")
    
    # Random sample
    all_files = [(f, 'nfe') for f in nfe_files] + [(f, 'comp') for f in comp_files]
    random.seed(42)  # Reproducible
    sample = random.sample(all_files, min(TOTAL_SAMPLES, len(all_files)))
    
    print(f"Sampling {len(sample)} files randomly...\n")
    
    # Initialize extractor
    extractor = EnsembleExtractor(use_gliner=False, use_ocr=False, use_vlm=False)
    
    # Results tracking
    results = {
        'nfe': {'total': 0, 'supplier_ok': 0, 'amount_ok': 0, 'type_ok': 0, 'failures': []},
        'boleto': {'total': 0, 'supplier_ok': 0, 'amount_ok': 0, 'type_ok': 0, 'failures': []},
        'comp': {'total': 0, 'supplier_ok': 0, 'amount_ok': 0, 'type_ok': 0, 'failures': []}
    }
    supplier_mismatches = []
    amount_mismatches = []
    type_errors = []
    
    for i, (pdf_path, folder_type) in enumerate(sample, 1):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(sample)}...")
        
        # Parse expected values
        if folder_type == 'nfe':
            expected = parse_nfe_filename(pdf_path.name)
            exp_type = expected.get('doc_type', '').upper()
            if 'BOLETO' in exp_type:
                cat = 'boleto'
            else:
                cat = 'nfe'
        else:
            expected = parse_comprovante_filename(pdf_path.name)
            cat = 'comp'
        
        results[cat]['total'] += 1
        
        # Extract
        try:
            result = extractor.extract_from_pdf(str(pdf_path))
        except Exception as e:
            results[cat]['failures'].append((pdf_path.name, str(e)))
            continue
        
        # Check type classification
        extracted_type = result.doc_type.name if result.doc_type else 'UNKNOWN'
        exp_type_check = expected.get('doc_type', expected.get('type_hint', '')).upper()
        
        if exp_type_check:
            if exp_type_check in extracted_type or extracted_type in exp_type_check:
                results[cat]['type_ok'] += 1
            elif cat == 'comp' and 'COMPROVANTE' in extracted_type:
                results[cat]['type_ok'] += 1
            else:
                type_errors.append({
                    'file': pdf_path.name[:50],
                    'expected': exp_type_check,
                    'got': extracted_type
                })
        
        # Check supplier
        exp_supplier = (expected.get('supplier') or '').upper()
        ext_supplier = (result.fornecedor or '').upper()
        
        if exp_supplier and ext_supplier:
            # Fuzzy match: first 10 chars or containment
            if (exp_supplier[:10] in ext_supplier or 
                ext_supplier[:10] in exp_supplier or
                exp_supplier in ext_supplier or 
                ext_supplier in exp_supplier):
                results[cat]['supplier_ok'] += 1
            else:
                supplier_mismatches.append({
                    'file': pdf_path.name[:50],
                    'expected': exp_supplier[:30],
                    'got': ext_supplier[:30],
                    'category': cat
                })
        elif ext_supplier and not exp_supplier:
            # Extracted but no expected - might be OK
            results[cat]['supplier_ok'] += 1
        elif not ext_supplier and exp_supplier:
            # Missing extraction
            supplier_mismatches.append({
                'file': pdf_path.name[:50],
                'expected': exp_supplier[:30],
                'got': 'NONE',
                'category': cat
            })
        
        # Check amount
        exp_amount = parse_amount(expected.get('expected_amount'))
        ext_amount = result.amount_cents or 0
        
        if exp_amount and ext_amount:
            if abs(exp_amount - ext_amount) < 100:  # Within R$ 1.00
                results[cat]['amount_ok'] += 1
            else:
                amount_mismatches.append({
                    'file': pdf_path.name[:50],
                    'expected': exp_amount,
                    'got': ext_amount,
                    'category': cat
                })
        elif ext_amount and not exp_amount:
            results[cat]['amount_ok'] += 1
    
    # Print results
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    
    for cat, data in results.items():
        total = data['total']
        if total == 0:
            continue
        print(f"\n{cat.upper()} ({total} files):")
        print(f"  Supplier: {data['supplier_ok']}/{total} ({100*data['supplier_ok']/total:.1f}%)")
        print(f"  Amount: {data['amount_ok']}/{total} ({100*data['amount_ok']/total:.1f}%)")
        if data['failures']:
            print(f"  Errors: {len(data['failures'])}")
    
    # Show specific failures for debugging
    print("\n" + "-" * 40)
    print("SUPPLIER MISMATCHES (sample):")
    for m in supplier_mismatches[:10]:
        print(f"  [{m['category']}] {m['file']}")
        print(f"    Expected: {m['expected']}, Got: {m['got']}")
    
    print(f"\nTotal supplier mismatches: {len(supplier_mismatches)}")
    
    print("\n" + "-" * 40)
    print("AMOUNT MISMATCHES (sample):")
    for m in amount_mismatches[:5]:
        print(f"  [{m['category']}] {m['file']}")
        print(f"    Expected: {m['expected']} cents, Got: {m['got']} cents")
    
    print(f"\nTotal amount mismatches: {len(amount_mismatches)}")
    
    if type_errors:
        print("\n" + "-" * 40)
        print("TYPE CLASSIFICATION ERRORS:")
        for e in type_errors[:5]:
            print(f"  {e['file']}: expected {e['expected']}, got {e['got']}")
    
    # Summary
    total_files = sum(r['total'] for r in results.values())
    total_supplier_ok = sum(r['supplier_ok'] for r in results.values())
    total_amount_ok = sum(r['amount_ok'] for r in results.values())
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total files tested: {total_files}")
    print(f"Overall supplier accuracy: {total_supplier_ok}/{total_files} ({100*total_supplier_ok/total_files:.1f}%)")
    print(f"Overall amount accuracy: {total_amount_ok}/{total_files} ({100*total_amount_ok/total_files:.1f}%)")
    
    return results, supplier_mismatches, amount_mismatches


if __name__ == "__main__":
    run_simulation()
