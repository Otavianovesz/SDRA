"""
SDRA Baseline Benchmark - Measures current extraction performance.

Uses 11.2025_NOVEMBRO_1.547 folder as ground truth (filenames are correct).
Runs multiple iterations to get reliable metrics on CPU/8GB RAM.

Metrics tracked:
- Supplier accuracy (fuzzy match)
- Amount accuracy (exact match within R$1.00)
- Date accuracy (exact match)
- Doc Number accuracy
- Average extraction time per document
"""

import os
import sys
import re
import time
import random
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor, DocumentType


# Configuration
NFE_DIR = Path(__file__).parent.parent / "11.2025_NOVEMBRO_1.547"
SAMPLE_SIZE = 50  # Start with 50 random files for baseline
RANDOM_SEED = 42


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction."""
    filename: str
    expected: Dict
    extracted: Dict
    supplier_match: bool = False
    amount_match: bool = False
    date_match: bool = False
    doc_number_match: bool = False
    doc_type_match: bool = False
    extraction_time_ms: float = 0.0
    error: str = ""


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    total_files: int = 0
    successful_extractions: int = 0
    supplier_correct: int = 0
    amount_correct: int = 0
    date_correct: int = 0
    doc_number_correct: int = 0
    doc_type_correct: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
{'='*70}
BASELINE BENCHMARK RESULTS
{'='*70}
Total Files:      {self.total_files}
Successful:       {self.successful_extractions} ({self.successful_extractions/max(1,self.total_files)*100:.1f}%)
Errors:           {len(self.errors)}

ACCURACY:
  Supplier:       {self.supplier_correct}/{self.total_files} ({self.supplier_correct/max(1,self.total_files)*100:.1f}%)
  Amount:         {self.amount_correct}/{self.total_files} ({self.amount_correct/max(1,self.total_files)*100:.1f}%)
  Date:           {self.date_correct}/{self.total_files} ({self.date_correct/max(1,self.total_files)*100:.1f}%)
  Doc Number:     {self.doc_number_correct}/{self.total_files} ({self.doc_number_correct/max(1,self.total_files)*100:.1f}%)
  Doc Type:       {self.doc_type_correct}/{self.total_files} ({self.doc_type_correct/max(1,self.total_files)*100:.1f}%)

PERFORMANCE:
  Total Time:     {self.total_time_ms/1000:.2f}s
  Avg per file:   {self.avg_time_ms:.0f}ms
{'='*70}
"""


def parse_filename(filename: str) -> Dict:
    """
    Parse ground truth from filename.
    Format: DD.MM.AAAA_ENTIDADE_FORNECEDOR_VALOR_[VALOR2]_[PARC X-Y]_TIPO_NUMERO.pdf
    
    Examples:
    - 01.11.2025_VG_CALDATTO SERVICOS_1.660,50_BOLETO_507.pdf
    - 01.11.2025_VG_AGRO BAGGIO_2.843,29_5.686,58_PARC 2-2_NFE_414978.pdf
    """
    result = {
        "date": None,
        "entity": None,
        "supplier": None,
        "amount_cents": 0,
        "doc_type": None,
        "doc_number": None,
    }
    
    # Remove extension
    name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    parts = name.split("_")
    
    if len(parts) < 5:
        return result
    
    # Date (DD.MM.AAAA)
    date_match = re.match(r'^(\d{2})\.(\d{2})\.(\d{4})$', parts[0])
    if date_match:
        d, m, y = date_match.groups()
        result["date"] = f"{y}-{m}-{d}"  # ISO format
    
    # Entity (VG or MV)
    if parts[1] in ["VG", "MV"]:
        result["entity"] = parts[1]
    
    # Find document type position
    doc_types = ["NFE", "NFSE", "BOLETO", "APOLICE", "COMPROVANTE", "FATURA"]
    type_idx = -1
    for i, part in enumerate(parts):
        if part.upper() in doc_types:
            type_idx = i
            result["doc_type"] = part.upper()
            break
    
    if type_idx == -1:
        return result
    
    # Doc number is after doc type
    doc_numbers = parts[type_idx + 1:]
    if doc_numbers:
        result["doc_number"] = "_".join(doc_numbers)
    
    # Parse supplier and values between entity and doc_type
    value_pattern = re.compile(r'^(\d{1,3}(?:\.\d{3})*,\d{2})$')
    supplier_parts = []
    values_found = []
    
    for i in range(2, type_idx):
        part = parts[i]
        
        # Check if it's a monetary value
        if value_pattern.match(part):
            val = part.replace(".", "").replace(",", ".")
            values_found.append(int(float(val) * 100))
        # Skip parcela indicators
        elif part.upper() == "PARC" or re.match(r'^\d+-\d+$', part):
            continue
        # Skip special terms
        elif part.upper() in ["JUROS", "FECHAMENTO"]:
            continue
        # Otherwise it's part of supplier name
        elif not values_found:
            supplier_parts.append(part)
    
    result["supplier"] = " ".join(supplier_parts) if supplier_parts else None
    
    # First value is the amount we compare against
    if values_found:
        result["amount_cents"] = values_found[0]
    
    return result


def fuzzy_match(s1: str, s2: str, threshold: float = 0.6) -> bool:
    """Check if two strings fuzzy match."""
    if not s1 or not s2:
        return False
    
    s1, s2 = s1.upper().strip(), s2.upper().strip()
    
    # Exact match
    if s1 == s2:
        return True
    
    # Containment
    if s1 in s2 or s2 in s1:
        return True
    
    # Word overlap
    words1 = set(s1.split())
    words2 = set(s2.split())
    if words1 and words2:
        common = words1 & words2
        if len(common) / min(len(words1), len(words2)) >= threshold:
            return True
    
    # First N chars match
    if len(s1) >= 5 and len(s2) >= 5:
        if s1[:5] == s2[:5]:
            return True
    
    return False


def run_benchmark(
    use_gliner: bool = False,
    use_ocr: bool = False,
    use_vlm: bool = False,
    sample_size: int = SAMPLE_SIZE,
    save_csv: bool = True
) -> BenchmarkResults:
    """
    Run extraction benchmark on sample files.
    
    Args:
        use_gliner: Enable GLiNER NER
        use_ocr: Enable OCR fallback
        use_vlm: Enable VLM (slow, use for testing only)
        sample_size: Number of files to test
        save_csv: Save detailed results to CSV
    """
    print(f"\n{'='*70}")
    print(f"SDRA BASELINE BENCHMARK")
    print(f"{'='*70}")
    print(f"Config: GLiNER={use_gliner}, OCR={use_ocr}, VLM={use_vlm}")
    print(f"Sample: {sample_size} files from {NFE_DIR.name}")
    print(f"{'='*70}\n")
    
    # Get all PDFs
    all_pdfs = list(NFE_DIR.glob("*.pdf"))
    if not all_pdfs:
        print(f"ERROR: No PDFs found in {NFE_DIR}")
        return BenchmarkResults()
    
    print(f"Found {len(all_pdfs)} total PDFs")
    
    # Random sample
    random.seed(RANDOM_SEED)
    sample = random.sample(all_pdfs, min(sample_size, len(all_pdfs)))
    
    # Initialize extractor
    print("Loading extractor...")
    start_load = time.time()
    extractor = EnsembleExtractor(
        use_gliner=use_gliner,
        use_ocr=use_ocr,
        use_vlm=use_vlm,
        lazy_load=True
    )
    print(f"Extractor loaded in {time.time()-start_load:.2f}s\n")
    
    # Results
    results = BenchmarkResults()
    results.total_files = len(sample)
    metrics_list: List[ExtractionMetrics] = []
    
    # Process files
    for i, pdf_path in enumerate(sample, 1):
        filename = pdf_path.name
        expected = parse_filename(filename)
        
        # Skip unparseable files
        if not expected.get("doc_type"):
            print(f"[{i}/{len(sample)}] SKIP: Cannot parse: {filename[:40]}...")
            continue
        
        metrics = ExtractionMetrics(filename=filename, expected=expected, extracted={})
        
        # Extract
        start_time = time.time()
        try:
            result = extractor.extract_from_pdf(str(pdf_path))
            elapsed_ms = (time.time() - start_time) * 1000
            
            metrics.extraction_time_ms = elapsed_ms
            results.total_time_ms += elapsed_ms
            results.successful_extractions += 1
            
            # Store extracted values
            primary_date = result.due_date or result.payment_date or result.emission_date
            metrics.extracted = {
                "supplier": result.fornecedor,
                "amount_cents": result.amount_cents,
                "date": primary_date,
                "doc_type": result.doc_type.name if result.doc_type else "UNKNOWN",
                "doc_number": result.doc_number,
            }
            
            # Compare: Supplier (fuzzy)
            if fuzzy_match(expected.get("supplier"), result.fornecedor):
                metrics.supplier_match = True
                results.supplier_correct += 1
            
            # Compare: Amount (within R$1.00)
            exp_amt = expected.get("amount_cents", 0) or 0
            ext_amt = result.amount_cents or 0
            if abs(exp_amt - ext_amt) <= 100:
                metrics.amount_match = True
                results.amount_correct += 1
            
            # Compare: Date (exact)
            if expected.get("date") == primary_date:
                metrics.date_match = True
                results.date_correct += 1
            
            # Compare: Doc Type
            exp_type = expected.get("doc_type", "").upper()
            ext_type = result.doc_type.name if result.doc_type else "UNKNOWN"
            if exp_type == ext_type:
                metrics.doc_type_match = True
                results.doc_type_correct += 1
            
            # Compare: Doc Number
            exp_num = expected.get("doc_number", "") or ""
            ext_num = str(result.doc_number or "")
            if exp_num and ext_num:
                exp_clean = exp_num.lstrip('0') or '0'
                ext_clean = ext_num.lstrip('0') or '0'
                if exp_clean == ext_clean or exp_clean in ext_clean or ext_clean in exp_clean:
                    metrics.doc_number_match = True
                    results.doc_number_correct += 1
            elif not exp_num:
                results.doc_number_correct += 1  # No expected
                metrics.doc_number_match = True
            
            # Progress
            status = "OK" if all([metrics.supplier_match, metrics.amount_match, metrics.date_match]) else "PARTIAL"
            print(f"[{i}/{len(sample)}] {status} ({elapsed_ms:.0f}ms) {filename[:40]}...")
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            metrics.error = str(e)
            results.errors.append(f"{filename}: {e}")
            print(f"[{i}/{len(sample)}] ERROR: {filename[:40]}... - {e}")
        
        metrics_list.append(metrics)
    
    # Calculate averages
    results.avg_time_ms = results.total_time_ms / max(1, results.successful_extractions)
    
    # Save to CSV
    if save_csv:
        csv_path = Path(__file__).parent.parent / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'supplier_match', 'amount_match', 'date_match', 
                'docnum_match', 'type_match', 'time_ms',
                'exp_supplier', 'ext_supplier', 'exp_amount', 'ext_amount',
                'exp_date', 'ext_date', 'error'
            ])
            for m in metrics_list:
                writer.writerow([
                    m.filename,
                    m.supplier_match,
                    m.amount_match,
                    m.date_match,
                    m.doc_number_match,
                    m.doc_type_match,
                    f"{m.extraction_time_ms:.0f}",
                    m.expected.get('supplier', ''),
                    m.extracted.get('supplier', ''),
                    m.expected.get('amount_cents', 0),
                    m.extracted.get('amount_cents', 0),
                    m.expected.get('date', ''),
                    m.extracted.get('date', ''),
                    m.error
                ])
        print(f"\nResults saved to: {csv_path}")
    
    # Print summary
    print(results.summary())
    
    # Show sample failures
    failures = [m for m in metrics_list if not all([m.supplier_match, m.amount_match, m.date_match]) and not m.error]
    if failures:
        print("\n--- SAMPLE FAILURES (first 5) ---")
        for m in failures[:5]:
            print(f"\n{m.filename[:50]}...")
            if not m.supplier_match:
                print(f"  SUPPLIER: exp='{m.expected.get('supplier', '')}' got='{m.extracted.get('supplier', '')}'")
            if not m.amount_match:
                exp_amt = m.expected.get('amount_cents', 0) / 100
                ext_amt = (m.extracted.get('amount_cents') or 0) / 100
                print(f"  AMOUNT: exp={exp_amt:.2f} got={ext_amt:.2f}")
            if not m.date_match:
                print(f"  DATE: exp='{m.expected.get('date', '')}' got='{m.extracted.get('date', '')}'")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SDRA Baseline Benchmark")
    parser.add_argument("--gliner", action="store_true", help="Enable GLiNER")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM (slow)")
    parser.add_argument("--sample", type=int, default=50, help="Sample size")
    
    args = parser.parse_args()
    
    run_benchmark(
        use_gliner=args.gliner,
        use_ocr=args.ocr,
        use_vlm=args.vlm,
        sample_size=args.sample
    )
