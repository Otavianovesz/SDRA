"""
Iterative Precision Testing for SDRA-Rural Extraction
======================================================
This script performs comprehensive testing and outputs detailed error analysis.
"""

import os
import re
import sys
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import asdict

sys.path.append(os.getcwd())

from ensemble_extractor import EnsembleExtractor, DocumentType

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("PrecisionTest")

class DetailedTestHarness:
    """Enhanced test harness with detailed error categorization."""
    
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.extractor = EnsembleExtractor()
        self.files = list(self.test_dir.glob("*.pdf")) + list(self.test_dir.glob("*.PDF"))
        self.error_categories = defaultdict(list)
        self.results_log = []
        
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse ground truth from filename."""
        name = Path(filename).stem
        parts = name.split('_')
        
        gt = {
            "date": None,
            "entity": None,
            "supplier": None,
            "amounts": [],
            "doc_type": None,
            "doc_number": None,
            "installment": None,
            "raw_filename": name
        }
        
        if len(parts) < 4:
            return gt
            
        # 1. Date (first part)
        date_str = parts[0]
        try:
            d, m, y = date_str.split('.')
            gt["date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        except:
            pass
            
        # 2. Entity (second part)
        gt["entity"] = parts[1]
        
        # 3. Supplier (third part, may contain spaces replaced with nothing)
        gt["supplier"] = parts[2]
        
        # Find doc type from end
        type_idx = -1
        known_types = ["NFE", "NFSE", "BOLETO", "FATURA", "COMPROVANTE", 
                       "APOLICE", "DAR", "CTE", "CONTRATO", "CC"]
        
        for i in range(len(parts)-1, 2, -1):
            part_upper = parts[i].upper()
            for t in known_types:
                if part_upper.startswith(t):
                    gt["doc_type"] = t
                    type_idx = i
                    break
            if type_idx != -1:
                break
        
        if type_idx != -1:
            # Doc numbers after type
            numbers = parts[type_idx+1:]
            if numbers:
                # Join multiple numbers (e.g., 123_456 -> 123_456)
                gt["doc_number"] = "_".join(numbers).replace(".pdf", "").replace(".PDF", "")
                
            # Amounts between supplier and type
            middle_parts = parts[3:type_idx]
            
            for p in middle_parts:
                # Match BR money format: X.XXX,XX or XXXX,XX
                if re.match(r'^[\d\.]+,?\d*$', p):
                    try:
                        clean = p.replace('.', '').replace(',', '.')
                        val = float(clean)
                        gt["amounts"].append(int(val * 100))
                    except:
                        pass
                elif "PARC" in p.upper():
                    gt["installment"] = p
                    
        return gt
    
    def normalize_supplier(self, s: str) -> set:
        """Normalize supplier name for comparison."""
        if not s:
            return set()
        stopwords = {"DA", "DE", "DO", "DAS", "DOS", "LTDA", "ME", "EPP", 
                     "EIRELI", "SA", "S.A", "COMERCIO", "E", "CIA", "LT", "SERVICOS"}
        tokens = set(re.findall(r'[A-Z0-9]+', s.upper()))
        return tokens - stopwords
    
    def compare_supplier(self, extracted: Optional[str], gt: str) -> Tuple[bool, float, str]:
        """Compare supplier names, return (match, score, reason)."""
        if not gt:
            return True, 1.0, "no_gt"
        
        if not extracted:
            return False, 0.0, "missing_extraction"
        
        gt_tokens = self.normalize_supplier(gt)
        ex_tokens = self.normalize_supplier(extracted)
        
        if not gt_tokens:
            return True, 1.0, "empty_gt_tokens"
        
        intersection = gt_tokens.intersection(ex_tokens)
        score = len(intersection) / len(gt_tokens)
        
        if score >= 0.5:
            return True, score, "match"
        else:
            return False, score, f"low_overlap_{score:.2f}"
    
    def compare_amount(self, extracted: Optional[int], gt_amounts: List[int]) -> Tuple[bool, str]:
        """Compare amounts with tolerance."""
        if not gt_amounts:
            return True, "no_gt"
        
        if not extracted:
            return False, f"missing_vs_{gt_amounts}"
        
        for gt_amt in gt_amounts:
            if abs(extracted - gt_amt) <= 10:  # 10 cents tolerance
                return True, "match"
        
        return False, f"mismatch_{extracted}_vs_{gt_amounts}"
    
    def compare_date(self, res, gt_date: Optional[str]) -> Tuple[bool, str]:
        """Compare dates."""
        if not gt_date:
            return True, "no_gt"
        
        ex_date = res.due_date or res.emission_date or res.payment_date
        if not ex_date:
            return False, f"missing_vs_{gt_date}"
        
        if ex_date == gt_date:
            return True, "match"
        
        # Check all extracted dates
        for d in [res.due_date, res.emission_date, res.payment_date]:
            if d == gt_date:
                return True, "alt_date_match"
        
        return False, f"mismatch_{ex_date}_vs_{gt_date}"
    
    def evaluate_all(self, limit: int = 100, randomize: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        files_to_test = self.files
        if limit and limit < len(files_to_test) and randomize:
            files_to_test = random.sample(self.files, limit)
        elif limit:
            files_to_test = files_to_test[:limit]
            
        print(f"Testing {len(files_to_test)} PDFs from {len(self.files)} total...")
        print("=" * 70)
        
        stats = {
            "total": 0,
            "success": 0,
            "supplier_correct": 0,
            "amount_correct": 0,
            "date_correct": 0,
            "docnum_correct": 0,
        }
        
        failures = {
            "supplier": [],
            "amount": [],
            "date": [],
            "docnum": [],
            "extraction_error": []
        }
        
        for f in files_to_test:
            stats["total"] += 1
            fname = f.name
            gt = self.parse_filename(fname)
            
            try:
                res = self.extractor.extract_from_pdf(str(f))
            except Exception as e:
                failures["extraction_error"].append({
                    "file": fname,
                    "error": str(e)
                })
                continue
            
            all_ok = True
            
            # Supplier Check
            supp_ok, supp_score, supp_reason = self.compare_supplier(res.fornecedor, gt["supplier"])
            if supp_ok:
                stats["supplier_correct"] += 1
            else:
                all_ok = False
                failures["supplier"].append({
                    "file": fname,
                    "extracted": res.fornecedor,
                    "expected": gt["supplier"],
                    "score": supp_score,
                    "reason": supp_reason
                })
            
            # Amount Check
            amt_ok, amt_reason = self.compare_amount(res.amount_cents, gt["amounts"])
            if amt_ok:
                stats["amount_correct"] += 1
            else:
                all_ok = False
                failures["amount"].append({
                    "file": fname,
                    "extracted": res.amount_cents,
                    "expected": gt["amounts"],
                    "reason": amt_reason
                })
            
            # Date Check
            date_ok, date_reason = self.compare_date(res, gt["date"])
            if date_ok:
                stats["date_correct"] += 1
            else:
                all_ok = False
                failures["date"].append({
                    "file": fname,
                    "extracted_due": res.due_date,
                    "extracted_emission": res.emission_date,
                    "expected": gt["date"],
                    "reason": date_reason
                })
            
            # Doc Number Check (fuzzy)
            if gt["doc_number"]:
                ex_num = str(res.doc_number or "")
                gt_num = gt["doc_number"]
                # Simple check: primary number should be in extracted
                primary = gt_num.split("_")[0]
                if primary in ex_num or ex_num == primary:
                    stats["docnum_correct"] += 1
                else:
                    all_ok = False
                    failures["docnum"].append({
                        "file": fname,
                        "extracted": res.doc_number,
                        "expected": gt["doc_number"]
                    })
            
            if all_ok:
                stats["success"] += 1
            
            self.results_log.append({
                "file": fname,
                "gt": gt,
                "extracted": {
                    "supplier": res.fornecedor,
                    "amount": res.amount_cents,
                    "due_date": res.due_date,
                    "doc_number": res.doc_number,
                    "doc_type": str(res.doc_type)
                },
                "all_ok": all_ok
            })
        
        # Calculate percentages
        total = stats["total"]
        percentages = {
            "overall": (stats["success"] / total * 100) if total else 0,
            "supplier": (stats["supplier_correct"] / total * 100) if total else 0,
            "amount": (stats["amount_correct"] / total * 100) if total else 0,
            "date": (stats["date_correct"] / total * 100) if total else 0,
        }
        
        return {
            "stats": stats,
            "percentages": percentages,
            "failures": failures
        }
    
    def print_detailed_report(self, results: Dict[str, Any]):
        """Print detailed failure analysis."""
        stats = results["stats"]
        pct = results["percentages"]
        failures = results["failures"]
        
        print("\n" + "=" * 70)
        print("PRECISION REPORT")
        print("=" * 70)
        print(f"Total Files Tested: {stats['total']}")
        print(f"All Fields Correct: {stats['success']} ({pct['overall']:.1f}%)")
        print("-" * 70)
        print(f"Supplier Accuracy:  {stats['supplier_correct']}/{stats['total']} ({pct['supplier']:.1f}%)")
        print(f"Amount Accuracy:    {stats['amount_correct']}/{stats['total']} ({pct['amount']:.1f}%)")
        print(f"Date Accuracy:      {stats['date_correct']}/{stats['total']} ({pct['date']:.1f}%)")
        print("=" * 70)
        
        # Supplier Failures
        if failures["supplier"]:
            print(f"\n### SUPPLIER FAILURES ({len(failures['supplier'])})")
            for f in failures["supplier"][:10]:
                print(f"  {f['file']}")
                print(f"    Expected: {f['expected']}")
                print(f"    Got:      {f['extracted']}")
        
        # Amount Failures
        if failures["amount"]:
            print(f"\n### AMOUNT FAILURES ({len(failures['amount'])})")
            for f in failures["amount"][:10]:
                print(f"  {f['file']}")
                print(f"    Expected: {f['expected']} cents")
                print(f"    Got:      {f['extracted']} cents")
        
        # Date Failures
        if failures["date"]:
            print(f"\n### DATE FAILURES ({len(failures['date'])})")
            for f in failures["date"][:10]:
                print(f"  {f['file']}")
                print(f"    Expected: {f['expected']}")
                print(f"    Got Due:  {f['extracted_due']}, Emission: {f['extracted_emission']}")
        
        # Doc Number Failures
        if failures["docnum"]:
            print(f"\n### DOC NUMBER FAILURES ({len(failures['docnum'])})")
            for f in failures["docnum"][:10]:
                print(f"  {f['file']}")
                print(f"    Expected: {f['expected']}")
                print(f"    Got:      {f['extracted']}")
        
        print("\n" + "=" * 70)
        
        return results
    
    def save_failures_csv(self, results: Dict, output_path: str):
        """Save failures to CSV for analysis."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "File", "Expected", "Extracted", "Reason"])
            
            for cat, items in results["failures"].items():
                for item in items:
                    expected = item.get("expected", "")
                    extracted = item.get("extracted", "")
                    reason = item.get("reason", "")
                    writer.writerow([cat, item["file"], expected, extracted, reason])
        
        print(f"Failures saved to: {output_path}")


if __name__ == "__main__":
    test_dir = r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547"
    
    harness = DetailedTestHarness(test_dir)
    
    # Run comprehensive test
    results = harness.evaluate_all(limit=100, randomize=True)
    harness.print_detailed_report(results)
    
    # Save failures for analysis
    harness.save_failures_csv(results, "precision_failures.csv")
