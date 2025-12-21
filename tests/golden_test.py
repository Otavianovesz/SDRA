"""
Golden Dataset Extraction Test

Tests extraction against manually verified ground truth.
Prioritizes ZERO FALSE POSITIVES for monetary values.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

from ensemble_extractor import EnsembleExtractor


@dataclass
class ConfusionMetrics:
    """Confusion matrix metrics for a single field."""
    true_positive: int = 0   # Correct extraction
    false_positive: int = 0  # Wrong value extracted (CRITICAL)
    false_negative: int = 0  # No value when one expected (acceptable)
    true_negative: int = 0   # Correctly empty


def load_golden_dataset() -> Dict:
    """Load the verified golden dataset."""
    golden_path = Path(__file__).parent / "golden_dataset.json"
    if not golden_path.exists():
        raise FileNotFoundError(
            "golden_dataset.json not found! Run bootstrap_golden_dataset.py first."
        )
    
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_golden_test(only_verified: bool = True) -> Dict:
    """
    Run extraction test against golden dataset.
    
    Args:
        only_verified: If True, only test documents marked as verified
        
    Returns:
        Dictionary with test results and confusion matrices
    """
    print("=" * 80)
    print("GOLDEN DATASET EXTRACTION TEST")
    print("Priority: ZERO FALSE POSITIVES")
    print("=" * 80)
    
    # Load dataset
    dataset = load_golden_dataset()
    documents = dataset.get("documents", [])
    
    if only_verified:
        verified = [d for d in documents if d.get("meta", {}).get("is_verified", False)]
        if not verified:
            print("\nNo verified documents found!")
            print("Run bootstrap_golden_dataset.py and verify at least 10 documents.")
            print("\nRunning test on ALL documents as fallback...\n")
            verified = documents
        documents = verified
    
    print(f"Testing {len(documents)} documents\n")
    
    # Initialize
    extractor = EnsembleExtractor(use_gliner=False, use_vlm=False, use_ocr=False)
    
    metrics = {
        "fornecedor": ConfusionMetrics(),
        "valor": ConfusionMetrics(),
        "doc_type": ConfusionMetrics(),
        "doc_number": ConfusionMetrics()
    }
    
    false_positives = []  # Track critical errors
    results = []
    
    for i, doc in enumerate(documents, 1):
        filename = doc.get("file", "unknown")
        folder = doc.get("folder", "")
        truth = doc.get("ground_truth", {})
        
        # Skip documents with errors in bootstrap
        if "error" in doc:
            continue
        
        # Find the PDF
        pdf_path = Path(__file__).parent.parent / folder / filename
        if not pdf_path.exists():
            print(f"  [{i}] SKIP: File not found - {filename}")
            continue
        
        print(f"  [{i}/{len(documents)}] {filename[:40]}...", end=" ")
        
        try:
            result = extractor.extract_from_pdf(str(pdf_path))
            
            # Compare fields
            entry = {"file": filename}
            
            # 1. VALOR (CRITICAL - No False Positives allowed)
            expected_valor = truth.get("valor_centavos", 0)
            extracted_valor = result.amount_cents or 0
            
            if expected_valor > 0 and extracted_valor > 0:
                # Both have values - check if they match
                if abs(expected_valor - extracted_valor) <= 100:  # Within R$ 1.00
                    metrics["valor"].true_positive += 1
                else:
                    metrics["valor"].false_positive += 1
                    false_positives.append({
                        "file": filename,
                        "field": "valor",
                        "expected": expected_valor,
                        "got": extracted_valor,
                        "severity": "CRITICAL"
                    })
            elif expected_valor == 0 and extracted_valor == 0:
                metrics["valor"].true_negative += 1
            elif expected_valor > 0 and extracted_valor == 0:
                metrics["valor"].false_negative += 1  # Acceptable
            elif expected_valor == 0 and extracted_valor > 0:
                metrics["valor"].false_positive += 1  # Invented a value!
                false_positives.append({
                    "file": filename,
                    "field": "valor",
                    "expected": 0,
                    "got": extracted_valor,
                    "severity": "CRITICAL"
                })
            
            # 2. FORNECEDOR
            expected_forn = (truth.get("fornecedor") or "").upper()
            extracted_forn = (result.fornecedor or "").upper()
            
            if expected_forn and extracted_forn:
                if expected_forn[:10] in extracted_forn or extracted_forn[:10] in expected_forn:
                    metrics["fornecedor"].true_positive += 1
                else:
                    metrics["fornecedor"].false_positive += 1
            elif expected_forn and not extracted_forn:
                metrics["fornecedor"].false_negative += 1
            elif not expected_forn and not extracted_forn:
                metrics["fornecedor"].true_negative += 1
            
            # 3. DOC_TYPE
            expected_type = truth.get("doc_type", "UNKNOWN")
            extracted_type = result.doc_type.name if result.doc_type else "UNKNOWN"
            
            if expected_type == extracted_type:
                metrics["doc_type"].true_positive += 1
            else:
                metrics["doc_type"].false_positive += 1
            
            entry["status"] = "OK"
            print("OK")
            
        except Exception as e:
            entry["status"] = f"ERROR: {e}"
            print(f"ERROR: {e}")
        
        results.append(entry)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 80)
    
    for field, m in metrics.items():
        total = m.true_positive + m.false_positive + m.false_negative + m.true_negative
        if total == 0:
            continue
        
        precision = m.true_positive / max(1, m.true_positive + m.false_positive)
        recall = m.true_positive / max(1, m.true_positive + m.false_negative)
        
        print(f"\n{field.upper()}:")
        print(f"  True Positive:  {m.true_positive}")
        print(f"  FALSE POSITIVE: {m.false_positive} {'<-- CRITICAL!' if m.false_positive > 0 else ''}")
        print(f"  False Negative: {m.false_negative}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
    
    # Critical errors report
    if false_positives:
        print("\n" + "=" * 80)
        print("CRITICAL FALSE POSITIVES")
        print("=" * 80)
        for fp in false_positives[:10]:
            print(f"  {fp['file']}")
            print(f"    {fp['field']}: Expected {fp['expected']}, Got {fp['got']}")
    
    print("\n" + "=" * 80)
    valor_fp = metrics["valor"].false_positive
    if valor_fp == 0:
        print("SUCCESS: ZERO FALSE POSITIVES on monetary values!")
    else:
        print(f"FAILURE: {valor_fp} FALSE POSITIVES on monetary values")
    print("=" * 80)
    
    return {
        "metrics": {k: vars(v) for k, v in metrics.items()},
        "false_positives": false_positives,
        "total_tested": len(results)
    }


if __name__ == "__main__":
    run_golden_test(only_verified=False)  # Test all until we have verified docs
