"""
Full Dataset Test - SDRA Extraction
"""
import sys
sys.path.insert(0, r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2")

from iterative_precision_test import DetailedTestHarness

test_dir = r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547"

harness = DetailedTestHarness(test_dir)

# Run comprehensive test - limit 500 for full coverage
results = harness.evaluate_all(limit=500, randomize=True)
harness.print_detailed_report(results)

# Save failures for analysis
harness.save_failures_csv("precision_failures.csv")
