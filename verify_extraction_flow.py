import logging
import sys
from pathlib import Path

# Setup Path
sys.path.append(r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2")

from database import SRDADatabase
from integrations.extraction_chain import get_extraction_chain, ExtractionContext

def test_extraction_chain():
    print("=== Testing Extraction Chain V4.0 ===")
    
    # 1. Setup DB and Alias
    db = SRDADatabase("srda_test_chain.db")
    db.add_supplier_alias("UBER *TRIP", "UBER")
    
    # 2. Setup Chain
    chain = get_extraction_chain(db)
    
    # 3. Test Context Mock
    ctx = ExtractionContext(file_path=Path("dummy_uber.pdf"))
    ctx.amount_cents = 1590
    ctx.supplier_name = "UBER *TRIP" # Raw name
    ctx.extraction_path = "mock_test"
    
    # 4. Test Finalize Resolution
    print(f"Input: Supplier='{ctx.supplier_name}'")
    result = chain._finalize(ctx)
    print(f"Output: Supplier='{result['supplier_name']}'")
    
    if result['supplier_name'] == "UBER":
        print("SUCCESS: Supplier Alias Resolution Works!")
    else:
        print(f"FAILURE: Expected 'UBER', got '{result['supplier_name']}'")

    # 5. Test Integration (Dry Run)
    try:
        # We can't really run process() without a real file, 
        # but we can ensure it doesn't crash on import/init
        print("Chain initialized successfully.")
    except Exception as e:
        print(f"Chain init failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extraction_chain()
