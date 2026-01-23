
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Importing database...")
    from database import SRDADatabase
    db = SRDADatabase("srda_verify.db")
    print("Database initialized.")
    stats = db.get_statistics()
    print(f"Stats: {stats}")
    
    print("Importing MCMFReconciler...")
    from mcmf_reconciler import MCMFReconciler, DocumentNode
    rec = MCMFReconciler(db)
    print("Reconciler initialized.")
    
    print("Importing main application...")
    # We won't run main() as it requires GUI, just import to check syntax
    import main
    print("Main imported.")
    
    print("VERIFICATION SUCCESS")
    
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
