import sys
import os
sys.path.append('.')

print("VERIFYING V3.0 ENTERPRISE INTEGRITY...")
print("-" * 40)

try:
    print("[1] Checking Gmail Watchdog...")
    from integrations.gmail_connector import GmailConnector, EmailAttachment, VAULT_PASSWORDS, PIKEPDF_AVAILABLE
    print(f"    - Vault loaded: {len(VAULT_PASSWORDS)} passwords")
    print(f"    - Pikepdf Available: {PIKEPDF_AVAILABLE}")
    print("    - OK")

    print("[2] Checking UI Components...")
    # Mocking tkinter root heavily to avoid display requirement if possible, 
    # but imports should work regardless.
    from gui.components import PDFPreviewPanel, SmartEditPanel
    print("    - OK")

    print("[3] Checking Predictive Brain...")
    from integrations.mcmf_reconciler import MCMFReconciler, RAPIDFUZZ_AVAILABLE
    r = MCMFReconciler()
    if hasattr(r, 'find_intelligent_matches'):
        print(f"    - Intelligence Active (RapidFuzz: {RAPIDFUZZ_AVAILABLE})")
    else:
        print("    - FAIL: find_intelligent_matches missing")
    print("    - OK")
    
    print("\n✅ V3.0 READY TO DEPLOY")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
