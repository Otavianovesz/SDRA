import sys
import os
import traceback

print("=== STARTUP VERIFICATION v2 ===")
print(f"CWD: {os.getcwd()}")
print(f"Python: {sys.version}")

try:
    print("\n1. Importing standard libraries...")
    import tkinter
    import logging
    print("   OK")

    print("\n2. Importing 3rd party dependencies...")
    try:
        import duckdb
        print("   duckdb: OK")
    except ImportError as e:
        print(f"   duckdb: FAILED ({e})")

    try:
        import ttkbootstrap
        # Verification of fixed import
        from ttkbootstrap.widgets.scrolled import ScrolledFrame
        print("   ttkbootstrap (ScrolledFrame fixed): OK")
    except ImportError as e:
        print(f"   ttkbootstrap: FAILED ({e})")
    except Exception as e:
        print(f"   ttkbootstrap import error: {e}")

    print("\n3. Importing local modules...")
    try:
        import database
        print("   database.py: OK")
        import main
        print("   main.py: OK")
    except ImportError as e:
        print(f"   Local modules: FAILED ({e})")
        traceback.print_exc()

    print("\n=== VERIFICATION COMPLETE ===")
except Exception as e:
    traceback.print_exc()
