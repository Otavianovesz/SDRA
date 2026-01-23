
import os
import sys
import re
import fitz
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Files that failed in previous runs (from memory/logs)
FAILED_FILES = [
    "28.11.2025_VG_ENERGISA MATO GROSSO_124,08_FATURA_646898532.pdf",
    "05.11.2025_VG_MARCIA LUCIO MARTINS_2.286,00_NFSE_22.pdf",
    "03.11.2025_VG_COELHOS TORNEARIA_250,00_BOLETO_4252.pdf",
    "04.11.2025_VG_DEL MORO_28,48_BOLETO_85089.pdf",
    "10.11.2025_VG_M. P. S. ALCANTARA_2.275,00_PARC 2-2_BOLETO_4067.pdf",
    "07.11.2025_VG_OESTE VEICULOS_4.065,44_BOLETO_33628_65831.pdf"
]

BASE_DIR = r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547"

def analyze_failures():
    print("="*60)
    print("FAILURE ANALYSIS - RAW TEXT DUMP")
    print("="*60)
    
    for fname in FAILED_FILES:
        path = Path(BASE_DIR) / fname
        if not path.exists():
            print(f"Skipping {fname} (not found)")
            continue
            
        print(f"\n>>> FILE: {fname}")
        try:
            doc = fitz.open(path)
            text = doc[0].get_text()
            print("--- BEGIN TEXT ---")
            print(text[:2000]) # First 2000 chars usually contain everything important
            print("--- END TEXT ---")
            
            # Specific checks
            print("\n[ANALYSIS]")
            
            # Check Supplier
            if "MARCIA LUCIO" in fname:
                match = re.search(r'MARCIA\s+LUCIO', text, re.IGNORECASE)
                print(f"Found Supplier Regex? {match is not None}")
                
            # Check Amount
            if "124,08" in fname:
                match = re.search(r'124,08', text)
                print(f"Found Amount 124,08? {match is not None}")
                
            doc.close()
        except Exception as e:
            print(f"Error reading: {e}")

if __name__ == "__main__":
    analyze_failures()
