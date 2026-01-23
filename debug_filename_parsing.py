import re

def test_parsing():
    filenames = [
        "30.11.2025_VG_TECNOSPEED_2.857,58_BOLETO_252472.pdf",
        "01.11.2025_VG_JUCELIA GONCALVES_2.573,00_2.287,00_NFSE_3159.pdf",
        "30.11.2025_VG_CADORE BIDOIA_1.200,00_NFSE_202521.pdf",
        "30.11.2025_VG_M.V. PECAS_100,00.pdf",
        "30.11.2025_VG_AGRO-PECUARIA_100.pdf" 
    ]
    
    print("Testing Supplier Parsing Logic...")
    for filename in filenames:
        print(f"\nFile: {filename}")
        # Current Regex
        match = re.match(r'^\d{2}\.\d{2}\.\d{4}_(?:[A-Z]{2})_([A-Z][A-Z\s]+?)_', filename)
        if match:
            print(f"  -> Strict Regex: '{match.group(1)}'")
        else:
            print("  -> Strict Regex: NO MATCH")
            
        # Proposed Regex (Lazy split)
        parts = filename.split('_')
        if len(parts) >= 3:
             print(f"  -> Split Logic: '{parts[2]}'")

if __name__ == "__main__":
    test_parsing()
