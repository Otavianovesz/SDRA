import re
import os

def test_amount_parsing():
    # Example filenames that failed (based on analysis or expectation)
    filenames = [
        "01.11.2025_VG_JUCELIA GONCALVES_2.573,00_2.287,00_NFSE_3159.pdf",
        "03.11.2025_VG_ARAGUAIA AGRICOLA_7.338,20_1.128,32_NFE_110993.pdf", # Failed previously
        "30.11.2025_VG_TECNOSPEED_2.857,58_BOLETO_252472.pdf", # Failed in validation 2
        "28.11.2025_VG_GF AUTO CENTER_309,35_NFE_3717.pdf" # Passed in validation 2
    ]
    
    print("Testing Amount Parsing Logic...")
    for fname in filenames:
        print(f"\nFile: {fname}")
        parts = fname.split('_')
        for p in parts[1:]:
            print(f"  Part: '{p}'")
            if re.search(r'\d', p):
                clean_p = re.sub(r'[^\d\.,]', '', p)
                print(f"    Clean: '{clean_p}'")
                
                if ',' in clean_p:
                    try:
                        norm = clean_p.replace('.', '').replace(',', '.')
                        val = float(norm)
                        print(f"    -> Parsed: {int(val * 100)}")
                    except:
                        print("    -> Parse Error")
                else:
                    print("    -> No comma, skipping")

if __name__ == "__main__":
    test_amount_parsing()
