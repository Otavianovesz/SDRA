
import os
import fitz
from spatial_extractor import get_spatial_extractor

# Define file path
pdf_path = r"C:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547\01.11.2025_VG_CALDATTO SERVICOS_47,60_BOLETO_510.pdf"

print(f"Testing Overlay logic on: {pdf_path}")

try:
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        exit(1)

    spatial = get_spatial_extractor(pdf_path)
    if not spatial:
        print("Error: Could not initialize SpatialExtractor")
        exit(1)

    # Simulate Logic for Boleto
    doc_type = "BOLETO"
    anchor_text = "VALOR DO DOCUMENTO"
    
    print(f"Searching for anchor: '{anchor_text}'")
    anchor = spatial.find_anchor(anchor_text)
    
    if anchor:
        print(f"FOUND Anchor at: {anchor}")
        # ROI Logic for Boleto
        roi = [anchor.x0 - 5, anchor.y1, anchor.x0 + 150, anchor.y1 + 40]
        print(f"Calculated ROI: {roi}")
        
        # Verify text in ROI
        val = spatial.extract_value_below(anchor_text)
        print(f"Extracted Value in ROI: '{val}'")
    else:
        print("Anchor NOT FOUND. Trying fallback patterns...")
        # Debug: Print all words on page 1? No, too verbose.
        # Try finding 'VALOR' generic
        anchor_gen = spatial.find_anchor("VALOR")
        if anchor_gen:
             print(f"Found generic 'VALOR' at {anchor_gen}")

except Exception as e:
    print(f"Exception: {e}")
