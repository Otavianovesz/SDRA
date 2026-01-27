import fitz
from pathlib import Path

file_path = Path(r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547\01.11.2025_VG_ABDALLA TRUCK_370,00_NFSE_20590.pdf")
doc = fitz.open(file_path)
text = doc[0].get_text()
with open("debug_output.txt", "w", encoding="utf-8") as f:
    f.write(text)
doc.close()
print("Escrito para debug_output.txt")
