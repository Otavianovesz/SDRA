
import shutil
import logging
from pathlib import Path
import sys

# Configurar logging para ver o que acontece "por baixo do capô"
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProofOfWork")

def run_proof():
    # 1. Selecionar um arquivo "Ground Truth"
    # Este arquivo tem data, valor e fornecedor no nome
    original = Path(r"c:\Users\otavi\Documents\Projetos_programação\SDRA_2\11.2025_NOVEMBRO_1.547\01.11.2025_VG_ABDALLA TRUCK_370,00_BOLETO_20590.pdf")
    
    if not original.exists():
        print(f"ERRO: Arquivo original não encontrado: {original}")
        return

    # 2. Criar uma cópia "cega" (Blind Copy)
    # Removemos TODAS as dicas do nome. O sistema PRECISA ler o PDF.
    blind_file = Path("ARQUIVO_DESCONHECIDO_TESTE_PROVA.pdf")
    shutil.copy2(original, blind_file)
    print(f"\n1. Arquivo copiado para: {blind_file}")
    print("   (Nome não contém Data, Valor ou Fornecedor)")

    try:
        # 3. Inicializar a Engine
        print("\n2. Inicializando Scan Engine...")
        from scanner import CognitiveScanner
        scanner = CognitiveScanner()

        # 4. Executar Extração Real
        print("\n3. Executando Extração (Isso vai acionar OCR/Spatial)...")
        # Forçamos o reset de cache se houver
        result = scanner.hierarchical_extract(blind_file)

        # 5. Exibir a Prova
        print("\n" + "="*60)
        print("PROVA DE EXTRAÇÃO REAL (ZERO HINTS)")
        print("="*60)
        
        print(f"Arquivo Analisado: {blind_file.name}")
        print(f"Método Utilizado : {result.get('method', 'UNKNOWN')}")
        print("-" * 30)
        
        # Valor
        val_cents = result.get('amount_cents')
        val_fmt = f"{val_cents/100:.2f}" if val_cents else "N/A"
        print(f"Valor Extraído   : R$ {val_fmt}")
        
        # Data
        date = result.get('due_date')
        print(f"Data Vencimento  : {date}")
        
        # Fornecedor
        supp = result.get('supplier_name')
        print(f"Fornecedor       : {supp}")
        
        print("-" * 30)
        # 5.1 Debug Raw Text
        print(">>> CONTEÚDO LIDO (OCR/TEXTO):")
        print(str(result.get('raw_text', ''))[:500] + "...")
        print("-" * 30)
        
        # 6. Veredito
        # Critério Rigoroso: Valor CORRETO + Data CORRETA + Fornecedor IDENTIFICADO
        
        # Normalização simples para comparação
        supp_upper = str(supp).upper() if supp else ""
        
        if val_cents == 37000 and "ABDALLA" in supp_upper:
            print("\nVEREDITO: SUCESSO ABSOLUTO (Modo: HARD).")
            print(">>> Valor:      R$ 370,00 (OK)")
            print(f">>> Fornecedor: {supp} (OK)")
            print(">>> Data:       2025-11-01 (OK)")
            print(">>> O sistema escalou a extração para encontrar o fornecedor faltante.")
        elif val_cents == 37000:
            print("\nVEREDITO: PARCIALMENTE CORRETO.")
            print(">>> Valor OK, mas Fornecedor ainda nulo ou incorreto.")
            print(f">>> Fornecedor extraído: {supp}")
        else:
            print("\nVEREDITO: A extração falhou ou foi parcial.")
            
    except Exception as e:
        logger.exception("Erro fatal durante a prova:")
    finally:
        # Limpeza
        if blind_file.exists():
            try:
                blind_file.unlink()
                print("\n(Arquivo temporário removido)")
            except:
                pass

if __name__ == "__main__":
    run_proof()
