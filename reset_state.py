#!/usr/bin/env python3
"""
reset_state.py - Script para resetar o estado do banco de dados
================================================================
Limpa o banco caso precise reiniciar do zero.

Fase 1, Step 17 do Master Protocol
"""

import os
import sys
import shutil
from pathlib import Path

# Diretório do projeto
PROJECT_DIR = Path(__file__).parent

# Arquivos do banco de dados
DB_FILES = [
    "srda_rural.db",
    "srda_rural.duckdb",
    "*.db-journal",
    "*.db-wal",
    "*.db-shm"
]

# Diretórios temporários
TEMP_DIRS = [
    "temp_images",
    "__pycache__",
    ".cache"
]


def confirm_reset() -> bool:
    """Solicita confirmação do usuário."""
    print("\n" + "=" * 60)
    print("ATENÇÃO: Este script irá APAGAR todo o banco de dados!")
    print("=" * 60)
    print("\nIsso inclui:")
    print("  - Todos os documentos importados")
    print("  - Todas as transações extraídas")
    print("  - Todo o histórico de reconciliação")
    print("  - Cache de fornecedores")
    print()
    
    response = input("Tem certeza que deseja continuar? (digite 'SIM APAGAR' para confirmar): ")
    return response.strip() == "SIM APAGAR"


def find_files(pattern: str) -> list:
    """Encontra arquivos matching o padrão."""
    if "*" in pattern:
        import glob
        return list(PROJECT_DIR.glob(pattern))
    else:
        path = PROJECT_DIR / pattern
        return [path] if path.exists() else []


def reset_database():
    """Remove todos os arquivos do banco de dados."""
    print("\n[1/3] Removendo arquivos do banco de dados...")
    
    removed = 0
    for pattern in DB_FILES:
        for path in find_files(pattern):
            try:
                if path.is_file():
                    path.unlink()
                    print(f"  ✓ Removido: {path.name}")
                    removed += 1
            except Exception as e:
                print(f"  ✗ Erro ao remover {path.name}: {e}")
    
    if removed == 0:
        print("  (nenhum arquivo de banco encontrado)")
    
    return removed


def reset_temp_dirs():
    """Remove diretórios temporários."""
    print("\n[2/3] Limpando diretórios temporários...")
    
    removed = 0
    for dir_name in TEMP_DIRS:
        dir_path = PROJECT_DIR / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"  ✓ Removido: {dir_name}/")
                removed += 1
            except Exception as e:
                print(f"  ✗ Erro ao remover {dir_name}: {e}")
    
    if removed == 0:
        print("  (nenhum diretório temporário encontrado)")
    
    return removed


def reset_lock_files():
    """Remove arquivos de lock."""
    print("\n[3/3] Removendo arquivos de lock...")
    
    lock_patterns = ["*.lock", ".srda_*.lock"]
    removed = 0
    
    for pattern in lock_patterns:
        for path in find_files(pattern):
            try:
                path.unlink()
                print(f"  ✓ Removido: {path.name}")
                removed += 1
            except Exception as e:
                print(f"  ✗ Erro ao remover {path.name}: {e}")
    
    if removed == 0:
        print("  (nenhum arquivo de lock encontrado)")
    
    return removed


def main():
    """Função principal."""
    print("\n" + "=" * 60)
    print("SRDA-Rural - Reset de Estado")
    print("=" * 60)
    
    # Verifica se está rodando em modo forçado
    force_mode = len(sys.argv) > 1 and sys.argv[1] in ('-f', '--force')
    
    if not force_mode:
        if not confirm_reset():
            print("\nOperação cancelada pelo usuário.")
            return 1
    else:
        print("\n[Modo forçado - sem confirmação]")
    
    # Executa reset
    db_count = reset_database()
    temp_count = reset_temp_dirs()
    lock_count = reset_lock_files()
    
    total = db_count + temp_count + lock_count
    
    print("\n" + "=" * 60)
    print(f"Reset concluído! {total} item(ns) removido(s).")
    print("=" * 60)
    print("\nPróximos passos:")
    print("  1. Execute 'python main.py' para reiniciar o sistema")
    print("  2. O banco será recriado automaticamente")
    print("  3. Importe seus documentos novamente")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
