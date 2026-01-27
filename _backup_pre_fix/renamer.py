"""
Renamer - Padronização de Arquivos
==================================
Fase 6/8: Renomeação e Entrega (Steps 131-150)
"""

import re
import shutil
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class Renamer:
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Remove caracteres inválidos para Windows/Linux."""
        # Remove / \ : * ? " < > |
        cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
        # Remove espaços extras
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def generate_filename(metadata: Dict) -> str:
        """
        Gera nome padrão: DATA_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
        """
        # Data (Vencimento ou Emissão)
        date_str = metadata.get('due_date') or metadata.get('emission_date') or datetime.now().strftime("%Y-%m-%d")
        try:
            # Converte YYYY-MM-DD para 2024.12.31 ou DD.MM.YYYY
            dt = datetime.fromisoformat(date_str)
            # Padrão: YYYY.MM.DD (ordenação fácil)
            date_fmt = dt.strftime("%Y.%m.%d")
        except:
            date_fmt = "0000.00.00"
            
        # Entidade
        entity = metadata.get('entity_tag', 'UNK')
        
        # Fornecedor
        supplier = metadata.get('supplier_name', 'FORNECEDOR').upper().replace(" ", "_")
        supplier = supplier[:30] # Trunca nomes longos
        
        # Valor
        try:
            val_cents = metadata.get('amount_cents', 0)
            val_fmt = f"{val_cents/100:.2f}".replace('.', ',')
        except:
            val_fmt = "0,00"
            
        # Tipo e Número
        doc_type = metadata.get('doc_type', 'DOC').upper()
        number = metadata.get('doc_number', '') or metadata.get('access_key', '')[-6:] or 'SN'
        
        # Monta partes
        parts = [
            date_fmt,
            entity,
            supplier,
            val_fmt,
            doc_type,
            number
        ]
        
        # Flags (Parcela)
        if metadata.get('is_installment'):
             curr = metadata.get('installment_current', 0)
             total = metadata.get('installment_total', 0)
             parts.append(f"PARC_{curr}-{total}")
             
        name = "_".join(str(p) for p in parts if p)
        return Renamer.sanitize_filename(name) + ".pdf"

    @staticmethod
    def execute_rename(source_path: Path, output_dir: Path, metadata: Dict) -> Path:
        """Copia e renomeia o arquivo."""
        new_name = Renamer.generate_filename(metadata)
        dest_path = output_dir / new_name
        
        # Trata colisão
        counter = 1
        while dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            # Se já tem contador (1), remove para incrementar
            if re.search(r'\(\d+\)$', stem):
                stem = re.sub(r'\(\d+\)$', '', stem)
            
            dest_path = output_dir / f"{stem.strip()} ({counter}){suffix}"
            counter += 1
            
        shutil.copy2(source_path, dest_path)
        return dest_path
