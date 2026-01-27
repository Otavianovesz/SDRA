"""
Renamer - Padronização de Arquivos
==================================
Fase 6/8: Renomeação e Entrega (Steps 131-150)
"""

import re
import shutil
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from database import DocumentStatus, SRDADatabase

logger = logging.getLogger(__name__)

@dataclass
class RenameResult:
    successful: int = 0
    failed: int = 0
    skipped: int = 0

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
        entity = metadata.get('entity_tag') or 'UNK'
        
        # Fornecedor
        supplier = (metadata.get('supplier_name') or metadata.get('supplier_clean') or 'FORNECEDOR').upper().replace(" ", "_")
        supplier = supplier[:30] # Trunca nomes longos
        
        # Valor
        try:
            val_cents = metadata.get('amount_cents', 0)
            val_fmt = f"{val_cents/100:.2f}".replace('.', ',')
        except:
            val_fmt = "0,00"
            
        # Tipo e Número
        doc_type = metadata.get('doc_type', 'DOC').upper()
        # doc_number field isn't explicitly in all DB queries, so fallback
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

class DocumentRenamer:
    """
    Componente responsável por orquestrar a renomeação em massa baseada no DB.
    Recupera documentos processados e aplica a padronização de nomes.
    """
    def __init__(self, db: SRDADatabase, output_dir: str = "Output"):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, dry_run: bool = False, copy_mode: bool = True) -> RenameResult:
        """
        Executa o processo de renomeação.
        
        Args:
            dry_run: Se True, apenas simula e loga.
            copy_mode: Se True, copia arquivos. Se False, move.
        
        Returns:
            RenameResult com estatísticas.
        """
        result = RenameResult()
        
        # Query documents eligible for renaming (Parsed or Reconciled, not renamed yet)
        # We fetch necessary metadata
        documents = self._fetch_eligible_documents()
        
        for doc in documents:
            try:
                original_path = Path(doc['original_path'])
                if not original_path.exists():
                    logger.warning(f"Arquivo original não encontrado: {original_path}")
                    result.failed += 1
                    continue
                
                # Generate new name
                new_filename = Renamer.generate_filename(doc)
                dest_path = self.output_dir / new_filename
                
                # Handle collisions
                counter = 1
                while dest_path.exists():
                    stem = dest_path.stem
                    dest_path = self.output_dir / f"{stem} ({counter}).pdf"
                    counter += 1
                
                if dry_run:
                    logger.info(f"[DRY RUN] {original_path.name} -> {dest_path.name}")
                    result.successful += 1
                    continue
                
                # Execute copy/move
                if copy_mode:
                    shutil.copy2(original_path, dest_path)
                else:
                    shutil.move(str(original_path), str(dest_path))
                    
                # Update status in DB
                self.db.update_document_status(doc['id'], DocumentStatus.RENAMED)
                logger.info(f"Renomeado: {original_path.name} -> {dest_path.name}")
                result.successful += 1
                
            except Exception as e:
                logger.error(f"Erro ao renomear doc {doc.get('id')}: {e}")
                result.failed += 1
                
        return result

    def _fetch_eligible_documents(self) -> List[Dict]:
        """Busca documentos que têm dados suficientes para renomear."""
        # Using DuckDB to join documents and transactions
        query = """
            SELECT 
                d.id, d.original_path, d.doc_type, d.entity_tag, d.doc_number, d.access_key,
                t.amount_cents, t.supplier_clean, t.emission_date, t.due_date
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.status IN ('PARSED', 'RECONCILED', 'INGESTED') 
              AND d.status != 'RENAMED'
              AND d.status != 'ERROR'
        """
        try:
            return self.db._fetchall_as_dict(query)
        except Exception as e:
            logger.error(f"Error fetching documents for rename: {e}")
            return []
