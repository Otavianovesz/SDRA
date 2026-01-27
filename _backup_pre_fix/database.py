"""
SRDA-Rural Database Module
==========================
Sistema de Reconciliação Documental e Automação Rural

Este módulo implementa a camada de persistência usando DuckDB:
- Motor OLAP para alta performance em reconciliação
- Suporte a transações ACID via MVCC
- Persistência em arquivo único (.db)

Referência: Automação e Reconciliação Financeira com IA.txt (Seção 5)
"""

import duckdb
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# ENUMS E CONSTANTES
# ==============================================================================

class EntityTag(Enum):
    """
    Entidades financeiras distintas do sistema.
    VG = Vagner, MV = Marcelli
    """
    VG = "VG"  # Vagner - CPF: 964.128.440-15
    MV = "MV"  # Marcelli


class DocumentType(Enum):
    """Tipos de documentos suportados pelo sistema."""
    NFE = "NFE"           # Nota Fiscal Eletrônica (Modelo 55)
    NFSE = "NFSE"         # Nota Fiscal de Serviço Eletrônica
    BOLETO = "BOLETO"     # Boleto Bancário
    COMPROVANTE = "COMPROVANTE"  # Comprovante de Pagamento
    UNKNOWN = "UNKNOWN"   # Tipo não identificado


class DocumentStatus(Enum):
    """
    Estados do processamento de documentos.
    Permite pausar e retomar o trabalho (crash-only software).
    """
    INGESTED = "INGESTED"       # Arquivo detectado e hasheado
    PARSED = "PARSED"           # Texto extraído e metadados parseados
    RECONCILED = "RECONCILED"   # Vinculado a outros documentos
    RENAMED = "RENAMED"         # Arquivo renomeado no disco
    ERROR = "ERROR"             # Erro durante processamento


class MatchType(Enum):
    """Tipos de correspondência entre documentos."""
    EXACT = "EXACT"       # Match exato de valores
    FUZZY = "FUZZY"       # Match aproximado (tolerância)
    MANUAL = "MANUAL"     # Vínculo forçado pelo usuário


# CPFs conhecidos para classificação automática de entidade
KNOWN_CPFS = {
    "96412844015": EntityTag.VG,      # Vagner
    "964.128.440-15": EntityTag.VG,   # Vagner (formatado)
    "89602692120": EntityTag.MV,      # Marceli
    "896.026.921-20": EntityTag.MV,   # Marceli (formatado)
}


# ==============================================================================
# GERENCIADOR DE BANCO DE DADOS
# ==============================================================================

class SRDADatabase:
    """
    Gerenciador de banco de dados DuckDB para o SRDA-Rural.
    
    Características:
    - Motor OLAP embutido (alta performance)
    - Suporte a concorrência via MVCC
    - Persistência em arquivo único (.db)
    """
    
    def __init__(self, db_path: str = "srda_rural.db"):
        self.db_path = db_path
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._initialize_database()
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Retorna a conexão ativa com o banco."""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path)
        return self._connection
    
    @contextmanager
    def transaction(self):
        """Context manager para transações."""
        self.connection.begin()
        try:
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e

    def _initialize_database(self):
        """Cria as tabelas e sequências se não existirem."""
        conn = self.connection
        
        # Sequências para IDs
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_documentos START 1;")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_transacoes START 1;")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_duplicatas START 1;")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_matches START 1;")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_corrections START 1;")
        
        # TABELAS
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documentos (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_documentos'),
                file_hash VARCHAR NOT NULL,
                original_path VARCHAR NOT NULL,
                doc_type VARCHAR DEFAULT 'UNKNOWN',
                entity_tag VARCHAR,
                raw_text VARCHAR,
                status VARCHAR DEFAULT 'INGESTED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                page_start INTEGER DEFAULT 1,
                page_end INTEGER DEFAULT 1,
                access_key VARCHAR,
                doc_number VARCHAR,
                UNIQUE(file_hash)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transacoes (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_transacoes'),
                doc_id INTEGER NOT NULL,
                supplier_clean VARCHAR,
                amount_cents INTEGER NOT NULL DEFAULT 0,
                entidade_pagadora VARCHAR,
                emission_date VARCHAR,
                due_date VARCHAR,
                payment_date VARCHAR,
                bank_code VARCHAR,
                barcode VARCHAR,
                digitable_line VARCHAR,
                sisbb_auth VARCHAR,
                is_scheduled INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documentos(id)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS duplicatas (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_duplicatas'),
                nfe_id INTEGER NOT NULL,
                seq_num INTEGER NOT NULL,
                due_date VARCHAR NOT NULL,
                amount_cents INTEGER NOT NULL,
                reconciled INTEGER DEFAULT 0,
                boleto_id INTEGER,
                FOREIGN KEY (nfe_id) REFERENCES documentos(id),
                FOREIGN KEY (boleto_id) REFERENCES documentos(id)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_matches'),
                parent_doc_id INTEGER NOT NULL,
                child_doc_id INTEGER NOT NULL,
                match_type VARCHAR DEFAULT 'EXACT',
                confidence DOUBLE DEFAULT 0.0,
                status VARCHAR DEFAULT 'SUGGESTED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confirmed_at TIMESTAMP,
                FOREIGN KEY (parent_doc_id) REFERENCES documentos(id),
                FOREIGN KEY (child_doc_id) REFERENCES documentos(id),
                UNIQUE(parent_doc_id, child_doc_id)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections_log (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_corrections'),
                doc_id INTEGER NOT NULL,
                field_name VARCHAR NOT NULL,
                ocr_value VARCHAR,
                user_value VARCHAR NOT NULL,
                image_hash VARCHAR,
                bbox VARCHAR,
                original_confidence DOUBLE,
                extractor_source VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documentos(id)
            );
        """)

        # Índices
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documentos_status ON documentos(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documentos_entity ON documentos(entity_tag);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transacoes_subset ON transacoes(supplier_clean, amount_cents);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_field ON corrections_log(field_name);")
        except:
            pass

    # --- Helpers ---
    def _fetchall_as_dict(self, query: str, params: List = []) -> List[Dict]:
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    
    # --- Document methods ---
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def document_exists(self, file_hash: str) -> bool:
        res = self.connection.execute("SELECT 1 FROM documentos WHERE file_hash = ?", [file_hash]).fetchone()
        return res is not None

    def insert_document(self, file_path: str, doc_type: DocumentType = DocumentType.UNKNOWN,
                        entity_tag: Optional[EntityTag] = None, page_start: int = 1, page_end: int = 1) -> Optional[int]:
        file_hash = self.calculate_file_hash(file_path)
        if self.document_exists(file_hash): return None
        try:
            res = self.connection.execute("""
                INSERT INTO documentos (file_hash, original_path, doc_type, entity_tag, page_start, page_end, status)
                VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id
            """, [file_hash, file_path, doc_type.value, entity_tag.value if entity_tag else None, 
                  page_start, page_end, DocumentStatus.INGESTED.value]).fetchone()
            return res[0] if res else None
        except Exception as e:
            logger.error(f"Insert doc error: {e}")
            return None

    def update_document_status(self, doc_id: int, status: DocumentStatus, raw_text: Optional[str] = None):
        if raw_text:
            self.connection.execute("UPDATE documentos SET status=?, raw_text=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                                    [status.value, raw_text, doc_id])
        else:
            self.connection.execute("UPDATE documentos SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                                    [status.value, doc_id])

    def get_documents_by_status(self, status: DocumentStatus) -> List[Dict[str, Any]]:
        return self._fetchall_as_dict("SELECT * FROM documentos WHERE status = ?", [status.value])

    def get_documents_by_entity(self, entity: EntityTag) -> List[Dict[str, Any]]:
        return self._fetchall_as_dict("SELECT * FROM documentos WHERE entity_tag = ?", [entity.value])
    
    # --- Transaction methods ---
    def insert_transaction(self, doc_id: int, amount_cents: int, entidade_pagadora: EntityTag,
                           supplier_clean: Optional[str] = None, emission_date: Optional[str] = None,
                           due_date: Optional[str] = None, payment_date: Optional[str] = None,
                           is_scheduled: bool = False) -> int:
        res = self.connection.execute("""
            INSERT INTO transacoes (doc_id, amount_cents, entidade_pagadora, supplier_clean, emission_date, due_date, payment_date, is_scheduled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
        """, [doc_id, amount_cents, entidade_pagadora.value, supplier_clean, emission_date, due_date, payment_date, 1 if is_scheduled else 0]).fetchone()
        return res[0]

    def get_open_transactions_by_supplier(self, supplier: str, entity: Optional[EntityTag] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT t.*, d.doc_type, d.status
            FROM transacoes t
            JOIN documentos d ON t.doc_id = d.id
            WHERE t.supplier_clean = ? AND d.status != 'RECONCILED'
        """
        params = [supplier]
        if entity:
            query += " AND t.entidade_pagadora = ?"
            params.append(entity.value)
        return self._fetchall_as_dict(query, params)

    # --- Installment methods ---
    def insert_installment(self, nfe_id: int, seq_num: int, due_date: str, amount_cents: int) -> int:
        res = self.connection.execute("""
            INSERT INTO duplicatas (nfe_id, seq_num, due_date, amount_cents)
            VALUES (?, ?, ?, ?) RETURNING id
        """, [nfe_id, seq_num, due_date, amount_cents]).fetchone()
        return res[0]
        
    def get_installments_by_nfe(self, nfe_id: int) -> List[Dict[str, Any]]:
        return self._fetchall_as_dict("SELECT * FROM duplicatas WHERE nfe_id = ? ORDER BY seq_num", [nfe_id])
        

    def count_installments(self, nfe_id: int) -> int:
        res = self.connection.execute("SELECT COUNT(*) FROM duplicatas WHERE nfe_id = ?", [nfe_id]).fetchone()
        return res[0] if res else 0

    def link_installment_to_boleto(self, nfe_id: int, boleto_id: int, amount_cents: int) -> bool:
        """
        Vincula uma parcela específica de uma NF-e a um boleto.
        Usado para gerar nomes como PARC_1-3.
        """
        # Tenta encontrar parcela com mesmo valor não reconciliada
        res = self.connection.execute("""
            UPDATE duplicatas 
            SET boleto_id = ?, reconciled = 1 
            WHERE nfe_id = ? AND amount_cents = ? AND reconciled = 0
            RETURNING id
        """, [boleto_id, nfe_id, amount_cents]).fetchone()
        return res is not None

    # --- Match methods ---
    def insert_match(self, parent_doc_id: int, child_doc_id: int, match_type: MatchType = MatchType.EXACT, confidence: float = 1.0) -> Optional[int]:
        try:
            res = self.connection.execute("""
                INSERT INTO matches (parent_doc_id, child_doc_id, match_type, confidence, status)
                VALUES (?, ?, ?, ?, 'SUGGESTED') RETURNING id
            """, [parent_doc_id, child_doc_id, match_type.value, confidence]).fetchone()
            return res[0]
        except: return None
    
    def confirm_match(self, match_id: int):
        self.connection.execute("UPDATE matches SET status='CONFIRMED', confirmed_at=CURRENT_TIMESTAMP WHERE id=?", [match_id])

    def get_matches_by_document(self, doc_id: int) -> List[Dict[str, Any]]:
        return self._fetchall_as_dict("""
            SELECT m.*, p.original_path as parent_path, c.original_path as child_path
            FROM matches m
            JOIN documentos p ON m.parent_doc_id = p.id
            JOIN documentos c ON m.child_doc_id = c.id
            WHERE m.parent_doc_id = ? OR m.child_doc_id = ?
        """, [doc_id, doc_id])
        
    def get_all_open_documents(self) -> List[Dict]:
        return self._fetchall_as_dict("""
            SELECT d.id, d.doc_type, d.entity_tag, d.original_path, 
                   t.amount_cents, t.supplier_clean, t.due_date, t.payment_date, t.emission_date
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.status != 'RECONCILED'
        """)

    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        try:
            stats['total_documents'] = self.connection.execute("SELECT COUNT(*) FROM documentos").fetchone()[0]
            stats['total_transactions'] = self.connection.execute("SELECT COUNT(*) FROM transacoes").fetchone()[0]
            stats['total_matches'] = self.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            stats['total_corrections'] = self.connection.execute("SELECT COUNT(*) FROM corrections_log").fetchone()[0]
        except:
             stats['total_documents'] = 0
             stats['total_transactions'] = 0
             stats['total_matches'] = 0
             stats['total_corrections'] = 0
        return stats

    # ==========================================================================
    # ACTIVE LEARNING - Correções e Aprendizado
    # ==========================================================================
    
    def log_correction(self, doc_id: int, field_name: str, ocr_value: Optional[str],
                       user_value: str, original_confidence: Optional[float] = None,
                       extractor_source: Optional[str] = None, bbox: Optional[str] = None,
                       image_hash: Optional[str] = None) -> Optional[int]:
        """
        Registra uma correção do usuário para Active Learning.
        
        Args:
            doc_id: ID do documento corrigido
            field_name: Nome do campo (fornecedor, valor, data_vencimento, etc.)
            ocr_value: Valor original extraído pelo OCR
            user_value: Valor correto informado pelo usuário
            original_confidence: Confiança original da extração
            extractor_source: Fonte da extração (spatial, paddle, florence, etc.)
            bbox: Bounding box do campo (para aprendizado espacial)
            image_hash: Hash da imagem para evitar duplicatas
            
        Returns:
            ID da correção inserida ou None
        """
        try:
            res = self.connection.execute("""
                INSERT INTO corrections_log 
                (doc_id, field_name, ocr_value, user_value, original_confidence, 
                 extractor_source, bbox, image_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
            """, [doc_id, field_name, ocr_value, user_value, original_confidence,
                  extractor_source, bbox, image_hash]).fetchone()
            
            logger.info(f"[ACTIVE LEARNING] Correção registrada: doc={doc_id}, "
                       f"campo={field_name}, '{ocr_value}' -> '{user_value}'")
            return res[0] if res else None
        except Exception as e:
            logger.error(f"Erro ao registrar correção: {e}")
            return None
    
    def get_learned_offsets(self, supplier: str, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Recupera offsets espaciais aprendidos para um fornecedor/campo.
        
        Usado pelo SpatialExtractor para ajustar ROIs baseado em correções anteriores.
        
        Args:
            supplier: Nome do fornecedor (normalizado)
            field_name: Nome do campo a buscar
            
        Returns:
            Dict com offset aprendido ou None se não houver correções
        """
        try:
            # Busca a correção mais recente para este fornecedor/campo
            result = self.connection.execute("""
                SELECT cl.bbox, cl.user_value, cl.extractor_source, cl.created_at
                FROM corrections_log cl
                JOIN documentos d ON cl.doc_id = d.id
                JOIN transacoes t ON d.id = t.doc_id
                WHERE UPPER(t.supplier_clean) = UPPER(?)
                  AND cl.field_name = ?
                ORDER BY cl.created_at DESC
                LIMIT 1
            """, [supplier, field_name]).fetchone()
            
            if result:
                return {
                    'bbox': result[0],
                    'learned_value': result[1],
                    'source': result[2],
                    'learned_at': result[3]
                }
            return None
        except Exception as e:
            logger.error(f"Erro ao buscar offsets aprendidos: {e}")
            return None
    
    def get_correction_history(self, field_name: Optional[str] = None, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retorna histórico de correções para análise/auditoria.
        
        Args:
            field_name: Filtrar por campo específico (opcional)
            limit: Máximo de registros a retornar
            
        Returns:
            Lista de correções ordenadas por data
        """
        query = """
            SELECT cl.*, d.doc_type, t.supplier_clean
            FROM corrections_log cl
            JOIN documentos d ON cl.doc_id = d.id
            LEFT JOIN transacoes t ON d.id = t.doc_id
        """
        params = []
        
        if field_name:
            query += " WHERE cl.field_name = ?"
            params.append(field_name)
        
        query += f" ORDER BY cl.created_at DESC LIMIT {limit}"
        
        return self._fetchall_as_dict(query, params)

    def close(self):
        if self._connection:
            self._connection.close()

    # --- Utility methods ---
    @staticmethod
    def parse_cpf_to_entity(cpf: str) -> Optional[EntityTag]:
        cpf_clean = cpf.replace(".", "").replace("-", "").strip()
        if cpf_clean in KNOWN_CPFS: return KNOWN_CPFS[cpf_clean]
        if cpf in KNOWN_CPFS: return KNOWN_CPFS[cpf]
        return None

    @staticmethod
    def amount_to_cents(value_str: str) -> int:
        clean = value_str.replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return int(float(clean) * 100)
        except: return 0

    @staticmethod
    def cents_to_display(cents: int) -> str:
        value = cents / 100
        formatted = f"{value:,.2f}"
        formatted = formatted.replace(",", "@").replace(".", ",").replace("@", ".")
        return formatted

if __name__ == "__main__":
    db = SRDADatabase("srda_teste.db")
    print("SRDA-Rural DuckDB Initialized")
    # Basic smoke test
    try:
        db.insert_document("test.pdf", DocumentType.NFE, EntityTag.VG)
        stats = db.get_statistics()
        print(f"Stats: {stats}")
    except Exception as e:
        print(f"Test failed: {e}")
