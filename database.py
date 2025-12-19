"""
SRDA-Rural Database Module
==========================
Sistema de Reconciliação Documental e Automação Rural

Este módulo implementa a camada de persistência usando SQLite com:
- Modo WAL para leituras/escritas simultâneas
- Suporte a crash-recovery (transações atômicas)
- Distinção clara entre entidades financeiras (Vagner/Marcelli)

Referência: Automação e Reconciliação Financeira com IA.txt (Seção 5)
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from contextlib import contextmanager


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
}


# ==============================================================================
# GERENCIADOR DE BANCO DE DADOS
# ==============================================================================

class SRDADatabase:
    """
    Gerenciador de banco de dados SQLite para o SRDA-Rural.
    
    Características:
    - Modo WAL ativado para performance
    - Índices compostos para buscas rápidas
    - Suporte a transações atômicas
    - Detecção de duplicatas via hash MD5
    """
    
    def __init__(self, db_path: str = "srda_rural.db"):
        """
        Inicializa a conexão com o banco de dados.
        
        Args:
            db_path: Caminho para o arquivo do banco SQLite
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_database()
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Retorna a conexão ativa com o banco."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            # Ativa modo WAL para leituras/escritas simultâneas
            self._connection.execute("PRAGMA journal_mode=WAL;")
            # Ativa foreign keys
            self._connection.execute("PRAGMA foreign_keys=ON;")
        return self._connection
    
    @contextmanager
    def transaction(self):
        """
        Context manager para transações atômicas.
        Garante commit ou rollback automático.
        """
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
    
    def _initialize_database(self):
        """Cria as tabelas se não existirem."""
        with self.transaction() as cursor:
            # ==========================================
            # TABELA: documentos (docs_raw)
            # Armazena integridade física e metadados
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documentos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Integridade e identificação
                    file_hash TEXT NOT NULL UNIQUE,
                    original_path TEXT NOT NULL,
                    
                    -- Classificação
                    doc_type TEXT DEFAULT 'UNKNOWN',
                    entity_tag TEXT,
                    
                    -- Cache de extração (evita re-OCR)
                    raw_text TEXT,
                    
                    -- Estado do processamento
                    status TEXT DEFAULT 'INGESTED',
                    
                    -- Metadados de auditoria
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Para arquivos combinados: intervalo de páginas
                    page_start INTEGER DEFAULT 1,
                    page_end INTEGER DEFAULT 1,
                    
                    -- Chave de acesso NF-e (44 dígitos)
                    access_key TEXT,
                    
                    -- Número do documento (normalizado)
                    doc_number TEXT
                );
            """)
            
            # ==========================================
            # TABELA: transacoes (entities)
            # Dados financeiros extraídos e normalizados
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Referência ao documento físico
                    doc_id INTEGER NOT NULL,
                    
                    -- Fornecedor/Prestador normalizado
                    supplier_clean TEXT,
                    
                    -- Valor em CENTAVOS (evita erros de ponto flutuante)
                    amount_cents INTEGER NOT NULL DEFAULT 0,
                    
                    -- Entidade pagadora (VG ou MV)
                    entidade_pagadora TEXT,
                    
                    -- Datas relevantes
                    emission_date TEXT,
                    due_date TEXT,
                    payment_date TEXT,
                    
                    -- Metadados bancários (para boletos/comprovantes)
                    bank_code TEXT,
                    barcode TEXT,
                    digitable_line TEXT,
                    
                    -- Autenticação SISBB (Banco do Brasil)
                    sisbb_auth TEXT,
                    
                    -- Indicador de agendamento vs pagamento efetivo
                    is_scheduled INTEGER DEFAULT 0,
                    
                    -- Auditoria
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (doc_id) REFERENCES documentos(id)
                        ON DELETE CASCADE
                );
            """)
            
            # ==========================================
            # TABELA: duplicatas (installments)
            # Tabela de parcelas extraídas de NF-e
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicatas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Referência à NF-e
                    nfe_id INTEGER NOT NULL,
                    
                    -- Número da parcela (001, 002, etc.)
                    seq_num INTEGER NOT NULL,
                    
                    -- Dados da duplicata
                    due_date TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    
                    -- Status de reconciliação
                    reconciled INTEGER DEFAULT 0,
                    boleto_id INTEGER,
                    
                    FOREIGN KEY (nfe_id) REFERENCES documentos(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (boleto_id) REFERENCES documentos(id)
                        ON DELETE SET NULL
                );
            """)
            
            # ==========================================
            # TABELA: matches (links)
            # Grafo de reconciliação (arestas)
            # ==========================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Geralmente Boleto ou Comprovante
                    parent_doc_id INTEGER NOT NULL,
                    
                    -- Geralmente Nota Fiscal
                    child_doc_id INTEGER NOT NULL,
                    
                    -- Tipo de correspondência
                    match_type TEXT DEFAULT 'EXACT',
                    
                    -- Score de confiança da IA (0.0 - 1.0)
                    confidence REAL DEFAULT 0.0,
                    
                    -- Status do vínculo
                    status TEXT DEFAULT 'SUGGESTED',
                    
                    -- Auditoria
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confirmed_at TIMESTAMP,
                    
                    FOREIGN KEY (parent_doc_id) REFERENCES documentos(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (child_doc_id) REFERENCES documentos(id)
                        ON DELETE CASCADE,
                    
                    -- Evita vínculos duplicados
                    UNIQUE(parent_doc_id, child_doc_id)
                );
            """)
            
            # ==========================================
            # ÍNDICES PARA PERFORMANCE
            # ==========================================
            
            # Índice para detecção de duplicatas
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documentos_hash 
                ON documentos(file_hash);
            """)
            
            # Índice para busca por status
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documentos_status 
                ON documentos(status);
            """)
            
            # Índice para busca por entidade
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documentos_entity 
                ON documentos(entity_tag);
            """)
            
            # Índice composto para algoritmo Subset Sum
            # Permite busca rápida de notas por fornecedor e valor
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transacoes_subset 
                ON transacoes(supplier_clean, amount_cents);
            """)
            
            # Índice para busca por entidade pagadora
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transacoes_entidade 
                ON transacoes(entidade_pagadora);
            """)
    
    # ==========================================================================
    # MÉTODOS PARA DOCUMENTOS
    # ==========================================================================
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        Calcula o hash MD5 de um arquivo para detecção de duplicatas.
        
        Args:
            file_path: Caminho absoluto do arquivo
            
        Returns:
            String hexadecimal do hash MD5
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def document_exists(self, file_hash: str) -> bool:
        """
        Verifica se um documento já foi processado (idempotência).
        
        Args:
            file_hash: Hash MD5 do arquivo
            
        Returns:
            True se o documento já existe no banco
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT 1 FROM documentos WHERE file_hash = ?",
            (file_hash,)
        )
        return cursor.fetchone() is not None
    
    def insert_document(
        self,
        file_path: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        entity_tag: Optional[EntityTag] = None,
        page_start: int = 1,
        page_end: int = 1
    ) -> Optional[int]:
        """
        Insere um novo documento no banco.
        
        Args:
            file_path: Caminho original do arquivo
            doc_type: Tipo do documento
            entity_tag: Entidade financeira (VG ou MV)
            page_start: Página inicial (para PDFs combinados)
            page_end: Página final
            
        Returns:
            ID do documento inserido ou None se já existir
        """
        file_hash = self.calculate_file_hash(file_path)
        
        if self.document_exists(file_hash):
            return None
        
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO documentos 
                (file_hash, original_path, doc_type, entity_tag, page_start, page_end, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_hash,
                file_path,
                doc_type.value,
                entity_tag.value if entity_tag else None,
                page_start,
                page_end,
                DocumentStatus.INGESTED.value
            ))
            return cursor.lastrowid
    
    def update_document_status(
        self,
        doc_id: int,
        status: DocumentStatus,
        raw_text: Optional[str] = None
    ):
        """
        Atualiza o status de processamento de um documento.
        
        Args:
            doc_id: ID do documento
            status: Novo status
            raw_text: Texto extraído (cache para evitar re-OCR)
        """
        with self.transaction() as cursor:
            if raw_text:
                cursor.execute("""
                    UPDATE documentos 
                    SET status = ?, raw_text = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status.value, raw_text, doc_id))
            else:
                cursor.execute("""
                    UPDATE documentos 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status.value, doc_id))
    
    def get_documents_by_status(
        self,
        status: DocumentStatus
    ) -> List[Dict[str, Any]]:
        """
        Retorna documentos filtrados por status.
        Útil para retomar processamento após crash.
        
        Args:
            status: Status desejado
            
        Returns:
            Lista de documentos como dicionários
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM documentos WHERE status = ?",
            (status.value,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_documents_by_entity(
        self,
        entity: EntityTag
    ) -> List[Dict[str, Any]]:
        """
        Retorna documentos filtrados por entidade (Vagner ou Marcelli).
        
        Args:
            entity: Tag da entidade (VG ou MV)
            
        Returns:
            Lista de documentos como dicionários
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM documentos WHERE entity_tag = ?",
            (entity.value,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================================================
    # MÉTODOS PARA TRANSAÇÕES
    # ==========================================================================
    
    def insert_transaction(
        self,
        doc_id: int,
        amount_cents: int,
        entidade_pagadora: EntityTag,
        supplier_clean: Optional[str] = None,
        emission_date: Optional[str] = None,
        due_date: Optional[str] = None,
        payment_date: Optional[str] = None,
        is_scheduled: bool = False
    ) -> int:
        """
        Insere uma transação extraída de um documento.
        
        Args:
            doc_id: ID do documento fonte
            amount_cents: Valor em centavos (ex: R$ 100,00 = 10000)
            entidade_pagadora: VG (Vagner) ou MV (Marcelli)
            supplier_clean: Nome do fornecedor normalizado
            emission_date: Data de emissão (ISO8601)
            due_date: Data de vencimento (ISO8601)
            payment_date: Data de pagamento efetivo (ISO8601)
            is_scheduled: True se for agendamento, não pagamento
            
        Returns:
            ID da transação inserida
        """
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO transacoes 
                (doc_id, amount_cents, entidade_pagadora, supplier_clean,
                 emission_date, due_date, payment_date, is_scheduled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                amount_cents,
                entidade_pagadora.value,
                supplier_clean,
                emission_date,
                due_date,
                payment_date,
                1 if is_scheduled else 0
            ))
            return cursor.lastrowid
    
    def get_open_transactions_by_supplier(
        self,
        supplier: str,
        entity: Optional[EntityTag] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca transações não reconciliadas de um fornecedor.
        Usado pelo algoritmo Subset Sum para reconciliação.
        
        Args:
            supplier: Nome normalizado do fornecedor
            entity: Filtrar por entidade (opcional)
            
        Returns:
            Lista de transações abertas
        """
        cursor = self.connection.cursor()
        
        if entity:
            cursor.execute("""
                SELECT t.*, d.doc_type, d.status
                FROM transacoes t
                JOIN documentos d ON t.doc_id = d.id
                WHERE t.supplier_clean = ?
                  AND t.entidade_pagadora = ?
                  AND d.status != 'RECONCILED'
            """, (supplier, entity.value))
        else:
            cursor.execute("""
                SELECT t.*, d.doc_type, d.status
                FROM transacoes t
                JOIN documentos d ON t.doc_id = d.id
                WHERE t.supplier_clean = ?
                  AND d.status != 'RECONCILED'
            """, (supplier,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================================================
    # MÉTODOS PARA DUPLICATAS/PARCELAS
    # ==========================================================================
    
    def insert_installment(
        self,
        nfe_id: int,
        seq_num: int,
        due_date: str,
        amount_cents: int
    ) -> int:
        """
        Insere uma parcela extraída da tabela de duplicatas de uma NF-e.
        
        Args:
            nfe_id: ID do documento NF-e
            seq_num: Número sequencial da parcela (1, 2, 3...)
            due_date: Data de vencimento (ISO8601)
            amount_cents: Valor da parcela em centavos
            
        Returns:
            ID da duplicata inserida
        """
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO duplicatas 
                (nfe_id, seq_num, due_date, amount_cents)
                VALUES (?, ?, ?, ?)
            """, (nfe_id, seq_num, due_date, amount_cents))
            return cursor.lastrowid
    
    def get_installments_by_nfe(self, nfe_id: int) -> List[Dict[str, Any]]:
        """
        Retorna todas as parcelas de uma NF-e.
        
        Args:
            nfe_id: ID do documento NF-e
            
        Returns:
            Lista de duplicatas ordenadas por sequência
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM duplicatas 
            WHERE nfe_id = ?
            ORDER BY seq_num
        """, (nfe_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def count_installments(self, nfe_id: int) -> int:
        """
        Conta o número total de parcelas de uma NF-e.
        Usado para gerar sufixos PARC_X-Y.
        
        Args:
            nfe_id: ID do documento NF-e
            
        Returns:
            Número total de parcelas
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM duplicatas WHERE nfe_id = ?",
            (nfe_id,)
        )
        return cursor.fetchone()[0]
    
    # ==========================================================================
    # MÉTODOS PARA MATCHES (RECONCILIAÇÃO)
    # ==========================================================================
    
    def insert_match(
        self,
        parent_doc_id: int,
        child_doc_id: int,
        match_type: MatchType = MatchType.EXACT,
        confidence: float = 1.0
    ) -> Optional[int]:
        """
        Cria um vínculo entre documentos (aresta do grafo).
        
        Args:
            parent_doc_id: ID do documento pai (geralmente Boleto)
            child_doc_id: ID do documento filho (geralmente NF)
            match_type: Tipo de correspondência
            confidence: Score de confiança (0.0 a 1.0)
            
        Returns:
            ID do match ou None se já existir
        """
        try:
            with self.transaction() as cursor:
                cursor.execute("""
                    INSERT INTO matches 
                    (parent_doc_id, child_doc_id, match_type, confidence, status)
                    VALUES (?, ?, ?, ?, 'SUGGESTED')
                """, (parent_doc_id, child_doc_id, match_type.value, confidence))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Vínculo já existe
            return None
    
    def confirm_match(self, match_id: int):
        """
        Confirma um vínculo sugerido pela IA.
        
        Args:
            match_id: ID do match a confirmar
        """
        with self.transaction() as cursor:
            cursor.execute("""
                UPDATE matches 
                SET status = 'CONFIRMED', confirmed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (match_id,))
    
    def get_matches_by_document(self, doc_id: int) -> List[Dict[str, Any]]:
        """
        Retorna todos os vínculos de um documento.
        
        Args:
            doc_id: ID do documento
            
        Returns:
            Lista de matches (como pai ou filho)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT m.*, 
                   p.original_path as parent_path,
                   c.original_path as child_path
            FROM matches m
            JOIN documentos p ON m.parent_doc_id = p.id
            JOIN documentos c ON m.child_doc_id = c.id
            WHERE m.parent_doc_id = ? OR m.child_doc_id = ?
        """, (doc_id, doc_id))
        return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================================================
    # MÉTODOS UTILITÁRIOS
    # ==========================================================================
    
    @staticmethod
    def parse_cpf_to_entity(cpf: str) -> Optional[EntityTag]:
        """
        Identifica a entidade financeira pelo CPF.
        
        Args:
            cpf: CPF (formatado ou apenas números)
            
        Returns:
            EntityTag.VG ou EntityTag.MV, ou None se não reconhecido
        """
        # Remove formatação
        cpf_clean = cpf.replace(".", "").replace("-", "").strip()
        
        # Busca no dicionário de CPFs conhecidos
        if cpf_clean in KNOWN_CPFS:
            return KNOWN_CPFS[cpf_clean]
        if cpf in KNOWN_CPFS:
            return KNOWN_CPFS[cpf]
        
        return None
    
    @staticmethod
    def amount_to_cents(value_str: str) -> int:
        """
        Converte string de valor monetário brasileiro para centavos.
        
        Args:
            value_str: Valor como string (ex: "3.300,00" ou "R$ 1.740,00")
            
        Returns:
            Valor em centavos (inteiro)
        """
        # Remove R$, espaços e pontos de milhar
        clean = value_str.replace("R$", "").replace(" ", "").replace(".", "")
        # Substitui vírgula por ponto para conversão
        clean = clean.replace(",", ".")
        # Converte para centavos
        return int(float(clean) * 100)
    
    @staticmethod
    def cents_to_display(cents: int) -> str:
        """
        Converte centavos para string de exibição brasileira.
        
        Args:
            cents: Valor em centavos
            
        Returns:
            String formatada (ex: "3.300,00")
        """
        value = cents / 100
        # Formata com separador de milhar e decimal brasileiro
        formatted = f"{value:,.2f}"
        # Troca . por @ temporariamente, vírgula por ponto, @ por vírgula
        formatted = formatted.replace(",", "@").replace(".", ",").replace("@", ".")
        return formatted
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do banco de dados.
        
        Returns:
            Dicionário com contagens e métricas
        """
        cursor = self.connection.cursor()
        
        stats = {}
        
        # Total de documentos
        cursor.execute("SELECT COUNT(*) FROM documentos")
        stats["total_documents"] = cursor.fetchone()[0]
        
        # Por status
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM documentos 
            GROUP BY status
        """)
        stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Por entidade (Vagner vs Marcelli)
        cursor.execute("""
            SELECT entity_tag, COUNT(*) as count 
            FROM documentos 
            WHERE entity_tag IS NOT NULL
            GROUP BY entity_tag
        """)
        stats["by_entity"] = {row["entity_tag"]: row["count"] for row in cursor.fetchall()}
        
        # Total de transações
        cursor.execute("SELECT COUNT(*) FROM transacoes")
        stats["total_transactions"] = cursor.fetchone()[0]
        
        # Total de matches
        cursor.execute("SELECT COUNT(*) FROM matches")
        stats["total_matches"] = cursor.fetchone()[0]
        
        # Matches confirmados
        cursor.execute("SELECT COUNT(*) FROM matches WHERE status = 'CONFIRMED'")
        stats["confirmed_matches"] = cursor.fetchone()[0]
        
        return stats
    
    def update_transaction_fields(
        self,
        doc_id: int,
        supplier_clean: Optional[str] = None,
        amount_cents: Optional[int] = None,
        due_date: Optional[str] = None,
        payment_date: Optional[str] = None
    ):
        """
        Atualiza campos de uma transação (para edição manual na GUI).
        
        Args:
            doc_id: ID do documento
            supplier_clean: Novo nome do fornecedor
            amount_cents: Novo valor em centavos
            due_date: Nova data de vencimento (ISO8601)
            payment_date: Nova data de pagamento (ISO8601)
        """
        with self.transaction() as cursor:
            updates = []
            params = []
            
            if supplier_clean is not None:
                updates.append("supplier_clean = ?")
                params.append(supplier_clean)
            if amount_cents is not None:
                updates.append("amount_cents = ?")
                params.append(amount_cents)
            if due_date is not None:
                updates.append("due_date = ?")
                params.append(due_date)
            if payment_date is not None:
                updates.append("payment_date = ?")
                params.append(payment_date)
            
            if updates:
                params.append(doc_id)
                cursor.execute(f"""
                    UPDATE transacoes 
                    SET {', '.join(updates)}
                    WHERE doc_id = ?
                """, params)
    
    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self._connection:
            self._connection.close()
            self._connection = None


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Demonstração de uso do módulo
    db = SRDADatabase("srda_teste.db")
    
    print("=" * 60)
    print("SRDA-Rural Database - Teste de Inicialização")
    print("=" * 60)
    
    # Teste de conversão de valores
    test_values = ["3.300,00", "R$ 6.600,00", "1.740,00"]
    print("\nConversão de valores monetários:")
    for val in test_values:
        cents = SRDADatabase.amount_to_cents(val)
        back = SRDADatabase.cents_to_display(cents)
        print(f"  {val} -> {cents} centavos -> {back}")
    
    # Teste de identificação de entidade por CPF
    test_cpfs = ["964.128.440-15", "96412844015", "000.000.000-00"]
    print("\nIdentificação de entidade por CPF:")
    for cpf in test_cpfs:
        entity = SRDADatabase.parse_cpf_to_entity(cpf)
        print(f"  {cpf} -> {entity.value if entity else 'NÃO IDENTIFICADO'}")
    
    # Estatísticas iniciais
    stats = db.get_statistics()
    print(f"\nEstatísticas do banco:")
    print(f"  Total de documentos: {stats['total_documents']}")
    print(f"  Total de transações: {stats['total_transactions']}")
    print(f"  Total de matches: {stats['total_matches']}")
    
    db.close()
    print("\n[OK] Banco de dados inicializado com sucesso!")
