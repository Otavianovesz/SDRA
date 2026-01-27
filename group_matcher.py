"""
SRDA-Rural GroupMatcher Engine
==============================
Motor de Agrupamento para Reconciliação Financeira

Vincula documentos relacionados (NFe + Boleto + Comprovante) em uma única
transação financeira usando matching por valor e fornecedor.

Arquitetura (Vol I, Seção 6 - Conciliação):
- Documentos de mesma transação compartilham um link_id
- Matching por: valor exato/similar + fornecedor similar
- Desambiguação via Gemini quando múltiplos matches

Autor: SRDA-Rural Team
Data: 2026-01-27
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES
# ==============================================================================

# Tolerância de valor em centavos (R$ 1,00 para arredondamentos)
VALUE_TOLERANCE_CENTS = 100

# Tolerância de juros (5% do valor original)
INTEREST_TOLERANCE_PERCENT = 0.05

# Similaridade mínima de fornecedor para match
SUPPLIER_SIMILARITY_THRESHOLD = 0.7


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class MatchCandidate:
    """Candidato a match encontrado no banco de dados."""
    doc_id: int
    doc_type: str
    amount_cents: int
    supplier_clean: str
    due_date: Optional[str] = None
    payment_date: Optional[str] = None
    emission_date: Optional[str] = None
    link_id: Optional[int] = None
    confidence: float = 0.0
    match_reason: str = ""


@dataclass
class DocumentGroup:
    """Grupo de documentos relacionados (uma transação financeira)."""
    link_id: int
    documents: List[MatchCandidate] = field(default_factory=list)
    master_date: Optional[str] = None
    status: str = "PENDING"  # PENDING, MATCHED, RECONCILED, ARCHIVED
    
    @property
    def has_nfe(self) -> bool:
        return any(d.doc_type in ('NFE', 'NFSE') for d in self.documents)
    
    @property
    def has_boleto(self) -> bool:
        return any(d.doc_type == 'BOLETO' for d in self.documents)
    
    @property
    def has_comprovante(self) -> bool:
        return any(d.doc_type == 'COMPROVANTE' for d in self.documents)
    
    @property
    def is_complete(self) -> bool:
        """Grupo completo = NFe + Boleto + Comprovante."""
        return self.has_nfe and self.has_boleto and self.has_comprovante
    
    @property
    def is_partially_matched(self) -> bool:
        """Parcialmente matched = tem match mas falta comprovante."""
        return (self.has_nfe or self.has_boleto) and not self.has_comprovante


# ==============================================================================
# CLASSE PRINCIPAL: GroupMatcher
# ==============================================================================

class GroupMatcher:
    """
    Engine de Agrupamento para Reconciliação Financeira.
    
    Responsabilidades:
    1. Encontrar documentos relacionados por valor + fornecedor
    2. Criar vínculos (link_id) entre documentos
    3. Determinar a Data Soberana do grupo
    4. Gerenciar status do ciclo de vida
    """
    
    def __init__(self, db):
        """
        Inicializa o GroupMatcher.
        
        Args:
            db: Instância do SRDADatabase
        """
        self.db = db
        self._next_link_id = None
    
    def _get_next_link_id(self) -> int:
        """Obtém o próximo link_id disponível."""
        try:
            result = self.db.connection.execute(
                "SELECT COALESCE(MAX(link_id), 0) + 1 FROM transacoes"
            ).fetchone()
            return result[0] if result else 1
        except Exception as e:
            logger.warning(f"Error getting next link_id: {e}")
            return 1
    
    def normalize_supplier(self, supplier: str) -> str:
        """
        Normaliza nome de fornecedor para comparação.
        
        Remove caracteres especiais, converte para uppercase,
        e aplica abreviações comuns.
        """
        if not supplier:
            return ""
        
        # Uppercase e limpa espaços
        clean = supplier.upper().strip()
        
        # Remove caracteres especiais
        clean = re.sub(r'[^\w\s]', '', clean)
        
        # Remove palavras comuns que não discriminam
        stopwords = ['LTDA', 'ME', 'EPP', 'EIRELI', 'SA', 'S/A', 'CIA', 'COMERCIO', 
                     'INDUSTRIA', 'SERVICOS', 'DE', 'DO', 'DA', 'DOS', 'DAS', 'E']
        words = clean.split()
        filtered = [w for w in words if w not in stopwords and len(w) > 1]
        
        return ' '.join(filtered)
    
    def calculate_supplier_similarity(self, s1: str, s2: str) -> float:
        """
        Calcula similaridade entre dois nomes de fornecedor.
        
        Usa Jaccard similarity nas palavras normalizadas.
        """
        n1 = self.normalize_supplier(s1)
        n2 = self.normalize_supplier(s2)
        
        if not n1 or not n2:
            return 0.0
        
        # Exact match após normalização
        if n1 == n2:
            return 1.0
        
        # Containment check
        if n1 in n2 or n2 in n1:
            return 0.9
        
        # Jaccard similarity
        words1 = set(n1.split())
        words2 = set(n2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_matching_documents(
        self,
        supplier: str,
        amount_cents: int,
        exclude_doc_id: Optional[int] = None,
        doc_type_filter: Optional[str] = None
    ) -> List[MatchCandidate]:
        """
        Busca documentos que podem ser do mesmo grupo.
        
        Critérios:
        1. Valor exato ou dentro da tolerância de juros
        2. Fornecedor similar (>70% Jaccard)
        
        Args:
            supplier: Nome do fornecedor
            amount_cents: Valor em centavos
            exclude_doc_id: ID do documento a excluir (self)
            doc_type_filter: Filtrar por tipo específico
        
        Returns:
            Lista de candidatos ordenados por confiança
        """
        candidates = []
        
        try:
            # Calcula tolerâncias
            interest_tolerance = int(amount_cents * INTEREST_TOLERANCE_PERCENT)
            total_tolerance = max(VALUE_TOLERANCE_CENTS, interest_tolerance)
            
            min_value = amount_cents - total_tolerance
            max_value = amount_cents + total_tolerance
            
            # Query base
            query = """
                SELECT 
                    d.id, d.doc_type, d.original_path,
                    t.amount_cents, t.supplier_clean, t.due_date, 
                    t.payment_date, t.emission_date, t.link_id
                FROM documentos d
                LEFT JOIN transacoes t ON d.id = t.doc_id
                WHERE t.amount_cents BETWEEN ? AND ?
                  AND d.status NOT IN ('ARCHIVED', 'ERROR')
            """
            params = [min_value, max_value]
            
            if exclude_doc_id:
                query += " AND d.id != ?"
                params.append(exclude_doc_id)
            
            if doc_type_filter:
                query += " AND d.doc_type = ?"
                params.append(doc_type_filter)
            
            results = self.db.connection.execute(query, params).fetchall()
            
            # Processar resultados
            for row in results:
                doc_id, doc_type, path, val, sup, due, pay, emiss, link = row
                
                # Calcular similaridade de fornecedor
                similarity = self.calculate_supplier_similarity(supplier, sup or "")
                
                if similarity >= SUPPLIER_SIMILARITY_THRESHOLD:
                    # Calcular confiança final
                    value_diff = abs(amount_cents - (val or 0))
                    value_confidence = 1.0 - (value_diff / total_tolerance) if total_tolerance > 0 else 1.0
                    
                    confidence = (similarity * 0.5) + (value_confidence * 0.5)
                    
                    candidate = MatchCandidate(
                        doc_id=doc_id,
                        doc_type=doc_type or "UNKNOWN",
                        amount_cents=val or 0,
                        supplier_clean=sup or "",
                        due_date=due,
                        payment_date=pay,
                        emission_date=emiss,
                        link_id=link,
                        confidence=confidence,
                        match_reason=f"Value diff: R${value_diff/100:.2f}, Supplier sim: {similarity:.2%}"
                    )
                    candidates.append(candidate)
            
            # Ordenar por confiança decrescente
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding matching documents: {e}")
        
        return candidates
    
    def create_document_link(self, doc_ids: List[int]) -> int:
        """
        Cria vínculo entre documentos (atribui mesmo link_id).
        
        Args:
            doc_ids: Lista de IDs de documentos a vincular
        
        Returns:
            O link_id atribuído ao grupo
        """
        if not doc_ids:
            return 0
        
        try:
            # Verificar se algum documento já tem link_id
            existing_link = None
            for doc_id in doc_ids:
                result = self.db.connection.execute(
                    "SELECT link_id FROM transacoes WHERE doc_id = ? AND link_id IS NOT NULL",
                    [doc_id]
                ).fetchone()
                if result and result[0]:
                    existing_link = result[0]
                    break
            
            # Usar link existente ou criar novo
            link_id = existing_link or self._get_next_link_id()
            
            # Atualizar todos os documentos com o link_id
            for doc_id in doc_ids:
                self.db.connection.execute(
                    "UPDATE transacoes SET link_id = ? WHERE doc_id = ?",
                    [link_id, doc_id]
                )
            
            logger.info(f"Created document link {link_id} for docs: {doc_ids}")
            return link_id
            
        except Exception as e:
            logger.error(f"Error creating document link: {e}")
            return 0
    
    def get_group_by_link(self, link_id: int) -> Optional[DocumentGroup]:
        """
        Recupera todos os documentos de um grupo pelo link_id.
        
        Args:
            link_id: ID do grupo
        
        Returns:
            DocumentGroup com todos os documentos vinculados
        """
        if not link_id:
            return None
        
        try:
            query = """
                SELECT 
                    d.id, d.doc_type, 
                    t.amount_cents, t.supplier_clean, t.due_date, 
                    t.payment_date, t.emission_date, t.link_id
                FROM documentos d
                LEFT JOIN transacoes t ON d.id = t.doc_id
                WHERE t.link_id = ?
            """
            results = self.db.connection.execute(query, [link_id]).fetchall()
            
            if not results:
                return None
            
            documents = []
            for row in results:
                doc_id, doc_type, val, sup, due, pay, emiss, lid = row
                candidate = MatchCandidate(
                    doc_id=doc_id,
                    doc_type=doc_type or "UNKNOWN",
                    amount_cents=val or 0,
                    supplier_clean=sup or "",
                    due_date=due,
                    payment_date=pay,
                    emission_date=emiss,
                    link_id=lid
                )
                documents.append(candidate)
            
            group = DocumentGroup(link_id=link_id, documents=documents)
            group.master_date = self.determine_master_date(documents)
            
            # Atualizar status baseado na composição
            if group.is_complete:
                group.status = "RECONCILED"
            elif group.is_partially_matched:
                group.status = "MATCHED"
            else:
                group.status = "PENDING"
            
            return group
            
        except Exception as e:
            logger.error(f"Error getting group by link {link_id}: {e}")
            return None
    
    def determine_master_date(self, documents: List[MatchCandidate]) -> Optional[str]:
        """
        Determina a Data Soberana do grupo.
        
        HIERARQUIA ABSOLUTA (Vol I, Seção 6.1):
        1. Data do COMPROVANTE confirmado (payment_date)
        2. Data de VENCIMENTO do boleto (due_date)
        3. Data de EMISSÃO da nota (emission_date)
        
        Args:
            documents: Lista de candidatos do grupo
        
        Returns:
            Data principal no formato YYYY-MM-DD ou None
        """
        # 1. Buscar data de comprovante
        comprovantes = [d for d in documents if d.doc_type == 'COMPROVANTE']
        for comp in comprovantes:
            if comp.payment_date:
                logger.debug(f"Master date from COMPROVANTE: {comp.payment_date}")
                return comp.payment_date
        
        # 2. Buscar data de vencimento do boleto
        boletos = [d for d in documents if d.doc_type == 'BOLETO']
        for bol in boletos:
            if bol.due_date:
                logger.debug(f"Master date from BOLETO due_date: {bol.due_date}")
                return bol.due_date
        
        # 3. Buscar data de emissão da nota
        notas = [d for d in documents if d.doc_type in ('NFE', 'NFSE')]
        for nota in notas:
            if nota.emission_date:
                logger.debug(f"Master date from NFE emission_date: {nota.emission_date}")
                return nota.emission_date
            # Fallback para due_date da nota (algumas notas têm)
            if nota.due_date:
                logger.debug(f"Master date from NFE due_date: {nota.due_date}")
                return nota.due_date
        
        logger.warning("No master date found in group")
        return None
    
    def auto_match_document(self, doc_id: int) -> Optional[int]:
        """
        Tenta encontrar e vincular automaticamente um documento.
        
        Chamado após extração de dados. Busca documentos relacionados
        e cria vínculo automático se confiança alta.
        
        Args:
            doc_id: ID do documento recém-processado
        
        Returns:
            link_id se criou vínculo, None caso contrário
        """
        try:
            # Buscar dados do documento
            query = """
                SELECT t.amount_cents, t.supplier_clean, d.doc_type
                FROM documentos d
                LEFT JOIN transacoes t ON d.id = t.doc_id
                WHERE d.id = ?
            """
            result = self.db.connection.execute(query, [doc_id]).fetchone()
            
            if not result:
                return None
            
            amount, supplier, doc_type = result
            
            if not amount or not supplier:
                logger.debug(f"Doc {doc_id} missing amount or supplier, cannot auto-match")
                return None
            
            # Buscar candidatos
            candidates = self.find_matching_documents(
                supplier=supplier,
                amount_cents=amount,
                exclude_doc_id=doc_id
            )
            
            if not candidates:
                logger.debug(f"No matching candidates found for doc {doc_id}")
                return None
            
            # Filtrar candidatos por confiança alta (>80%)
            high_confidence = [c for c in candidates if c.confidence >= 0.8]
            
            if high_confidence:
                # Se candidato já tem link_id, adicionar a esse grupo
                for candidate in high_confidence:
                    if candidate.link_id:
                        # Adicionar ao grupo existente
                        self.db.connection.execute(
                            "UPDATE transacoes SET link_id = ? WHERE doc_id = ?",
                            [candidate.link_id, doc_id]
                        )
                        logger.info(f"Added doc {doc_id} to existing group {candidate.link_id}")
                        return candidate.link_id
                
                # Criar novo grupo com o melhor candidato
                best = high_confidence[0]
                link_id = self.create_document_link([doc_id, best.doc_id])
                logger.info(f"Created new group {link_id} for docs {doc_id} and {best.doc_id}")
                return link_id
            
            logger.debug(f"No high-confidence matches for doc {doc_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error in auto_match_document for {doc_id}: {e}")
            return None
    
    def get_unmatched_documents(self, doc_type: Optional[str] = None) -> List[MatchCandidate]:
        """
        Lista documentos sem vínculo (orphans).
        
        Args:
            doc_type: Filtrar por tipo (opcional)
        
        Returns:
            Lista de documentos sem link_id
        """
        try:
            query = """
                SELECT 
                    d.id, d.doc_type, 
                    t.amount_cents, t.supplier_clean, t.due_date, 
                    t.payment_date, t.emission_date
                FROM documentos d
                LEFT JOIN transacoes t ON d.id = t.doc_id
                WHERE (t.link_id IS NULL OR t.link_id = 0)
                  AND d.status NOT IN ('ARCHIVED', 'ERROR')
            """
            params = []
            
            if doc_type:
                query += " AND d.doc_type = ?"
                params.append(doc_type)
            
            results = self.db.connection.execute(query, params).fetchall()
            
            documents = []
            for row in results:
                doc_id, dtype, val, sup, due, pay, emiss = row
                candidate = MatchCandidate(
                    doc_id=doc_id,
                    doc_type=dtype or "UNKNOWN",
                    amount_cents=val or 0,
                    supplier_clean=sup or "",
                    due_date=due,
                    payment_date=pay,
                    emission_date=emiss
                )
                documents.append(candidate)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting unmatched documents: {e}")
            return []
    
    def get_all_groups(self, status_filter: Optional[str] = None) -> List[DocumentGroup]:
        """
        Lista todos os grupos de documentos.
        
        Args:
            status_filter: Filtrar por status (opcional)
        
        Returns:
            Lista de DocumentGroup
        """
        try:
            query = "SELECT DISTINCT link_id FROM transacoes WHERE link_id IS NOT NULL AND link_id > 0"
            results = self.db.connection.execute(query).fetchall()
            
            groups = []
            for row in results:
                link_id = row[0]
                group = self.get_group_by_link(link_id)
                if group:
                    if status_filter is None or group.status == status_filter:
                        groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error getting all groups: {e}")
            return []


# ==============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ==============================================================================

def create_group_matcher(db) -> GroupMatcher:
    """Factory function para criar GroupMatcher."""
    return GroupMatcher(db)
