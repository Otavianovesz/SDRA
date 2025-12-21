"""
SRDA-Rural Graph Reconciler
===========================
Motor de Reconciliacao baseado em Teoria dos Grafos (NetworkX)

Arquitetura (Vol I, Secao 4.1.2):
- Documentos = Nós do grafo
- Relacoes = Arestas com pesos (confianca)
- Transacoes = Componentes Conexos (ilhas isoladas)

Este modulo substitui a abordagem linear de if/else por analise matematica.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from itertools import combinations

import networkx as nx
from rapidfuzz import fuzz

from database import SRDADatabase, DocumentType, DocumentStatus, EntityTag, MatchType

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES
# ==============================================================================

# Tolerancia financeira para matching de valores (em centavos)
VALUE_TOLERANCE_CENTS = 5  # R$ 0.05

# Tolerancia percentual para juros/multas
INTEREST_TOLERANCE_PERCENT = 2.0

# Janela temporal maxima (dias) para matching de datas
DATE_WINDOW_DAYS = 45


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class DocumentNode:
    """No do grafo representando um documento."""
    id: int
    doc_type: str
    amount_cents: int
    supplier: Optional[str] = None
    due_date: Optional[str] = None
    payment_date: Optional[str] = None
    emission_date: Optional[str] = None
    entity_tag: Optional[str] = None
    payment_status: str = "UNKNOWN"
    original_path: str = ""


@dataclass
class TransactionIsland:
    """
    Componente Conexo do grafo = Uma transacao financeira completa.
    
    Pode conter:
    - 1+ NF-e/NFS-e
    - 1+ Boletos
    - 0+ Comprovantes
    """
    nodes: List[DocumentNode]
    edges: List[Tuple[int, int, Dict]]
    master_date: Optional[str] = None
    total_notas: int = 0
    total_boletos: int = 0
    total_comprovantes: int = 0
    is_complete: bool = False
    confidence: float = 0.0
    
    def __post_init__(self):
        self._analyze_composition()
    
    def _analyze_composition(self):
        """Analisa a composicao da ilha."""
        for node in self.nodes:
            if node.doc_type == "NFE" or node.doc_type == "NFSE":
                self.total_notas += 1
            elif node.doc_type == "BOLETO":
                self.total_boletos += 1
            elif node.doc_type == "COMPROVANTE":
                self.total_comprovantes += 1
        
        # Ilha completa = tem pelo menos 1 nota E 1 boleto
        self.is_complete = self.total_notas >= 1 and self.total_boletos >= 1
    
    def determine_master_date(self):
        """
        Determina a data principal da ilha.
        
        HIERARQUIA ABSOLUTA (Vol I, Secao 6.1):
        1. Data do COMPROVANTE confirmado (se existir)
        2. Data de VENCIMENTO do boleto
        3. Data de EMISSAO da nota
        """
        # 1. Busca comprovante confirmado
        for node in self.nodes:
            if node.doc_type == "COMPROVANTE" and node.payment_status == "CONFIRMED":
                if node.payment_date:
                    self.master_date = node.payment_date
                    logger.info(f"Master date de COMPROVANTE: {self.master_date}")
                    return
        
        # 2. Busca vencimento do boleto
        for node in self.nodes:
            if node.doc_type == "BOLETO" and node.due_date:
                self.master_date = node.due_date
                logger.info(f"Master date de BOLETO: {self.master_date}")
                return
        
        # 3. Busca emissao da nota
        for node in self.nodes:
            if node.doc_type in ["NFE", "NFSE"] and node.emission_date:
                self.master_date = node.emission_date
                logger.info(f"Master date de NOTA: {self.master_date}")
                return


@dataclass
class ReconciliationResult:
    """Resultado da reconciliacao."""
    total_documents: int = 0
    total_islands: int = 0
    complete_islands: int = 0
    orphan_documents: int = 0
    islands: List[TransactionIsland] = field(default_factory=list)


# ==============================================================================
# CLASSE PRINCIPAL: GraphReconciler
# ==============================================================================

class GraphReconciler:
    """
    Motor de Reconciliacao baseado em Grafos.
    
    Usa NetworkX para resolver relacoes N-para-M entre documentos
    atraves de Componentes Conexos.
    """
    
    def __init__(self, db: SRDADatabase):
        """
        Inicializa o reconciliador.
        
        Args:
            db: Instancia do banco de dados SRDA
        """
        self.db = db
        self.graph = nx.Graph()
        self._nodes_map: Dict[int, DocumentNode] = {}
    
    # ==========================================================================
    # CONSTRUCAO DO GRAFO
    # ==========================================================================
    
    def build_graph(self) -> nx.Graph:
        """
        Constroi o grafo de documentos.
        
        Documentos = Nós
        Relacoes = Arestas com pesos (confianca)
        """
        self.graph = nx.Graph()
        self._nodes_map = {}
        
        # Carrega todos os documentos do banco
        documents = self._load_all_documents()
        
        # Adiciona nos
        for doc in documents:
            node = DocumentNode(
                id=doc["id"],
                doc_type=doc.get("doc_type", "UNKNOWN"),
                amount_cents=doc.get("amount_cents", 0),
                supplier=doc.get("supplier_clean"),
                due_date=doc.get("due_date"),
                payment_date=doc.get("payment_date"),
                emission_date=doc.get("emission_date"),
                entity_tag=doc.get("entity_tag"),
                payment_status=doc.get("payment_status", "UNKNOWN"),
                original_path=doc.get("original_path", "")
            )
            
            self.graph.add_node(
                doc["id"],
                type=node.doc_type,
                amount=node.amount_cents,
                supplier=node.supplier
            )
            self._nodes_map[doc["id"]] = node
        
        logger.info(f"Grafo inicializado com {len(self._nodes_map)} nos")
        
        # Cria arestas baseadas em evidencias
        self._create_edges_by_value()
        self._create_edges_by_supplier()
        self._create_edges_by_installments()
        self._create_edges_by_barcode()
        
        logger.info(f"Grafo tem {self.graph.number_of_edges()} arestas")
        
        return self.graph
    
    def _load_all_documents(self) -> List[Dict]:
        """Carrega todos os documentos com transacoes."""
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.doc_type, d.entity_tag, d.original_path, d.status,
                t.amount_cents, t.supplier_clean, t.due_date, t.payment_date,
                t.emission_date, t.is_scheduled
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.status NOT IN ('RENAMED')
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def _create_edges_by_value(self):
        """Cria arestas entre documentos com valores compativeis."""
        nodes = list(self._nodes_map.values())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Ignora se mesmo tipo
                if node1.doc_type == node2.doc_type:
                    continue
                
                # Ignora se valores zerados
                if node1.amount_cents == 0 or node2.amount_cents == 0:
                    continue
                
                # Match exato
                if abs(node1.amount_cents - node2.amount_cents) <= VALUE_TOLERANCE_CENTS:
                    weight = 0.8
                    self.graph.add_edge(node1.id, node2.id, weight=weight, reason="value_exact")
                
                # Match com tolerancia de juros (2%)
                elif self._within_interest_tolerance(node1.amount_cents, node2.amount_cents):
                    weight = 0.5
                    self.graph.add_edge(node1.id, node2.id, weight=weight, reason="value_interest")
    
    def _create_edges_by_supplier(self):
        """Cria arestas entre documentos do mesmo fornecedor."""
        # Agrupa por fornecedor
        by_supplier: Dict[str, List[DocumentNode]] = {}
        
        for node in self._nodes_map.values():
            if node.supplier:
                supplier_key = node.supplier.upper()[:20]  # Primeiros 20 chars
                if supplier_key not in by_supplier:
                    by_supplier[supplier_key] = []
                by_supplier[supplier_key].append(node)
        
        # Cria arestas entre documentos do mesmo fornecedor
        for supplier, docs in by_supplier.items():
            if len(docs) > 1:
                notas = [d for d in docs if d.doc_type in ["NFE", "NFSE"]]
                boletos = [d for d in docs if d.doc_type == "BOLETO"]
                
                for nota in notas:
                    for boleto in boletos:
                        if not self.graph.has_edge(nota.id, boleto.id):
                            self.graph.add_edge(nota.id, boleto.id, weight=0.3, reason="supplier")
    
    def _create_edges_by_installments(self):
        """Cria arestas baseadas na tabela de duplicatas."""
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT dup.nfe_id, dup.amount_cents, dup.due_date, dup.seq_num
            FROM duplicatas dup
            WHERE dup.reconciled = 0
        """)
        
        installments = cursor.fetchall()
        
        for inst in installments:
            nfe_id = inst["nfe_id"]
            amount = inst["amount_cents"]
            due_date = inst["due_date"]
            
            # Busca boletos com mesmo valor e data proxima
            for node in self._nodes_map.values():
                if node.doc_type == "BOLETO":
                    if abs(node.amount_cents - amount) <= VALUE_TOLERANCE_CENTS:
                        if not self.graph.has_edge(nfe_id, node.id):
                            self.graph.add_edge(nfe_id, node.id, weight=0.9, reason="installment")
    
    def _create_edges_by_barcode(self):
        """Cria arestas entre boletos e comprovantes com mesmo codigo de barras."""
        # Busca documentos com linha digitavel
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT d.id, t.digitable_line
            FROM documentos d
            JOIN transacoes t ON d.id = t.doc_id
            WHERE t.digitable_line IS NOT NULL AND t.digitable_line != ''
        """)
        
        docs_with_barcode = cursor.fetchall()
        
        # Agrupa por linha digitavel
        by_barcode: Dict[str, List[int]] = {}
        for doc in docs_with_barcode:
            barcode = doc["digitable_line"][:20] if doc["digitable_line"] else None
            if barcode:
                if barcode not in by_barcode:
                    by_barcode[barcode] = []
                by_barcode[barcode].append(doc["id"])
        
        # Cria arestas entre documentos com mesmo codigo
        for barcode, doc_ids in by_barcode.items():
            if len(doc_ids) > 1:
                for i, id1 in enumerate(doc_ids):
                    for id2 in doc_ids[i+1:]:
                        if not self.graph.has_edge(id1, id2):
                            self.graph.add_edge(id1, id2, weight=1.0, reason="barcode")
    
    def _within_interest_tolerance(self, value1: int, value2: int) -> bool:
        """Verifica se valores estao dentro da tolerancia de juros."""
        larger = max(value1, value2)
        smaller = min(value1, value2)
        diff_percent = ((larger - smaller) / smaller) * 100
        return diff_percent <= INTEREST_TOLERANCE_PERCENT
    
    # ==========================================================================
    # RECONCILIACAO VIA COMPONENTES CONEXOS
    # ==========================================================================
    
    def reconcile(self) -> ReconciliationResult:
        """
        Executa a reconciliacao via Componentes Conexos.
        
        Cada componente conexo representa uma "ilha" de documentos
        relacionados = uma transacao financeira completa.
        """
        result = ReconciliationResult()
        result.total_documents = len(self._nodes_map)
        
        # Encontra componentes conexos
        components = list(nx.connected_components(self.graph))
        result.total_islands = len(components)
        
        logger.info(f"Encontradas {len(components)} ilhas de transacao")
        
        for component in components:
            # Extrai subgrafo
            subgraph = self.graph.subgraph(component)
            
            # Cria ilha
            nodes = [self._nodes_map[node_id] for node_id in component]
            edges = [(u, v, self.graph[u][v]) for u, v in subgraph.edges()]
            
            island = TransactionIsland(nodes=nodes, edges=edges)
            island.determine_master_date()
            island.confidence = self._calculate_island_confidence(island)
            
            result.islands.append(island)
            
            if island.is_complete:
                result.complete_islands += 1
            else:
                result.orphan_documents += len(island.nodes)
        
        logger.info(f"Ilhas completas: {result.complete_islands}, Orfaos: {result.orphan_documents}")
        
        return result
    
    def _calculate_island_confidence(self, island: TransactionIsland) -> float:
        """Calcula a confianca de uma ilha."""
        if not island.edges:
            return 0.0
        
        # Media ponderada dos pesos das arestas
        total_weight = sum(edge[2].get("weight", 0) for edge in island.edges)
        return total_weight / len(island.edges)
    
    # ==========================================================================
    # SUBSET SUM PARA AGRUPAMENTOS
    # ==========================================================================
    
    def find_subset_sum_match(
        self, 
        target_value: int, 
        candidates: List[DocumentNode],
        tolerance: int = VALUE_TOLERANCE_CENTS
    ) -> Optional[List[DocumentNode]]:
        """
        Encontra subconjunto de notas que somam ao valor do boleto.
        
        Resolve o problema de N Notas = 1 Boleto.
        
        Args:
            target_value: Valor alvo (boleto) em centavos
            candidates: Lista de notas candidatas
            tolerance: Tolerancia em centavos
            
        Returns:
            Lista de notas que somam ao target, ou None
        """
        if not candidates:
            return None
        
        # Limita para evitar explosao combinatoria
        max_combinations = 5
        
        for r in range(1, min(len(candidates) + 1, max_combinations + 1)):
            for combo in combinations(candidates, r):
                total = sum(c.amount_cents for c in combo)
                if abs(total - target_value) <= tolerance:
                    return list(combo)
        
        return None
    
    # ==========================================================================
    # VINCULACAO MANUAL
    # ==========================================================================
    
    def force_link(
        self, 
        source_id: int, 
        target_id: int, 
        reason: str = "manual"
    ) -> bool:
        """
        Forca um vinculo manual entre dois documentos.
        
        Usado quando o usuario arrasta um documento sobre outro na GUI.
        """
        if source_id not in self._nodes_map or target_id not in self._nodes_map:
            return False
        
        # Adiciona aresta com peso maximo
        self.graph.add_edge(source_id, target_id, weight=1.0, reason=reason)
        
        # Persiste no banco
        self.db.insert_match(
            parent_doc_id=source_id,
            child_doc_id=target_id,
            match_type=MatchType.MANUAL,
            confidence=1.0
        )
        
        logger.info(f"Vinculo manual criado: {source_id} -> {target_id}")
        return True
    
    def remove_link(self, source_id: int, target_id: int) -> bool:
        """Remove um vinculo entre dois documentos."""
        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)
            logger.info(f"Vinculo removido: {source_id} -> {target_id}")
            return True
        return False
    
    # ==========================================================================
    # PERSISTENCIA
    # ==========================================================================
    
    def persist_reconciliation(self, result: ReconciliationResult):
        """Persiste o resultado da reconciliacao no banco."""
        for island in result.islands:
            if island.is_complete and island.confidence >= 0.7:
                # Cria matches entre documentos da ilha
                for node1 in island.nodes:
                    for node2 in island.nodes:
                        if node1.id < node2.id:  # Evita duplicatas
                            self.db.insert_match(
                                parent_doc_id=node1.id,
                                child_doc_id=node2.id,
                                match_type=MatchType.EXACT if island.confidence > 0.9 else MatchType.FUZZY,
                                confidence=island.confidence
                            )
                
                # Atualiza status dos documentos
                for node in island.nodes:
                    self.db.update_document_status(
                        doc_id=node.id,
                        status=DocumentStatus.RECONCILED
                    )
        
        logger.info(f"Reconciliacao persistida: {result.complete_islands} ilhas completas")
    
    # ==========================================================================
    # VISUALIZACAO
    # ==========================================================================
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Retorna dados do grafo para visualizacao."""
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            doc_node = self._nodes_map.get(node_id)
            nodes.append({
                "id": node_id,
                "type": attrs.get("type", "UNKNOWN"),
                "amount": attrs.get("amount", 0),
                "supplier": attrs.get("supplier", ""),
                "label": f"{attrs.get('type', '?')} #{node_id}"
            })
        
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": attrs.get("weight", 0),
                "reason": attrs.get("reason", "")
            })
        
        return {"nodes": nodes, "edges": edges}


# ==============================================================================
# TESTE
# ==============================================================================

def test_graph_reconciliation():
    """Testa a reconciliacao baseada em grafos."""
    from database import SRDADatabase
    
    db = SRDADatabase()
    reconciler = GraphReconciler(db)
    
    # Constroi grafo
    reconciler.build_graph()
    
    # Executa reconciliacao
    result = reconciler.reconcile()
    
    print("=" * 60)
    print("RESULTADO DA RECONCILIACAO")
    print("=" * 60)
    print(f"Total de documentos: {result.total_documents}")
    print(f"Total de ilhas: {result.total_islands}")
    print(f"Ilhas completas: {result.complete_islands}")
    print(f"Documentos orfaos: {result.orphan_documents}")
    
    print("\nILHAS DE TRANSACAO:")
    for i, island in enumerate(result.islands):
        print(f"\n  Ilha {i+1}:")
        print(f"    Documentos: {len(island.nodes)}")
        print(f"    Notas: {island.total_notas}, Boletos: {island.total_boletos}, Comprovantes: {island.total_comprovantes}")
        print(f"    Completa: {island.is_complete}")
        print(f"    Master Date: {island.master_date}")
        print(f"    Confianca: {island.confidence:.1%}")
        
        for node in island.nodes:
            print(f"      - [{node.doc_type}] ID={node.id} | R$ {node.amount_cents/100:.2f} | {node.supplier or '-'}")


if __name__ == "__main__":
    test_graph_reconciliation()
