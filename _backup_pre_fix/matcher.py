"""
SRDA-Rural Matcher Module
=========================
Camada de Logica e Reconciliacao (The Matcher)

Este modulo implementa o Motor de Reconciliacao Algoritmica:
- Algoritmo Subset Sum com tolerancia financeira
- Resolucao de vinculos N-para-M (multiplas notas para um boleto)
- Sistema de pontuacao de confianca (Confidence Scoring)
- Suporte a parcelamentos (1 nota, N boletos)

Referencia: Blindagem Cognitiva e Resiliencia via SLM (Secao 4)
"""

import itertools
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag,
    MatchType
)


# ==============================================================================
# CONFIGURACOES E CONSTANTES
# ==============================================================================

# Tolerancia padrao para matching de valores (em centavos)
# R$ 0,05 = 5 centavos
DEFAULT_TOLERANCE_CENTS = 5

# Tolerancia percentual para juros/multas (2%)
LATE_PAYMENT_TOLERANCE_PERCENT = 0.02

# Janela de tempo para matching de datas (dias)
DATE_WINDOW_DAYS = 45

# Numero maximo de notas para testar combinacoes
MAX_NOTES_COMBINATION = 5

# Pesos para calculo de confianca
WEIGHTS = {
    "value": 0.40,      # Peso do match de valor
    "date": 0.25,       # Peso do match de data
    "supplier": 0.25,   # Peso do match de fornecedor
    "identifier": 0.10, # Peso de identificadores explicitos
}

# Limiar minimo de confianca para sugestao automatica
MIN_CONFIDENCE_THRESHOLD = 0.70

# Limiar de alta confianca (match perfeito)
HIGH_CONFIDENCE_THRESHOLD = 0.95


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class MatchCandidate:
    """Representa um candidato a vinculo entre documentos."""
    boleto_id: int
    boleto_amount: int
    nfe_ids: List[int]
    nfe_amounts: List[int]
    total_nfe_amount: int
    difference_cents: int
    confidence: float
    match_type: MatchType
    
    # Dados adicionais para auditoria
    boleto_date: Optional[str] = None
    nfe_dates: List[str] = field(default_factory=list)
    supplier: Optional[str] = None
    is_installment: bool = False
    installment_info: Optional[str] = None
    
    @property
    def is_exact(self) -> bool:
        """Retorna True se for match exato (diferenca <= tolerancia)."""
        return self.difference_cents <= DEFAULT_TOLERANCE_CENTS
    
    @property
    def is_high_confidence(self) -> bool:
        """Retorna True se a confianca for alta."""
        return self.confidence >= HIGH_CONFIDENCE_THRESHOLD


@dataclass
class ReconciliationResult:
    """Resultado do processo de reconciliacao."""
    total_boletos: int = 0
    matched_boletos: int = 0
    unmatched_boletos: int = 0
    
    total_notas: int = 0
    matched_notas: int = 0
    
    matches_created: int = 0
    high_confidence_matches: int = 0
    low_confidence_matches: int = 0
    
    # Candidatos para revisao manual
    pending_review: List[MatchCandidate] = field(default_factory=list)


# ==============================================================================
# CLASSE PRINCIPAL: ReconciliationEngine
# ==============================================================================

class ReconciliationEngine:
    """
    Motor de Reconciliacao Algoritmica.
    
    Implementa a resolucao do problema de alocacao de recursos financeiros
    usando variacoes do Subset Sum Problem com tolerancia.
    
    Cenarios suportados:
    1. 1 Nota / 1 Boleto (match simples)
    2. N Notas / 1 Boleto (agrupamento)
    3. 1 Nota / N Boletos (parcelamento)
    """
    
    def __init__(
        self,
        db: Optional[SRDADatabase] = None,
        tolerance_cents: int = DEFAULT_TOLERANCE_CENTS
    ):
        """
        Inicializa o motor de reconciliacao.
        
        Args:
            db: Instancia do banco de dados
            tolerance_cents: Tolerancia para matching (em centavos)
        """
        self.db = db or SRDADatabase()
        self.tolerance_cents = tolerance_cents
    
    # ==========================================================================
    # ALGORITMO SUBSET SUM
    # ==========================================================================
    
    def find_subset_sum(
        self,
        target_cents: int,
        candidates: List[Dict[str, Any]],
        max_items: int = MAX_NOTES_COMBINATION
    ) -> List[List[Dict[str, Any]]]:
        """
        Encontra subconjuntos de notas cuja soma seja igual ao target.
        
        Implementa o algoritmo Subset Sum com tolerancia financeira.
        Para conjuntos pequenos (< 50 itens), usa forca bruta otimizada.
        
        Args:
            target_cents: Valor alvo em centavos
            candidates: Lista de notas candidatas (dict com 'id' e 'amount_cents')
            max_items: Numero maximo de itens por combinacao
            
        Returns:
            Lista de combinacoes validas (cada uma e uma lista de notas)
        """
        valid_combinations = []
        
        # Limita o numero de candidatos para performance
        if len(candidates) > 50:
            # Ordena por proximidade do valor alvo
            candidates = sorted(
                candidates,
                key=lambda x: abs(x['amount_cents'] - target_cents)
            )[:50]
        
        # Testa combinacoes de 1 a max_items notas
        for r in range(1, min(max_items + 1, len(candidates) + 1)):
            for combo in itertools.combinations(candidates, r):
                total = sum(c['amount_cents'] for c in combo)
                
                # Verifica se a soma esta dentro da tolerancia
                if abs(total - target_cents) <= self.tolerance_cents:
                    valid_combinations.append(list(combo))
        
        # Ordena por numero de itens (prefere menos notas)
        valid_combinations.sort(key=len)
        
        return valid_combinations
    
    def find_installment_match(
        self,
        boleto_amount: int,
        nfe_amount: int,
        installments: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Verifica se um boleto corresponde a uma parcela de NF-e.
        
        Args:
            boleto_amount: Valor do boleto em centavos
            nfe_amount: Valor total da NF-e em centavos
            installments: Lista de duplicatas da NF-e
            
        Returns:
            Duplicata correspondente ou None
        """
        for inst in installments:
            inst_amount = inst['amount_cents']
            if abs(inst_amount - boleto_amount) <= self.tolerance_cents:
                return inst
        
        # Fallback: verifica divisores comuns (50%, 33%, etc.)
        common_fractions = [2, 3, 4, 5, 6, 10, 12]
        for divisor in common_fractions:
            expected = nfe_amount // divisor
            if abs(expected - boleto_amount) <= (self.tolerance_cents * 10):
                return {
                    'seq_num': 0,  # Desconhecido
                    'amount_cents': boleto_amount,
                    'inferred': True,
                    'total_parts': divisor
                }
        
        return None
    
    # ==========================================================================
    # CALCULO DE CONFIANCA
    # ==========================================================================
    
    def calculate_value_score(
        self,
        boleto_amount: int,
        nfe_total: int
    ) -> float:
        """
        Calcula o score de match de valor.
        
        Args:
            boleto_amount: Valor do boleto em centavos
            nfe_total: Soma dos valores das notas em centavos
            
        Returns:
            Score de 0.0 a 1.0
        """
        if boleto_amount == 0:
            return 0.0
        
        difference = abs(boleto_amount - nfe_total)
        
        # Match exato
        if difference <= self.tolerance_cents:
            return 1.0
        
        # Decaimento exponencial baseado na diferenca percentual
        percent_diff = difference / boleto_amount
        
        if percent_diff <= 0.02:  # Ate 2% (juros/multas)
            return 0.85
        elif percent_diff <= 0.05:  # Ate 5%
            return 0.70
        elif percent_diff <= 0.10:  # Ate 10%
            return 0.50
        else:
            return 0.0
    
    def calculate_date_score(
        self,
        boleto_date: Optional[str],
        nfe_date: Optional[str]
    ) -> float:
        """
        Calcula o score de match de datas.
        
        Regra: Data de vencimento do boleto deve ser >= data de emissao
        da nota e <= data de emissao + 45 dias.
        
        Args:
            boleto_date: Data do boleto (ISO8601)
            nfe_date: Data da nota (ISO8601)
            
        Returns:
            Score de 0.0 a 1.0
        """
        if not boleto_date or not nfe_date:
            return 0.5  # Neutro se dados ausentes
        
        try:
            bol_dt = datetime.fromisoformat(boleto_date)
            nfe_dt = datetime.fromisoformat(nfe_date)
            
            # Diferenca em dias
            diff_days = (bol_dt - nfe_dt).days
            
            # Boleto deve ser posterior a emissao da nota
            if diff_days < 0:
                return 0.3
            
            # Dentro da janela de 45 dias
            if diff_days <= DATE_WINDOW_DAYS:
                return 1.0
            
            # Decaimento apos 45 dias
            if diff_days <= 90:
                return 0.7
            elif diff_days <= 180:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def calculate_confidence(
        self,
        boleto_amount: int,
        nfe_total: int,
        boleto_date: Optional[str] = None,
        nfe_dates: List[str] = None,
        supplier_match: bool = True
    ) -> float:
        """
        Calcula o score de confianca total para um match.
        
        Formula: S = (w_v * S_valor) + (w_d * S_data) + (w_f * S_fornecedor) + (w_id * S_id)
        
        Args:
            boleto_amount: Valor do boleto
            nfe_total: Soma dos valores das notas
            boleto_date: Data do boleto
            nfe_dates: Lista de datas das notas
            supplier_match: Se o fornecedor foi verificado
            
        Returns:
            Score de confianca (0.0 a 1.0)
        """
        # Score de valor (peso maior)
        score_value = self.calculate_value_score(boleto_amount, nfe_total)
        
        # Score de data (usa a primeira data disponivel)
        nfe_date = nfe_dates[0] if nfe_dates else None
        score_date = self.calculate_date_score(boleto_date, nfe_date)
        
        # Score de fornecedor (assumimos ja filtrado por fornecedor)
        score_supplier = 1.0 if supplier_match else 0.5
        
        # Score de identificador (placeholder - seria check de numero no boleto)
        score_id = 0.5
        
        # Calculo ponderado
        confidence = (
            WEIGHTS["value"] * score_value +
            WEIGHTS["date"] * score_date +
            WEIGHTS["supplier"] * score_supplier +
            WEIGHTS["identifier"] * score_id
        )
        
        return round(confidence, 3)
    
    # ==========================================================================
    # RECONCILIACAO PRINCIPAL
    # ==========================================================================
    
    def get_unreconciled_boletos(self) -> List[Dict[str, Any]]:
        """Retorna boletos que ainda nao foram reconciliados."""
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.original_path, d.entity_tag,
                t.amount_cents, t.due_date, t.supplier_clean, t.entidade_pagadora
            FROM documentos d
            JOIN transacoes t ON d.id = t.doc_id
            WHERE d.doc_type = 'BOLETO'
              AND d.status != 'RECONCILED'
              AND t.is_scheduled = 0
            ORDER BY t.amount_cents DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_unreconciled_notas(
        self,
        supplier: Optional[str] = None,
        entity: Optional[EntityTag] = None
    ) -> List[Dict[str, Any]]:
        """
        Retorna notas fiscais que ainda nao foram reconciliadas.
        
        Args:
            supplier: Filtrar por fornecedor
            entity: Filtrar por entidade (VG ou MV)
            
        Returns:
            Lista de notas com seus valores
        """
        cursor = self.db.connection.cursor()
        
        query = """
            SELECT 
                d.id, d.original_path, d.doc_type, d.entity_tag,
                t.amount_cents, t.emission_date, t.due_date, t.supplier_clean
            FROM documentos d
            JOIN transacoes t ON d.id = t.doc_id
            WHERE d.doc_type IN ('NFE', 'NFSE')
              AND d.status != 'RECONCILED'
        """
        params = []
        
        if supplier:
            query += " AND t.supplier_clean = ?"
            params.append(supplier)
        
        if entity:
            query += " AND t.entidade_pagadora = ?"
            params.append(entity.value)
        
        query += " ORDER BY t.amount_cents DESC"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def find_matches_for_boleto(
        self,
        boleto: Dict[str, Any]
    ) -> List[MatchCandidate]:
        """
        Encontra possiveis matches para um boleto.
        
        Args:
            boleto: Dados do boleto
            
        Returns:
            Lista de candidatos ordenados por confianca
        """
        candidates = []
        boleto_amount = boleto.get('amount_cents', 0)
        boleto_date = boleto.get('due_date')
        boleto_entity = boleto.get('entidade_pagadora')
        supplier = boleto.get('supplier_clean')
        
        if boleto_amount <= 0:
            return candidates
        
        # Busca notas do mesmo fornecedor e entidade
        entity = EntityTag(boleto_entity) if boleto_entity else None
        notas = self.get_unreconciled_notas(supplier=supplier, entity=entity)
        
        if not notas:
            # Tenta sem filtro de fornecedor
            notas = self.get_unreconciled_notas(entity=entity)
        
        if not notas:
            return candidates
        
        # ==========================================
        # CENARIO 1: Match Simples (1 Nota = 1 Boleto)
        # ==========================================
        for nota in notas:
            nota_amount = nota.get('amount_cents', 0)
            difference = abs(boleto_amount - nota_amount)
            
            if difference <= self.tolerance_cents:
                confidence = self.calculate_confidence(
                    boleto_amount=boleto_amount,
                    nfe_total=nota_amount,
                    boleto_date=boleto_date,
                    nfe_dates=[nota.get('emission_date') or nota.get('due_date')],
                    supplier_match=(supplier is not None)
                )
                
                candidates.append(MatchCandidate(
                    boleto_id=boleto['id'],
                    boleto_amount=boleto_amount,
                    nfe_ids=[nota['id']],
                    nfe_amounts=[nota_amount],
                    total_nfe_amount=nota_amount,
                    difference_cents=difference,
                    confidence=confidence,
                    match_type=MatchType.EXACT,
                    boleto_date=boleto_date,
                    nfe_dates=[nota.get('emission_date')],
                    supplier=supplier
                ))
        
        # ==========================================
        # CENARIO 2: Agrupamento (N Notas = 1 Boleto)
        # ==========================================
        if len(notas) > 1:
            notas_dict = [
                {'id': n['id'], 'amount_cents': n.get('amount_cents', 0), 'data': n}
                for n in notas
            ]
            
            combinations = self.find_subset_sum(boleto_amount, notas_dict)
            
            for combo in combinations[:5]:  # Limita a 5 combinacoes
                nfe_ids = [c['id'] for c in combo]
                nfe_amounts = [c['amount_cents'] for c in combo]
                total = sum(nfe_amounts)
                difference = abs(boleto_amount - total)
                
                # Evita duplicar matches simples
                if len(combo) == 1:
                    continue
                
                nfe_dates = [
                    c['data'].get('emission_date') or c['data'].get('due_date')
                    for c in combo
                ]
                
                confidence = self.calculate_confidence(
                    boleto_amount=boleto_amount,
                    nfe_total=total,
                    boleto_date=boleto_date,
                    nfe_dates=nfe_dates,
                    supplier_match=(supplier is not None)
                )
                
                # Penaliza levemente combinacoes com muitos itens
                confidence -= 0.02 * (len(combo) - 1)
                confidence = max(confidence, 0)
                
                candidates.append(MatchCandidate(
                    boleto_id=boleto['id'],
                    boleto_amount=boleto_amount,
                    nfe_ids=nfe_ids,
                    nfe_amounts=nfe_amounts,
                    total_nfe_amount=total,
                    difference_cents=difference,
                    confidence=confidence,
                    match_type=MatchType.FUZZY if difference > 0 else MatchType.EXACT,
                    boleto_date=boleto_date,
                    nfe_dates=nfe_dates,
                    supplier=supplier
                ))
        
        # ==========================================
        # CENARIO 3: Parcelamento (1 Nota > N Boletos)
        # ==========================================
        for nota in notas:
            nota_amount = nota.get('amount_cents', 0)
            
            # Se o boleto e menor que a nota, pode ser parcela
            if boleto_amount < nota_amount:
                # Busca duplicatas da nota
                installments = self.db.get_installments_by_nfe(nota['id'])
                
                if installments:
                    match_inst = self.find_installment_match(
                        boleto_amount, nota_amount, installments
                    )
                    
                    if match_inst:
                        total_parts = len(installments)
                        seq_num = match_inst.get('seq_num', 0)
                        
                        confidence = self.calculate_confidence(
                            boleto_amount=boleto_amount,
                            nfe_total=boleto_amount,  # Para parcela, o proprio valor
                            boleto_date=boleto_date,
                            nfe_dates=[nota.get('emission_date')],
                            supplier_match=(supplier is not None)
                        )
                        
                        candidates.append(MatchCandidate(
                            boleto_id=boleto['id'],
                            boleto_amount=boleto_amount,
                            nfe_ids=[nota['id']],
                            nfe_amounts=[nota_amount],
                            total_nfe_amount=boleto_amount,
                            difference_cents=0,
                            confidence=confidence,
                            match_type=MatchType.EXACT,
                            boleto_date=boleto_date,
                            nfe_dates=[nota.get('emission_date')],
                            supplier=supplier,
                            is_installment=True,
                            installment_info=f"PARC_{seq_num}-{total_parts}"
                        ))
        
        # Ordena por confianca (maior primeiro)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def apply_match(self, candidate: MatchCandidate) -> bool:
        """
        Aplica um match, atualizando o banco de dados.
        
        Args:
            candidate: Candidato a vinculo
            
        Returns:
            True se o match foi aplicado com sucesso
        """
        try:
            # Insere vinculos na tabela matches
            for nfe_id in candidate.nfe_ids:
                self.db.insert_match(
                    parent_doc_id=candidate.boleto_id,
                    child_doc_id=nfe_id,
                    match_type=candidate.match_type,
                    confidence=candidate.confidence
                )
            
            # Atualiza status do boleto
            self.db.update_document_status(
                doc_id=candidate.boleto_id,
                status=DocumentStatus.RECONCILED
            )
            
            # Atualiza status das notas
            for nfe_id in candidate.nfe_ids:
                self.db.update_document_status(
                    doc_id=nfe_id,
                    status=DocumentStatus.RECONCILED
                )
            
            return True
            
        except Exception as e:
            print(f"[ERRO] Falha ao aplicar match: {e}")
            return False
    
    def reconcile_all(
        self,
        auto_confirm: bool = True,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    ) -> ReconciliationResult:
        """
        Executa a reconciliacao de todos os boletos pendentes.
        
        Args:
            auto_confirm: Se True, confirma automaticamente matches
                         com confianca >= min_confidence
            min_confidence: Limiar minimo de confianca para auto-confirm
            
        Returns:
            ReconciliationResult com estatisticas
        """
        print("=" * 60)
        print("SRDA-Rural Matcher - Iniciando Reconciliacao")
        print("=" * 60)
        print(f"Tolerancia: R$ {self.tolerance_cents / 100:.2f}")
        print(f"Confianca minima: {min_confidence * 100:.0f}%")
        print(f"Auto-confirm: {'SIM' if auto_confirm else 'NAO'}")
        print("-" * 60)
        
        result = ReconciliationResult()
        
        # Busca boletos pendentes
        boletos = self.get_unreconciled_boletos()
        result.total_boletos = len(boletos)
        
        if not boletos:
            print("\n[INFO] Nenhum boleto pendente para reconciliar.")
            return result
        
        print(f"\nEncontrados {len(boletos)} boletos pendentes")
        
        # Conta notas disponiveis
        notas = self.get_unreconciled_notas()
        result.total_notas = len(notas)
        print(f"Notas fiscais disponiveis: {len(notas)}")
        print("-" * 60)
        
        # Processa cada boleto
        for i, boleto in enumerate(boletos, 1):
            boleto_id = boleto['id']
            boleto_amount = boleto.get('amount_cents', 0)
            boleto_display = SRDADatabase.cents_to_display(boleto_amount)
            
            print(f"\n[{i}/{len(boletos)}] Boleto #{boleto_id}: R$ {boleto_display}")
            
            # Busca candidatos
            candidates = self.find_matches_for_boleto(boleto)
            
            if not candidates:
                print("  -> Nenhum match encontrado")
                result.unmatched_boletos += 1
                continue
            
            # Pega o melhor candidato
            best = candidates[0]
            
            print(f"  -> Melhor match: {len(best.nfe_ids)} nota(s)")
            print(f"     Confianca: {best.confidence * 100:.1f}%")
            print(f"     Diferenca: R$ {best.difference_cents / 100:.2f}")
            
            if best.is_installment:
                print(f"     Parcela: {best.installment_info}")
            
            # Decide se aplica automaticamente
            if auto_confirm and best.confidence >= min_confidence:
                if self.apply_match(best):
                    print("  -> [VINCULADO] Match aplicado automaticamente")
                    result.matched_boletos += 1
                    result.matched_notas += len(best.nfe_ids)
                    result.matches_created += 1
                    
                    if best.is_high_confidence:
                        result.high_confidence_matches += 1
                    else:
                        result.low_confidence_matches += 1
            else:
                print("  -> [PENDENTE] Aguardando revisao manual")
                result.pending_review.append(best)
                result.unmatched_boletos += 1
        
        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO DA RECONCILIACAO")
        print("=" * 60)
        print(f"Boletos processados:    {result.total_boletos}")
        print(f"Boletos vinculados:     {result.matched_boletos}")
        print(f"Boletos pendentes:      {result.unmatched_boletos}")
        print(f"Notas vinculadas:       {result.matched_notas}")
        print(f"Matches criados:        {result.matches_created}")
        print(f"  - Alta confianca:     {result.high_confidence_matches}")
        print(f"  - Baixa confianca:    {result.low_confidence_matches}")
        print(f"Pendentes de revisao:   {len(result.pending_review)}")
        
        return result


# ==============================================================================
# FUNCOES UTILITARIAS
# ==============================================================================

def display_pending_matches(pending: List[MatchCandidate]):
    """Exibe candidatos pendentes de revisao."""
    if not pending:
        print("\nNenhum match pendente de revisao.")
        return
    
    print("\n" + "=" * 60)
    print("MATCHES PENDENTES DE REVISAO MANUAL")
    print("=" * 60)
    
    for i, match in enumerate(pending, 1):
        print(f"\n[{i}] Boleto #{match.boleto_id}")
        print(f"    Valor Boleto: R$ {SRDADatabase.cents_to_display(match.boleto_amount)}")
        print(f"    Notas: {match.nfe_ids}")
        print(f"    Soma Notas: R$ {SRDADatabase.cents_to_display(match.total_nfe_amount)}")
        print(f"    Diferenca: R$ {match.difference_cents / 100:.2f}")
        print(f"    Confianca: {match.confidence * 100:.1f}%")
        print(f"    Tipo: {match.match_type.value}")
        if match.is_installment:
            print(f"    Parcela: {match.installment_info}")


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Cria instancia do motor de reconciliacao
    engine = ReconciliationEngine()
    
    # Executa reconciliacao
    result = engine.reconcile_all(auto_confirm=True, min_confidence=0.70)
    
    # Exibe pendentes
    display_pending_matches(result.pending_review)
    
    # Estatisticas do banco
    print("\n" + "-" * 60)
    print("Estatisticas do Banco:")
    stats = engine.db.get_statistics()
    print(f"  Total de documentos: {stats['total_documents']}")
    print(f"  Total de matches: {stats['total_matches']}")
    print(f"  Matches confirmados: {stats['confirmed_matches']}")
    
    engine.db.close()
    print("\n[OK] Reconciliacao concluida!")
