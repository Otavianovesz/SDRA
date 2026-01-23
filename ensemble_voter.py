"""
Ensemble Voter Module
=====================
Implementação de algoritmos de votação sofisticada para extração de dados.

Inclui:
1. Weighted Majority Voting
2. Levenshtein Distance & Fuzzy Matching
3. ROVER (Recognizer Output Voting Error Reduction) - Simplificado
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import difflib

# Tenta importar rapidfuzz para performance, fallback para difflib
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VoteCandidate:
    value: str
    confidence: float
    source: str
    weight: float = 1.0


class EnsembleVoter:
    """
    Motor de votação para combinar extrações de múltiplas fontes.
    """
    
    def __init__(self):
        # Pesos padrão por fonte (configurável)
        self.source_weights = {
            "extract_hybrid": 2.0,      # VLM+OCR Híbrido (Tier 4)
            "paddle_ocr": 1.5,          # PaddleOCR puro (Tier 2.5)
            "gliner": 1.2,              # GLiNER NER (Tier 2)
            "regex_specific": 1.0,      # Regex Específico
            "regex_fallback": 0.5,      # Regex Genérico
            "tesseract": 0.8,           # OCR Legacy
            "layout": 0.7,              # Heurística de posição
            "filename": 0.9,            # Nome do arquivo (curado)
            "qwen2_vlm": 0.6            # VLM puro (hallucinates sometimes)
        }

    def calculate_similarity(self, s1: str, s2: str) -> float:
        """Calcula similaridade entre duas strings (0.0 a 1.0)."""
        if not s1 or not s2:
            return 0.0
        
        s1 = s1.upper().strip()
        s2 = s2.upper().strip()
        
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(s1, s2) / 100.0
        else:
            return difflib.SequenceMatcher(None, s1, s2).ratio()

    def weighted_vote(self, candidates: List[VoteCandidate]) -> Optional[VoteCandidate]:
        """
        Seleciona o melhor candidato baseado em confiança ponderada e consenso.
        """
        if not candidates:
            return None
            
        if len(candidates) == 1:
            return candidates[0]
            
        # 1. Agrupar candidatos similares (Clustering)
        clusters = []
        processed = set()
        
        for i, c1 in enumerate(candidates):
            if i in processed:
                continue
                
            # Novo cluster
            current_cluster = [c1]
            processed.add(i)
            
            for j, c2 in enumerate(candidates):
                if j in processed:
                    continue
                
                # Se similaridade > 0.85, considera mesmo valor
                if self.calculate_similarity(c1.value, c2.value) > 0.85:
                    current_cluster.append(c2)
                    processed.add(j)
            
            clusters.append(current_cluster)
            
        # 2. Avaliar clusters
        best_cluster = None
        max_score = -1.0
        
        for cluster in clusters:
            # Score do cluster = soma(confiança * peso) dos membros
            cluster_score = 0.0
            
            # Penalidade para divergência dentro do cluster (length variance)
            lengths = [len(c.value) for c in cluster]
            len_variance = max(lengths) - min(lengths)
            length_penalty = 1.0 if len_variance < 3 else 0.8
            
            for cand in cluster:
                # Peso base da fonte
                source_w = self.source_weights.get(cand.source, 1.0)
                # Confiança do extrator
                vote_score = cand.confidence * source_w
                cluster_score += vote_score
            
            cluster_score *= length_penalty
            
            # Boost por multiplicidade (mais fontes concordando = melhor)
            if len(cluster) > 1:
                cluster_score *= 1.2
            
            if cluster_score > max_score:
                max_score = cluster_score
                best_cluster = cluster
        
        if not best_cluster:
            return None
            
        # Retorna o representante do melhor cluster 
        # (preferência pelo mais longo/completo ou fonte mais confiável)
        # Ordena por (Source Weight * Confidence) descerescente
        best_candidate = sorted(
            best_cluster, 
            key=lambda x: self.source_weights.get(x.source, 1.0) * x.confidence,
            reverse=True
        )[0]
        
        # Ajusta confiança final baseada no consenso
        final_confidence = min(0.99, max_score / 2.0) # Normalização heurística
        best_candidate.confidence = final_confidence
        
        return best_candidate

    def rover_alignment(self, text_list: List[str]) -> str:
        """
        Implementação simplificada do ROVER para alinhar textos de OCRs diferentes.
        Útil para corrigir erros de caracteres (e.g. 'B0let0' vs 'Boleto').
        """
        if not text_list:
            return ""
        if len(text_list) == 1:
            return text_list[0]
            
        # Usa difflib para encontrar sequencias correspondentes
        # Pega a string mais longa como base
        base = max(text_list, key=len)
        others = [t for t in text_list if t != base]
        
        result = base
        # TODO: Implementar alinhamento token-a-token real para ROVER completo
        # Por enquanto, retorna a string que tem mais caracteres alfanumericos (heurística de limpeza)
        
        best_str = max(text_list, key=lambda s: sum(c.isalnum() for c in s))
        return best_str

# Singleton
_voter_instance = None
def get_ensemble_voter():
    global _voter_instance
    if _voter_instance is None:
        _voter_instance = EnsembleVoter()
    return _voter_instance
