"""
Reconciler - Cruzamento de Dados (Subset Sum)
=============================================
Fase 8: Reconciliação (Steps 111-130)
"""

import logging
from typing import List, Dict, Any, Tuple
from itertools import combinations
import pandas as pd

logger = logging.getLogger(__name__)

class Reconciler:
    
    @staticmethod
    def find_matches(boletos: List[Dict], nfes: List[Dict]) -> List[Dict]:
        """
        Encontra matches entre Boletos e NFEs.
        Suporta 1:1 e N:1 (Várias NFEs somando um Boleto).
        """
        matches = []
        
        # Indexar NFEs por Fornecedor (normalizado) para performance
        nfes_by_supplier = {}
        for nfe in nfes:
            supp = nfe.get('supplier_clean', 'UNKNOWN')
            if supp not in nfes_by_supplier:
                nfes_by_supplier[supp] = []
            nfes_by_supplier[supp].append(nfe)
            
        # Itera Boletos
        for boleto in boletos:
            b_val = boleto.get('amount_cents', 0)
            b_supp = boleto.get('supplier_clean', 'UNKNOWN')
            
            if b_val <= 0:
                continue
                
            candidates = nfes_by_supplier.get(b_supp, [])
            if not candidates:
                continue
                
            # 1. Match Exato (1:1)
            exact_match = None
            for nfe in candidates:
                if nfe.get('amount_cents') == b_val:
                    exact_match = nfe
                    break
            
            if exact_match:
                matches.append({
                    "type": "1:1",
                    "boleto_id": boleto['id'],
                    "nfe_ids": [exact_match['id']],
                    "amount": b_val
                })
                # Remove nfe dos candidatos para não reusar (simplificado)
                candidates.remove(exact_match) 
                continue
                
            # 2. Match Subset Sum (N:1) - CUIDADO: Custo Exponencial
            # Limita candidatos (Step 121)
            if len(candidates) > 12:
                candidates = candidates[:12] # Heurística de segurança
                
            # Tenta combinações de 2 a 5 notas
            found_subset = False
            for r in range(2, min(6, len(candidates) + 1)):
                if found_subset: break
                
                for subset in combinations(candidates, r):
                    subset_sum = sum(n.get('amount_cents', 0) for n in subset)
                    
                    if subset_sum == b_val:
                        matches.append({
                            "type": f"{r}:1",
                            "boleto_id": boleto['id'],
                            "nfe_ids": [n['id'] for n in subset],
                            "amount": b_val
                        })
                        found_subset = True
                        break
        
        return matches
