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
    def find_matches(boletos: List[Dict], nfes: List[Dict], 
                     date_tolerance_days: int = 15) -> List[Dict]:
        """
        Encontra matches entre Boletos e NFEs.
        Suporta 1:1 e N:1 (Várias NFEs somando um Boleto).
        
        OTIMIZAÇÃO: Aplica restrição temporal para reduzir explosão combinatória.
        Só considera NFEs emitidas até date_tolerance_days antes do vencimento do boleto.
        
        Args:
            boletos: Lista de boletos
            nfes: Lista de NFEs
            date_tolerance_days: Tolerância em dias para match temporal (default: 15)
        """
        from datetime import datetime, timedelta
        
        def parse_date(d: str) -> datetime:
            if not d:
                return None
            try:
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(d, fmt)
                    except ValueError:
                        continue
            except:
                pass
            return None
        
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
            b_due = parse_date(boleto.get('due_date'))
            
            if b_val <= 0:
                continue
                
            all_candidates = nfes_by_supplier.get(b_supp, [])
            if not all_candidates:
                continue
            
            # OTIMIZAÇÃO: Filtrar por janela temporal
            if b_due and date_tolerance_days > 0:
                min_date = b_due - timedelta(days=date_tolerance_days + 30)  # NFE antes do boleto
                max_date = b_due + timedelta(days=date_tolerance_days)
                
                candidates = []
                for nfe in all_candidates:
                    nfe_date = parse_date(nfe.get('emission_date'))
                    if nfe_date is None:
                        # Se não tem data, inclui por segurança
                        candidates.append(nfe)
                    elif min_date <= nfe_date <= max_date:
                        candidates.append(nfe)
            else:
                candidates = all_candidates
            
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
            if len(candidates) > 10:
                candidates = candidates[:10]  # Heurística de segurança
                
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
