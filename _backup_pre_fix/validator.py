"""
Validator - Lógica de Negócios e Sanitização
============================================
Fase 8: Validação dos dados extraídos.
"""

import re
import logging
from datetime import datetime, date
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class Validator:
    
    @staticmethod
    def validate_cnpj(cnpj: str) -> bool:
        """
        Valida Dígitos Verificadores do CNPJ (Módulo 11).
        Retorna True se válido.
        """
        if not cnpj:
            return False
            
        # Limpa string
        digits = ''.join(c for c in cnpj if c.isdigit())
        if len(digits) != 14:
            return False
            
        # Evita sequências repetidas (ex: 00000000000000)
        if digits == digits[0] * 14:
            return False
            
        def calculate_digit(digits_slice, weights):
            s = sum(int(d) * w for d, w in zip(digits_slice, weights))
            rem = s % 11
            return 0 if rem < 2 else 11 - rem
            
        # Primeiro Dígito
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        d1 = calculate_digit(digits[:12], weights1)
        if int(digits[12]) != d1:
            return False
            
        # Segundo Dígito
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        d2 = calculate_digit(digits[:13], weights2)
        if int(digits[13]) != d2:
            return False
            
        return True

    @staticmethod
    def fix_cnpj_ocr_errors(cnpj: str) -> Optional[str]:
        """
        Tenta corrigir erros comuns de OCR em CNPJ (B->8, O->0, etc)
        e retorna CNPJ válido se correção funcionar.
        """
        if not cnpj:
            return None
            
        if Validator.validate_cnpj(cnpj):
            return cnpj
            
        # Mapa de substituições
        subs = {
            'B': '8', 'O': '0', 'Q': '0', 'I': '1', 'L': '1',
            'S': '5', 'G': '6', 'Z': '2', '?': '7'
        }
        
        corrected = cnpj.upper()
        for k, v in subs.items():
            corrected = corrected.replace(k, v)
        
        # Reconstrói apenas dígitos
        digits = ''.join(c for c in corrected if c.isdigit())
        
        if len(digits) == 14 and Validator.validate_cnpj(digits):
            # Formata
            return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
            
        return None

    @staticmethod
    def validate_dates(emission: Optional[str], due: Optional[str]) -> Tuple[bool, str]:
        """
        Valida relação entre datas.
        Regra: Vencimento >= Emissão
        Regra: Ano razoável (2020+)
        """
        try:
            em_dt = datetime.fromisoformat(emission).date() if emission else None
            due_dt = datetime.fromisoformat(due).date() if due else None
            today = date.today()
            
            # Valida Emissão
            if em_dt:
                if em_dt.year < 2020:
                    return False, f"Data emissão antiga: {em_dt}"
                if em_dt.year > today.year + 1:
                    return False, f"Data emissão no futuro distante: {em_dt}"
                    
            # Valida Vencimento
            if due_dt:
                if due_dt.year < 2020:
                    return False, f"Data vencimento antiga: {due_dt}"
            
            # Valida Relação
            if em_dt and due_dt:
                if due_dt < em_dt:
                    return False, "Vencimento anterior à emissão"
                    
            return True, "OK"
            
        except ValueError:
            return False, "Formato de data inválido"

    @staticmethod
    def normalize_supplier(raw_name: str, known_suppliers: list) -> str:
        """
        Normaliza nome do fornecedor usando Fuzzy Matching.
        Pode usar rapidfuzz se disponível (importado via ensemble/resource_manager).
        """
        if not raw_name:
            return "DESCONHECIDO"
            
        try:
            from rapidfuzz import process, fuzz
            match = process.extractOne(raw_name, known_suppliers, scorer=fuzz.token_sort_ratio)
            if match:
                best_match, score, idx = match
                if score >= 85:
                    return best_match
                elif score >= 60:
                    return best_match # Poderia marcar warning
        except ImportError:
            pass # Fallback se rapidfuzz faltar (mas check_deps exige)
            
        # Fallback: limpeza básica
        clean = re.sub(r'[^\w\s]', '', raw_name).upper()
        return clean.strip()
