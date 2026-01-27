"""
Módulo Core de Processamento de Documentos SRDA-Rural.
Contém lógica de Regex robusto, Fuzzy Matching otimizado e Validação Cruzada.
Autor: Engenharia de ML SRDA
Data: 2026-01-26
"""

import re
import unicodedata
from typing import Optional, Dict, Any, List, Tuple
from rapidfuzz import fuzz, process

class OCRSanitizer:
    """
    Motor de limpeza e normalização de texto vindo de OCR ruidoso.
    Implementa as tabelas de substituição e normalização especificadas.
    """
    
    # Mapa de substituição para corrigir números lidos como letras
    # Aplicado apenas em contextos onde se espera um valor numérico
    CHAR_FIX_MAP = str.maketrans({
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1', 'B': '8', 
        'S': '5', 's': '5', 'Z': '2', 'G': '6'
    })

    @staticmethod
    def clean_currency_string(text: str) -> str:
        """
        Prepara uma string candidata a valor monetário.
        Remove o símbolo R$, normaliza espaços e corrige substituições comuns.
        """
        if not text: return ""
        
        # 1. Remove R$ e espaços extras
        clean = re.sub(r'(?i)R\$', '', text).strip()
        
        # 2. Heurística de Correção de Caracteres:
        # Se a string contém caracteres problemáticos (O, I, S) misturados com dígitos,
        # assume-se que é um erro de OCR e aplica-se o mapa.
        # Ex: "1.2OO,50" -> "1.200,50"
        if re.search(r'[OIoDSsZG]', clean, re.IGNORECASE):
            clean = clean.translate(OCRSanitizer.CHAR_FIX_MAP)
            
        return clean

    @staticmethod
    def normalize_company_name(name: str) -> str:
        """
        Normaliza nomes de empresas removendo sufixos legais e ruído.
        Essencial para o sucesso do Fuzzy Matching via token_set_ratio.
        """
        if not name: return ""
        
        # 1. Remover acentos (Unicode Normalization)
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        name = name.lower()
        
        # 2. Remover pontuação (exceto espaços)
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # 3. Remover Sufixos Legais (Lista de Stopwords Jurídicas)
        # A ordem importa: remover os mais longos primeiro para evitar sobras
        legal_suffixes = [
            r'\bcomercio\b', r'\bindustria\b', r'\bservicos\b',
            r'\bltda\b', r'\blimitada\b', 
            r'\bs\s?a\b', r'\bs\/?a\b', r'\bsociedade anonima\b',
            r'\beireli\b', r'\bme\b', r'\bepp\b', r'\bmei\b',
            r'\bcia\b', r'\bcompanhia\b', r'\bsociedade\b'
        ]
        
        regex_clean = '|'.join(legal_suffixes)
        name = re.sub(regex_clean, '', name)
        
        # 4. Colapsar espaços
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

class FinancialRegexRegistry:
    """
    A 'Bíblia' dos padrões Regex para documentos brasileiros.
    Otimizados para alta tolerância a falhas de OCR.
    """
    
    # Valor Monetário: Captura "1.200,00", "1 200.00", "1200,00"
    # Grupo 1: Parte inteira e milhares
    # Grupo 2: Parte decimal (opcional)
    PATTERN_BRL_VALUE = re.compile(
        r'(?P<inteiro>(?:[1-9]\d{0,2}(?:[\.\s]?\d{3})*)|0)' # Milhares com. ou espaço
        r'(?:[\,\.]\s*(?P<decimal>\d{2}))?'                 # Decimal com , ou. e possíveis espaços
    )

    # Data: DD/MM/AAAA com separadores quebrados
    PATTERN_DATE_BR = re.compile(
        r'\b'
        r'(?P<dia>0?[1-9]|\d|3)'      # Dia
        r'[\/\.\-\s]+'                        # Separador ruidoso
        r'(?P<mes>0?[1-9]|1)'            # Mês
        r'[\/\.\-\s]+'                        # Separador ruidoso
        r'(?P<ano>(?:19|20)\d{2})'            # Ano (apenas séculos 20 e 21)
        r'\b'
    )

    # CNPJ: Captura ampla (com ou sem formatação)
    PATTERN_CNPJ_CANDIDATE = re.compile(
        r'(?<!\d)(\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\/\s]?\d{4}[-\s]?\d{2})(?!\d)'
    )

    @staticmethod
    def validate_cnpj_checksum(cnpj: str) -> bool:
        """
        Validação matemática (Módulo 11) do CNPJ.
        Filtro final para remover falsos positivos do Regex.
        """
        # Limpar tudo que não é dígito
        numbers = [int(digit) for digit in cnpj if digit.isdigit()]
        
        if len(numbers) != 14: return False
        if len(set(numbers)) == 1: return False # Rejeita 1111...

        # Cálculo do primeiro dígito verificador
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum1 = sum(w * n for w, n in zip(weights1, numbers[:12]))
        r1 = sum1 % 11
        d1 = 0 if r1 < 2 else 11 - r1
        if d1 != numbers[12]: return False

        # Cálculo do segundo dígito verificador
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum2 = sum(w * n for w, n in zip(weights2, numbers[:13]))
        r2 = sum2 % 11
        d2 = 0 if r2 < 2 else 11 - r2
        if d2 != numbers[13]: return False

        return True

class DocumentProcessor:
    """
    Orquestrador da validação cruzada e extração.
    """
    def __init__(self, master_db: Dict[str, str]):
        """
        master_db: Dicionário {Nome Normalizado: CNPJ Limpo}
        """
        self.master_db = master_db
        # Inverte o DB para busca por CNPJ {CNPJ: Nome}
        self.cnpj_index = {v: k for k, v in master_db.items()}
        # Lista de nomes para Fuzzy Search
        self.supplier_names = list(master_db.keys())

    def cross_validate_entity(self, extracted_name: str, extracted_cnpj: str) -> Dict[str, Any]:
        """
        Executa a lógica de validação cruzada (Cross-Reference Lock).
        """
        # 1. Normalização
        clean_name = OCRSanitizer.normalize_company_name(extracted_name)
        clean_cnpj_input = re.sub(r'\D', '', extracted_cnpj) if extracted_cnpj else None
        
        result = {
            "status": "REJECTED",
            "confidence": 0.0,
            "final_cnpj": None,
            "final_name": None,
            "method": "NONE",
            "review_needed": False
        }

        # 2. Validação via Chave Primária (CNPJ)
        cnpj_valid = False
        if clean_cnpj_input and FinancialRegexRegistry.validate_cnpj_checksum(clean_cnpj_input):
            cnpj_valid = True
            if clean_cnpj_input in self.cnpj_index:
                # CNPJ existe no mestre. Verificar consistência do nome.
                db_name = self.cnpj_index[clean_cnpj_input]
                # Usa token_set_ratio para ignorar ordem e ruído extra
                name_match_score = fuzz.token_set_ratio(clean_name, db_name)
                
                if name_match_score >= 80:
                    return {
                        "status": "APPROVED",
                        "confidence": 1.0,
                        "final_cnpj": clean_cnpj_input,
                        "final_name": db_name,
                        "method": "CNPJ_EXACT_MATCH",
                        "review_needed": False
                    }
                else:
                    # CNPJ bate, mas nome diverge muito. Confiar no CNPJ (Chave Forte) mas alertar.
                    return {
                        "status": "WARNING",
                        "confidence": 0.9, # Alta confiança na entidade, baixa no OCR do nome
                        "final_cnpj": clean_cnpj_input,
                        "final_name": db_name,
                        "method": "CNPJ_PRIORITY_NAME_MISMATCH",
                        "review_needed": True
                    }

        # 3. Fallback: Busca via Fuzzy no Nome (se CNPJ falhou ou não existe)
        if clean_name:
            # extractOne retorna (match, score, index)
            match_result = process.extractOne(
                clean_name, 
                self.supplier_names, 
                scorer=fuzz.token_set_ratio
            )
            
            if match_result:
                match, score, _ = match_result
            
                if score >= 95: # Limiar alto para inferência puramente nominal
                    inferred_cnpj = self.master_db[match]
                    return {
                        "status": "APPROVED",
                        "confidence": score / 100.0,
                        "final_cnpj": inferred_cnpj,
                        "final_name": match,
                        "method": "FUZZY_NAME_INFERENCE",
                        "review_needed": False
                    }
        
        return result
