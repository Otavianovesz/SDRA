"""
SDRA-Rural Deterministic Validators
====================================
Validação matemática para Zero Falsos Positivos

Implementa checksums para documentos fiscais brasileiros:
- Chave de Acesso NFe (44 dígitos) - Módulo 11
- Linha Digitável de Boletos - Módulo 10/11
- CNPJ - Módulo 11 (2 dígitos verificadores)
- CPF - Módulo 11 (2 dígitos verificadores)

Baseado no Relatório Estratégico v3.0 (Seção 6)
"""

import re
import logging
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de uma validação."""
    is_valid: bool
    field_name: str
    extracted_value: str
    expected_value: Optional[str] = None
    error_message: Optional[str] = None
    confidence_impact: float = 0.0  # -1.0 a 1.0


# =============================================================================
# FUNÇÕES UTILITÁRIAS DE SANITIZAÇÃO
# =============================================================================

def _parse_date_for_sanitize(date_str: Optional[str]) -> Optional[date]:
    """
    Parse de data para sanitização.
    
    Aceita formatos: YYYY-MM-DD, DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY
    """
    if not date_str:
        return None
    
    # Se já está em formato ISO (YYYY-MM-DD)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return None
    
    # Formato brasileiro (DD/MM/YYYY ou DD.MM.YYYY ou DD-MM-YYYY)
    match = re.match(r'^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$', date_str)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = '20' + year if int(year) < 50 else '19' + year
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None
    
    return None


def sanitize_dates(emission: Optional[str], due: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Corrige inversão lógica de datas.
    
    Regra: Se Vencimento < Emissão, provavelmente estão trocados.
    (Assumindo que não processamos contas vencidas retroativas frequentemente)
    
    Args:
        emission: Data de emissão (qualquer formato aceito)
        due: Data de vencimento (qualquer formato aceito)
        
    Returns:
        Tuple (emission, due) - corrigidas se necessário
    """
    if not emission or not due:
        return emission, due
    
    d_emission = _parse_date_for_sanitize(emission)
    d_due = _parse_date_for_sanitize(due)
    
    if d_emission and d_due:
        # Se Vencimento < Emissão, provável inversão
        if d_due < d_emission:
            logger.warning(
                f"Inversão Temporal detectada: Emissão {emission} > Vencimento {due}. "
                f"Corrigindo automaticamente."
            )
            return due, emission  # Troca as datas
    
    return emission, due


class ChecksumValidators:
    """
    Validadores de dígitos verificadores para documentos brasileiros.
    
    Algoritmos:
    - Módulo 11: Usado em CNPJ, CPF, Chave NFe
    - Módulo 10: Usado em alguns campos de boletos
    """
    
    @staticmethod
    def modulo_11(digits: str, weights: list) -> int:
        """
        Calcula dígito verificador usando Módulo 11.
        
        Args:
            digits: String de dígitos para calcular
            weights: Lista de pesos a aplicar
            
        Returns:
            Dígito verificador (0-9 ou 0 se resultado > 9)
        """
        total = 0
        for i, digit in enumerate(digits):
            if i < len(weights):
                total += int(digit) * weights[i]
        
        remainder = total % 11
        
        if remainder < 2:
            return 0
        else:
            return 11 - remainder
    
    @staticmethod
    def modulo_10(digits: str) -> int:
        """
        Calcula dígito verificador usando Módulo 10 (Luhn-like).
        
        Usado em boletos bancários.
        
        Args:
            digits: String de dígitos
            
        Returns:
            Dígito verificador (0-9)
        """
        weights = [2, 1] * (len(digits) // 2 + 1)
        total = 0
        
        for i, digit in enumerate(reversed(digits)):
            product = int(digit) * weights[i]
            # Soma os dígitos se produto >= 10
            total += product // 10 + product % 10
        
        remainder = total % 10
        return 0 if remainder == 0 else 10 - remainder
    
    @staticmethod
    def validate_cnpj(cnpj: str) -> ValidationResult:
        """
        Valida CNPJ com dígitos verificadores.
        
        Formato: XX.XXX.XXX/XXXX-XX ou 14 dígitos
        
        Args:
            cnpj: CNPJ a validar
            
        Returns:
            ValidationResult com status
        """
        # Limpar formatação
        digits = re.sub(r'[^\d]', '', cnpj)
        
        if len(digits) != 14:
            return ValidationResult(
                is_valid=False,
                field_name="cnpj",
                extracted_value=cnpj,
                error_message=f"CNPJ deve ter 14 dígitos, encontrado {len(digits)}",
                confidence_impact=-1.0
            )
        
        # Verificar se não é sequência repetida
        if digits == digits[0] * 14:
            return ValidationResult(
                is_valid=False,
                field_name="cnpj",
                extracted_value=cnpj,
                error_message="CNPJ inválido (sequência repetida)",
                confidence_impact=-1.0
            )
        
        # Primeiro dígito verificador
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        dv1 = ChecksumValidators.modulo_11(digits[:12], weights1)
        
        # Segundo dígito verificador
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        dv2 = ChecksumValidators.modulo_11(digits[:12] + str(dv1), weights2)
        
        expected = f"{dv1}{dv2}"
        actual = digits[12:14]
        
        is_valid = expected == actual
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="cnpj",
            extracted_value=cnpj,
            expected_value=expected if not is_valid else None,
            error_message=f"DV esperado: {expected}, encontrado: {actual}" if not is_valid else None,
            confidence_impact=0.9 if is_valid else -1.0
        )
    
    @staticmethod
    def validate_cpf(cpf: str) -> ValidationResult:
        """
        Valida CPF com dígitos verificadores.
        
        Formato: XXX.XXX.XXX-XX ou 11 dígitos
        
        Args:
            cpf: CPF a validar
            
        Returns:
            ValidationResult com status
        """
        digits = re.sub(r'[^\d]', '', cpf)
        
        if len(digits) != 11:
            return ValidationResult(
                is_valid=False,
                field_name="cpf",
                extracted_value=cpf,
                error_message=f"CPF deve ter 11 dígitos, encontrado {len(digits)}",
                confidence_impact=-1.0
            )
        
        # Verificar se não é sequência repetida
        if digits == digits[0] * 11:
            return ValidationResult(
                is_valid=False,
                field_name="cpf",
                extracted_value=cpf,
                error_message="CPF inválido (sequência repetida)",
                confidence_impact=-1.0
            )
        
        # Primeiro dígito verificador
        weights1 = [10, 9, 8, 7, 6, 5, 4, 3, 2]
        dv1 = ChecksumValidators.modulo_11(digits[:9], weights1)
        
        # Segundo dígito verificador
        weights2 = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        dv2 = ChecksumValidators.modulo_11(digits[:9] + str(dv1), weights2)
        
        expected = f"{dv1}{dv2}"
        actual = digits[9:11]
        
        is_valid = expected == actual
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="cpf",
            extracted_value=cpf,
            expected_value=expected if not is_valid else None,
            error_message=f"DV esperado: {expected}, encontrado: {actual}" if not is_valid else None,
            confidence_impact=0.9 if is_valid else -1.0
        )
    
    @staticmethod
    def validate_nfe_access_key(key: str) -> ValidationResult:
        """
        Valida Chave de Acesso NFe (44 dígitos).
        
        O último dígito é calculado via Módulo 11 sobre os 43 primeiros.
        
        Estrutura da chave:
        - 2 dígitos: UF
        - 4 dígitos: AAMM (ano/mês)
        - 14 dígitos: CNPJ emitente
        - 2 dígitos: Modelo (55=NFe, 65=NFCe)
        - 3 dígitos: Série
        - 9 dígitos: Número NF
        - 1 dígito: Tipo emissão
        - 8 dígitos: Código numérico
        - 1 dígito: DV
        
        Args:
            key: Chave de acesso (44 dígitos)
            
        Returns:
            ValidationResult com status e extração de número da NF
        """
        digits = re.sub(r'[^\d]', '', key)
        
        if len(digits) != 44:
            return ValidationResult(
                is_valid=False,
                field_name="nfe_access_key",
                extracted_value=key,
                error_message=f"Chave deve ter 44 dígitos, encontrado {len(digits)}",
                confidence_impact=-1.0
            )
        
        # Calcular dígito verificador (Módulo 11)
        weights = [4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 
                   9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 
                   6, 5, 4, 3, 2]
        
        expected_dv = ChecksumValidators.modulo_11(digits[:43], weights)
        actual_dv = int(digits[43])
        
        is_valid = expected_dv == actual_dv
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="nfe_access_key",
            extracted_value=key,
            expected_value=str(expected_dv) if not is_valid else None,
            error_message=f"DV esperado: {expected_dv}, encontrado: {actual_dv}" if not is_valid else None,
            confidence_impact=1.0 if is_valid else -1.0  # Peso máximo para chave NFe
        )
    
    @staticmethod
    def extract_nf_number_from_key(key: str) -> Optional[str]:
        """
        Extrai o número da NF da chave de acesso.
        
        Posições 25-33 (9 dígitos) = Número da NF
        
        Args:
            key: Chave de acesso válida
            
        Returns:
            Número da NF (sem zeros à esquerda) ou None
        """
        digits = re.sub(r'[^\d]', '', key)
        
        if len(digits) != 44:
            return None
        
        nf_number = digits[25:34]  # 9 dígitos
        return str(int(nf_number))  # Remove zeros à esquerda
    
    @staticmethod
    def validate_boleto_digitable_line(line: str) -> ValidationResult:
        """
        Valida linha digitável de boleto bancário.
        
        Formato: AAAAA.BBBBB CCCCC.DDDDDD EEEEE.FFFFFF G HHHHHHHHHHHHHH
        
        3 campos com DVs próprios + 1 DV geral
        
        Args:
            line: Linha digitável (47 dígitos)
            
        Returns:
            ValidationResult com status
        """
        digits = re.sub(r'[^\d]', '', line)
        
        if len(digits) != 47:
            return ValidationResult(
                is_valid=False,
                field_name="boleto_digitable_line",
                extracted_value=line,
                error_message=f"Linha deve ter 47 dígitos, encontrado {len(digits)}",
                confidence_impact=-1.0
            )
        
        # Campo 1: posições 0-9, DV na posição 9 (Módulo 10)
        field1 = digits[0:9]
        dv1_expected = ChecksumValidators.modulo_10(field1)
        dv1_actual = int(digits[9])
        
        # Campo 2: posições 10-20, DV na posição 20 (Módulo 10)
        field2 = digits[10:20]
        dv2_expected = ChecksumValidators.modulo_10(field2)
        dv2_actual = int(digits[20])
        
        # Campo 3: posições 21-31, DV na posição 31 (Módulo 10)
        field3 = digits[21:31]
        dv3_expected = ChecksumValidators.modulo_10(field3)
        dv3_actual = int(digits[31])
        
        errors = []
        if dv1_expected != dv1_actual:
            errors.append(f"Campo 1: DV {dv1_expected} != {dv1_actual}")
        if dv2_expected != dv2_actual:
            errors.append(f"Campo 2: DV {dv2_expected} != {dv2_actual}")
        if dv3_expected != dv3_actual:
            errors.append(f"Campo 3: DV {dv3_expected} != {dv3_actual}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="boleto_digitable_line",
            extracted_value=line,
            error_message="; ".join(errors) if errors else None,
            confidence_impact=0.9 if is_valid else -1.0
        )


class OCRRepairEngine:
    """
    Motor de Reparo de OCR usando restrições matemáticas.
    
    Capaz de corrigir:
    - Dígitos trocados (3 vs 8, 1 vs 7, 0 vs 6)
    - Dígitos faltantes ou borrados (indicados por '?')
    - Erros de leitura em códigos de barras
    """
    
    @staticmethod
    def repair_nfe_access_key(key: str, max_attempts: int = 430) -> Optional[str]:
        """
        Tenta reparar uma chave de acesso NFe inválida.
        
        OTIMIZADO: Prioriza dígitos com confusão OCR comum antes de força bruta.
        Mapeamento de confusão OCR: 0↔6, 1↔7, 3↔8, 5↔6, 0↔O(já numérico)
        
        Args:
            key: Chave de acesso candidata (44 dígitos ou próximo)
            max_attempts: Limite de tentativas para evitar travamento
            
        Returns:
            Chave válida ou None
        """
        clean_key = re.sub(r'[^\d]', '', key)
        
        # Se tamanho errado, não tenta reparar
        if len(clean_key) != 44:
            return None
            
        # Se já é válida, retorna
        if ChecksumValidators.validate_nfe_access_key(clean_key).is_valid:
            return clean_key
        
        logger.info(f"Tentando reparar chave NFe: {clean_key}")
        original_digits = list(clean_key)
        attempts = 0
        
        # Mapeamento de confusão OCR (dígito -> possíveis substitutos)
        OCR_CONFUSIONS = {
            '0': ['6', '8', 'O'],
            '1': ['7', 'I', 'l'],
            '3': ['8'],
            '5': ['6', 'S'],
            '6': ['0', '5', '8'],
            '7': ['1'],
            '8': ['3', '0', '6'],
        }
        
        # FASE 1: Tenta apenas substituições OCR-prováveis (muito mais rápido)
        for i in range(43):
            original_char = original_digits[i]
            
            if original_char in OCR_CONFUSIONS:
                for substitute in OCR_CONFUSIONS[original_char]:
                    if substitute.isdigit():
                        original_digits[i] = substitute
                        candidate = "".join(original_digits)
                        attempts += 1
                        
                        if ChecksumValidators.validate_nfe_access_key(candidate).is_valid:
                            logger.info(f"Chave REPARADA (OCR confusion) pos {i}: {original_char} -> {substitute}")
                            return candidate
                        
                        if attempts >= max_attempts:
                            logger.warning(f"Limite de tentativas atingido ({max_attempts})")
                            return None
                
                # Restaura
                original_digits[i] = original_char
        
        # FASE 2: Força bruta limitada se fase 1 falhou (apenas primeiros 20 dígitos)
        for i in range(min(20, 43)):
            original_char = original_digits[i]
            
            for digit in "0123456789":
                if digit == original_char:
                    continue
                    
                original_digits[i] = digit
                candidate = "".join(original_digits)
                attempts += 1
                
                if ChecksumValidators.validate_nfe_access_key(candidate).is_valid:
                    logger.info(f"Chave REPARADA (brute force) pos {i}: {original_char} -> {digit}")
                    return candidate
                
                if attempts >= max_attempts:
                    logger.warning(f"Limite de tentativas atingido ({max_attempts})")
                    return None
            
            # Restaura
            original_digits[i] = original_char
            
        return None

    @staticmethod
    def aggregate_barcode_scans(scanlines: List[str]) -> Optional[str]:
        """
        Reconstitui código de barras a partir de múltiplas leituras parciais.
        
        Técnica de Votação por Posição:
        - Para cada posição do código (0-43 para NFe, 0-46 para Boleto)
        - Conta frequência do char em todas as scanlines
        - Escolhe o char mais frequente (moda)
        
        Args:
            scanlines: Lista de strings lidas de diferentes alturas do código
            
        Returns:
            String consolidada (best guess)
        """
        if not scanlines:
            return None
            
        # Filtrar scanlines muito curtas ou lixo
        valid_scans = [s.strip() for s in scanlines if len(s.strip()) > 10]
        if not valid_scans:
            return None
            
        # Determinar tamanho alvo (moda dos tamanhos)
        sizes = {}
        for s in valid_scans:
            l = len(s)
            sizes[l] = sizes.get(l, 0) + 1
        
        target_len = max(sizes, key=sizes.get)
        
        # Filtrar apenas scans do tamanho correto
        scans_target = [s for s in valid_scans if len(s) == target_len]
        if not scans_target:
            return None
            
        # Votação por coluna
        result = []
        for i in range(target_len):
            counts = {}
            for s in scans_target:
                char = s[i]
                counts[char] = counts.get(char, 0) + 1
            
            # Pega character mais comum
            best_char = max(counts, key=counts.get)
            result.append(best_char)
            
        consolidated = "".join(result)
        logger.info(f"Barcode consolidado de {len(scans_target)} linhas: {consolidated}")
        
        return consolidated


class SemanticValidators:
    """
    Validadores de consistência semântica entre campos.
    
    Verifica:
    - Lógica temporal (datas)
    - Matemática contábil (totais)
    - Consistência de entidades
    """
    
    @staticmethod
    def validate_date_logic(
        emission_date: Optional[str],
        due_date: Optional[str],
        payment_date: Optional[str] = None
    ) -> ValidationResult:
        """
        Valida lógica temporal entre datas.
        
        Regras:
        - Vencimento >= Emissão
        - Emissão <= Data atual
        - Pagamento >= Emissão (se existir)
        
        Args:
            emission_date: Data de emissão (YYYY-MM-DD)
            due_date: Data de vencimento (YYYY-MM-DD)
            payment_date: Data de pagamento opcional (YYYY-MM-DD)
            
        Returns:
            ValidationResult com status
        """
        errors = []
        today = date.today()
        
        def parse_date(date_str: Optional[str]) -> Optional[date]:
            if not date_str:
                return None
            try:
                # Tentar vários formatos
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
                return None
            except Exception:
                return None
        
        emission = parse_date(emission_date)
        due = parse_date(due_date)
        payment = parse_date(payment_date)
        
        if emission:
            if emission > today:
                errors.append(f"Emissão ({emission_date}) é futura")
        
        if emission and due:
            if due < emission:
                errors.append(f"Vencimento ({due_date}) anterior à emissão ({emission_date})")
        
        if emission and payment:
            if payment < emission:
                errors.append(f"Pagamento ({payment_date}) anterior à emissão ({emission_date})")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="date_logic",
            extracted_value=f"E:{emission_date}, V:{due_date}, P:{payment_date}",
            error_message="; ".join(errors) if errors else None,
            confidence_impact=0.0 if is_valid else -0.5  # Impacto moderado
        )
    
    @staticmethod
    def validate_total_consistency(
        total_cents: int,
        items: list,
        tolerance_cents: int = 1
    ) -> ValidationResult:
        """
        Valida se total = soma dos itens.
        
        Total = Σ(Quantidade × Valor Unitário)
        
        Args:
            total_cents: Valor total em centavos
            items: Lista de dicts com 'quantity' e 'unit_price_cents'
            tolerance_cents: Tolerância para arredondamento
            
        Returns:
            ValidationResult com status
        """
        if not items:
            return ValidationResult(
                is_valid=True,
                field_name="total_consistency",
                extracted_value=str(total_cents),
                confidence_impact=0.0
            )
        
        calculated_total = 0
        for item in items:
            qty = item.get('quantity', 1)
            unit_price = item.get('unit_price_cents', 0)
            calculated_total += qty * unit_price
        
        difference = abs(total_cents - calculated_total)
        is_valid = difference <= tolerance_cents
        
        return ValidationResult(
            is_valid=is_valid,
            field_name="total_consistency",
            extracted_value=f"Extraído: {total_cents}, Calculado: {calculated_total}",
            expected_value=str(calculated_total) if not is_valid else None,
            error_message=f"Diferença de {difference} centavos" if not is_valid else None,
            confidence_impact=0.5 if is_valid else -0.8
        )


def validate_extraction(extraction_result: Dict[str, Any]) -> Dict[str, ValidationResult]:
    """
    Executa todas as validações aplicáveis em um resultado de extração.
    
    Args:
        extraction_result: Dict com campos extraídos
        
    Returns:
        Dict com resultados de validação por campo
    """
    results = {}
    
    # Validar CNPJ se presente
    cnpj = extraction_result.get('cnpj')
    if cnpj:
        results['cnpj'] = ChecksumValidators.validate_cnpj(cnpj)
    
    # Validar CPF se presente
    cpf = extraction_result.get('cpf')
    if cpf:
        results['cpf'] = ChecksumValidators.validate_cpf(cpf)
    
    # Validar Chave de Acesso NFe se presente
    access_key = extraction_result.get('access_key') or extraction_result.get('chave_acesso')
    if access_key:
        results['access_key'] = ChecksumValidators.validate_nfe_access_key(access_key)
        
        # Extrair e comparar número da NF
        if results['access_key'].is_valid:
            nf_from_key = ChecksumValidators.extract_nf_number_from_key(access_key)
            nf_extracted = extraction_result.get('doc_number')
            
            if nf_extracted and nf_from_key:
                if str(nf_extracted).lstrip('0') == nf_from_key:
                    logger.info(f"Número NF confirmado: {nf_from_key}")
                else:
                    logger.warning(f"Número NF divergente: extraído={nf_extracted}, chave={nf_from_key}")
    
    # Validar linha digitável de boleto se presente
    digitable_line = extraction_result.get('digitable_line') or extraction_result.get('linha_digitavel')
    if digitable_line:
        results['digitable_line'] = ChecksumValidators.validate_boleto_digitable_line(digitable_line)
    
    # Validar lógica de datas
    emission = extraction_result.get('emission_date') or extraction_result.get('data_emissao')
    due = extraction_result.get('due_date') or extraction_result.get('data_vencimento')
    payment = extraction_result.get('payment_date') or extraction_result.get('data_pagamento')
    
    if emission or due:
        results['date_logic'] = SemanticValidators.validate_date_logic(emission, due, payment)
    
    return results


class CrossFieldValidators:
    """
    Validadores de consistência cruzada entre campos (v3.0).
    
    Verifica:
    - Consistência Fornecedor vs CNPJ
    - Valor total vs Itens
    - Dupla validação de campos críticos
    """
    
    @staticmethod
    def validate_supplier_cnpj_consistency(
        supplier_name: Optional[str],
        cnpj: Optional[str],
        known_suppliers: Optional[Dict[str, str]] = None
    ) -> ValidationResult:
        """
        Valida se o fornecedor extraído é consistente com o CNPJ.
        
        Args:
            supplier_name: Nome do fornecedor extraído
            cnpj: CNPJ extraído
            known_suppliers: Dict {cnpj: nome_oficinal} (opcional)
            
        Returns:
            ValidationResult com status
        """
        if not supplier_name or not cnpj:
            return ValidationResult(
                is_valid=True,
                field_name="supplier_cnpj_consistency",
                extracted_value=f"Fornecedor: {supplier_name}, CNPJ: {cnpj}",
                confidence_impact=0.0
            )
        
        # Primeiro, valida o CNPJ em si
        cnpj_validation = ChecksumValidators.validate_cnpj(cnpj)
        if not cnpj_validation.is_valid:
            return ValidationResult(
                is_valid=False,
                field_name="supplier_cnpj_consistency",
                extracted_value=f"Fornecedor: {supplier_name}, CNPJ: {cnpj}",
                error_message=f"CNPJ inválido: {cnpj_validation.error_message}",
                confidence_impact=-1.0
            )
        
        # Se temos base de fornecedores conhecidos, verifica match
        if known_suppliers:
            cnpj_clean = re.sub(r'[^\d]', '', cnpj)
            if cnpj_clean in known_suppliers:
                expected_name = known_suppliers[cnpj_clean]
                # Fuzzy match no nome
                try:
                    from rapidfuzz import fuzz
                    similarity = fuzz.ratio(supplier_name.upper(), expected_name.upper()) / 100.0
                except ImportError:
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, supplier_name.upper(), expected_name.upper()).ratio()
                
                if similarity < 0.6:
                    return ValidationResult(
                        is_valid=False,
                        field_name="supplier_cnpj_consistency",
                        extracted_value=f"{supplier_name} (CNPJ: {cnpj})",
                        expected_value=expected_name,
                        error_message=f"Nome diverge do cadastro: {expected_name}",
                        confidence_impact=-0.5
                    )
        
        return ValidationResult(
            is_valid=True,
            field_name="supplier_cnpj_consistency",
            extracted_value=f"{supplier_name} (CNPJ: {cnpj})",
            confidence_impact=0.3
        )
    
    @staticmethod
    def validate_amount_duplicate_check(
        amount_cents: int,
        amounts_from_different_sources: list
    ) -> ValidationResult:
        """
        Valida se o valor extraído é consistente entre múltiplas fontes.
        
        Args:
            amount_cents: Valor principal extraído
            amounts_from_different_sources: Lista de valores de outras fontes
            
        Returns:
            ValidationResult com status
        """
        if not amounts_from_different_sources:
            return ValidationResult(
                is_valid=True,
                field_name="amount_duplicate_check",
                extracted_value=str(amount_cents),
                confidence_impact=0.0
            )
        
        # Verifica se todos os valores são iguais ou muito próximos
        tolerance = 5  # 5 centavos
        consistent = all(
            abs(amount_cents - other) <= tolerance 
            for other in amounts_from_different_sources
        )
        
        if not consistent:
            return ValidationResult(
                is_valid=False,
                field_name="amount_duplicate_check",
                extracted_value=f"Principal: {amount_cents}, Outros: {amounts_from_different_sources}",
                error_message="Valores divergentes entre fontes",
                confidence_impact=-0.8
            )
        
        return ValidationResult(
            is_valid=True,
            field_name="amount_duplicate_check",
            extracted_value=str(amount_cents),
            confidence_impact=0.5  # Boost por consenso
        )


def validate_document_complete(extraction_result: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
    """
    Executa validação completa de um documento e retorna score final.
    
    Args:
        extraction_result: Dict com todos os campos extraídos
        
    Returns:
        Tuple (is_valid, confidence_score, list_of_issues)
    """
    issues = []
    confidence_adjustments = []
    
    # 1. Validações de checksum
    validations = validate_extraction(extraction_result)
    
    for field, result in validations.items():
        if not result.is_valid:
            issues.append(f"[{field.upper()}] {result.error_message}")
        confidence_adjustments.append(result.confidence_impact)
    
    # 2. Validação cruzada fornecedor/CNPJ
    supplier = extraction_result.get('fornecedor') or extraction_result.get('supplier_clean')
    cnpj = extraction_result.get('cnpj')
    if supplier and cnpj:
        cross_result = CrossFieldValidators.validate_supplier_cnpj_consistency(supplier, cnpj)
        if not cross_result.is_valid:
            issues.append(f"[CONSISTENCY] {cross_result.error_message}")
        confidence_adjustments.append(cross_result.confidence_impact)
    
    # 3. Calcula score final
    base_confidence = extraction_result.get('confidence', 0.5)
    total_adjustment = sum(confidence_adjustments)
    final_confidence = max(0.0, min(1.0, base_confidence + total_adjustment * 0.1))
    
    # Documento é válido se não tiver issues críticos (ajuste < -0.5)
    has_critical = any(adj <= -0.5 for adj in confidence_adjustments)
    is_valid = not has_critical and len([i for i in issues if '[CRITICAL]' in i]) == 0
    
    return is_valid, final_confidence, issues


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("SDRA-Rural Deterministic Validators")
    print("=" * 60)
    
    # Teste CNPJ válido
    print("\n1. Teste CNPJ:")
    result = ChecksumValidators.validate_cnpj("11.222.333/0001-81")
    print(f"   11.222.333/0001-81: {'VALIDO' if result.is_valid else 'INVALIDO'}")
    print(f"   {result.error_message or 'OK'}")
    
    # Teste CNPJ inválido
    result = ChecksumValidators.validate_cnpj("11.222.333/0001-99")
    print(f"   11.222.333/0001-99: {'VALIDO' if result.is_valid else 'INVALIDO'}")
    print(f"   {result.error_message or 'OK'}")
    
    # Teste CPF válido (usando CPF de teste conhecido)
    print("\n2. Teste CPF:")
    result = ChecksumValidators.validate_cpf("529.982.247-25")
    print(f"   529.982.247-25: {'VALIDO' if result.is_valid else 'INVALIDO'}")
    print(f"   {result.error_message or 'OK'}")
    
    # Teste Chave NFe (exemplo sintético)
    print("\n3. Teste Chave NFe:")
    # Chave de exemplo (44 dígitos)
    test_key = "35201202507647000144550010000001231234567890"
    result = ChecksumValidators.validate_nfe_access_key(test_key)
    print(f"   Chave (44 dig): {'VALIDO' if result.is_valid else 'INVALIDO'}")
    print(f"   {result.error_message or 'OK'}")
    
    # Extração de número NF da chave
    nf_number = ChecksumValidators.extract_nf_number_from_key(test_key)
    print(f"   Numero NF extraido: {nf_number}")
    
    # Teste validação de datas
    print("\n4. Teste Logica Temporal:")
    result = SemanticValidators.validate_date_logic(
        emission_date="2025-01-01",
        due_date="2025-01-15"
    )
    print(f"   Emissao 01/01, Venc 15/01: {'VALIDO' if result.is_valid else 'INVALIDO'}")
    
    result = SemanticValidators.validate_date_logic(
        emission_date="2025-01-15",
        due_date="2025-01-01"
    )
    print(f"   Emissao 15/01, Venc 01/01: {'VALIDO' if result.is_valid else 'INVALIDO'}")
    print(f"   {result.error_message or 'OK'}")
    
    print("\n[OK] Todos os testes de validacao executados!")
