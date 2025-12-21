"""
SRDA-Rural Renamer Module
=========================
Modulo de Renomeacao Padronizada de Arquivos

Este modulo implementa a camada de saida do sistema, responsavel por:
- Geracao de nomes de arquivo padronizados
- Aplicacao das regras de nomenclatura VG/MV
- Sanitizacao de caracteres invalidos
- Renomeacao segura com dry-run

Referencia: NOMENCLATURAS BOLETOS E NOTAS.txt
           Automacao e Reconciliacao Financeira com IA.txt (Secao 6)
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag,
    MatchType
)

# Importa supplier matcher para nomes canonicos (fuzzy matching)
try:
    from supplier_matcher import match_supplier
    SUPPLIER_MATCHER_AVAILABLE = True
except ImportError:
    match_supplier = None
    SUPPLIER_MATCHER_AVAILABLE = False


# ==============================================================================
# CONFIGURACOES E CONSTANTES
# ==============================================================================

# Pasta de saida padrao
DEFAULT_OUTPUT_FOLDER = "Output"

# Sufixos juridicos a remover de nomes de fornecedores
LEGAL_SUFFIXES = [
    " LTDA",
    " ME",
    " EPP",
    " EIRELI",
    " S.A.",
    " S/A",
    " SA",
    " LTDA.",
    " - ME",
    " - EPP",
    " - EIRELI",
    " COMERCIO",
    " INDUSTRIA",
    " E COMERCIO",
    " SERVICOS",
    " & CIA",
]

# Caracteres invalidos para nomes de arquivo (Windows/Linux)
INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

# Caracteres a substituir por underscore
REPLACE_CHARS = r'[\s\-\.\,]+'


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class RenameOperation:
    """Representa uma operacao de renomeacao pendente."""
    doc_id: int
    original_path: str
    new_name: str
    new_path: str
    doc_type: DocumentType
    entity: EntityTag
    
    # Metadados para auditoria
    date_str: str = ""
    supplier: str = ""
    amount_display: str = ""
    doc_number: str = ""
    
    # Status
    executed: bool = False
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Verifica se a operacao e valida."""
        return (
            self.original_path and
            self.new_name and
            self.new_path and
            os.path.exists(self.original_path)
        )


@dataclass
class RenameResult:
    """Resultado do processo de renomeacao."""
    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    operations: List[RenameOperation] = None
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = []


# ==============================================================================
# CLASSE PRINCIPAL: DocumentRenamer
# ==============================================================================

class DocumentRenamer:
    """
    Renomeador de documentos financeiros.
    
    Implementa as regras de nomenclatura padronizada:
    - Formato: DATA_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
    - Suporte a parcelamentos: PARC_X-Y
    - Suporte a agrupamentos: multiplos numeros de nota
    
    Template: DATA_ENTIDADE_FORNECEDOR_VALORBOLETO_[VALORNF_PARC_X-Y_]TIPO_NUMERO.pdf
    """
    
    def __init__(
        self,
        db: Optional[SRDADatabase] = None,
        output_folder: str = DEFAULT_OUTPUT_FOLDER
    ):
        """
        Inicializa o renomeador.
        
        Args:
            db: Instancia do banco de dados
            output_folder: Pasta de destino para arquivos renomeados
        """
        self.db = db or SRDADatabase()
        self.output_folder = Path(output_folder)
        
        # Cria pasta de saida se nao existir
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # SANITIZACAO E NORMALIZACAO
    # ==========================================================================
    
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        Remove caracteres invalidos e normaliza o nome do arquivo.
        
        Args:
            name: Nome a sanitizar
            
        Returns:
            Nome sanitizado compativel com Windows/Linux
        """
        # Remove caracteres invalidos
        name = re.sub(INVALID_CHARS, '', name)
        
        # Substitui espacos e pontuacao por underscore
        name = re.sub(REPLACE_CHARS, '_', name)
        
        # Remove underscores duplicados
        name = re.sub(r'_+', '_', name)
        
        # Remove underscore no inicio/fim
        name = name.strip('_')
        
        # Limita tamanho (Windows max = 255)
        if len(name) > 200:
            name = name[:200]
        
        return name
    
    @staticmethod
    def normalize_supplier(supplier: str) -> str:
        """
        Normaliza nome do fornecedor removendo sufixos juridicos.
        
        Args:
            supplier: Nome original do fornecedor
            
        Returns:
            Nome normalizado (caixa alta, sem sufixos)
        """
        if not supplier:
            return "FORNECEDOR"
        
        # Converte para maiusculas
        normalized = supplier.upper().strip()
        
        # Remove sufixos juridicos
        for suffix in LEGAL_SUFFIXES:
            if normalized.endswith(suffix.upper()):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove pontuacao extra
        normalized = re.sub(r'[\.\,\-]+$', '', normalized)
        
        # Sanitiza
        normalized = DocumentRenamer.sanitize_filename(normalized)
        
        # Aplica fuzzy matching para nome canonico (se disponivel)
        if SUPPLIER_MATCHER_AVAILABLE and match_supplier and normalized:
            try:
                match_result = match_supplier(normalized, min_similarity=0.6)
                if match_result:
                    canonical, score = match_result
                    normalized = DocumentRenamer.sanitize_filename(canonical)
            except:
                pass
        
        return normalized if normalized else "FORNECEDOR"
    
    @staticmethod
    def normalize_doc_number(number: str) -> str:
        """
        Normaliza numero do documento removendo zeros a esquerda.
        
        Args:
            number: Numero original (ex: "000.006.270")
            
        Returns:
            Numero normalizado (ex: "6270")
        """
        if not number:
            return "0"
        
        # Remove pontuacao
        clean = re.sub(r'[\.\-\s]', '', number)
        
        # Converte para inteiro e volta para string (remove zeros)
        try:
            return str(int(clean))
        except ValueError:
            return clean
    
    @staticmethod
    def format_date(date_str: str) -> str:
        """
        Formata data para o padrao de nomenclatura (DD.MM.AAAA).
        
        Args:
            date_str: Data em formato ISO8601 (YYYY-MM-DD)
            
        Returns:
            Data formatada (DD.MM.AAAA)
        """
        if not date_str:
            return datetime.now().strftime("%d.%m.%Y")
        
        try:
            # Tenta parse de varios formatos
            if '-' in date_str:
                dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            elif '/' in date_str:
                dt = datetime.strptime(date_str[:10], "%d/%m/%Y")
            else:
                dt = datetime.strptime(date_str[:10], "%d.%m.%Y")
            
            return dt.strftime("%d.%m.%Y")
        except Exception:
            return datetime.now().strftime("%d.%m.%Y")
    
    @staticmethod
    def format_amount(amount_cents: int) -> str:
        """
        Formata valor para exibicao no nome do arquivo.
        
        Args:
            amount_cents: Valor em centavos
            
        Returns:
            Valor formatado sem R$ (ex: "3300,00")
        """
        if amount_cents <= 0:
            return "0,00"
        
        # Formata com virgula decimal
        value = amount_cents / 100
        formatted = f"{value:,.2f}"
        
        # Troca separadores para formato brasileiro
        # Mas no nome do arquivo, usamos ponto para milhar e virgula para decimal
        formatted = formatted.replace(",", "@").replace(".", "").replace("@", ",")
        
        return formatted
    
    # ==========================================================================
    # GERACAO DE NOMES
    # ==========================================================================
    
    def get_doc_type_suffix(self, doc_type: DocumentType) -> str:
        """Retorna o sufixo de tipo para o nome do arquivo."""
        suffixes = {
            DocumentType.NFE: "NFE",
            DocumentType.NFSE: "NFSE",
            DocumentType.BOLETO: "BOLETO",
            DocumentType.COMPROVANTE: "COMPROV",
            DocumentType.UNKNOWN: "DOC",
        }
        return suffixes.get(doc_type, "DOC")
    
    def build_filename(
        self,
        doc_type: DocumentType,
        entity: EntityTag,
        date_str: str,
        supplier: str,
        amount_cents: int,
        doc_number: str,
        nfe_amount_cents: Optional[int] = None,
        installment_info: Optional[str] = None,
        related_numbers: Optional[List[str]] = None
    ) -> str:
        """
        Constroi o nome do arquivo seguindo as regras de nomenclatura.
        
        Formato base: DATA_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
        
        Variacoes:
        - Parcelamento: DATA_ENTIDADE_FORNECEDOR_VLRBOL_VLRNF_PARC_X-Y_TIPO_NUM.pdf
        - Agrupamento: DATA_ENTIDADE_FORNECEDOR_VLRBOL_TIPO_NUM1_NUM2.pdf
        
        Args:
            doc_type: Tipo do documento
            entity: Entidade financeira (VG/MV)
            date_str: Data principal (ISO8601)
            supplier: Nome do fornecedor
            amount_cents: Valor principal em centavos
            doc_number: Numero do documento
            nfe_amount_cents: Valor da NF-e (para parcelamentos)
            installment_info: Info de parcela (ex: "PARC_1-2")
            related_numbers: Numeros de documentos relacionados
            
        Returns:
            Nome do arquivo completo com extensao
        """
        parts = []
        
        # 1. DATA (DD.MM.AAAA)
        parts.append(self.format_date(date_str))
        
        # 2. ENTIDADE (VG ou MV)
        parts.append(entity.value if entity else "XX")
        
        # 3. FORNECEDOR (normalizado)
        parts.append(self.normalize_supplier(supplier))
        
        # 4. VALOR DO BOLETO
        parts.append(self.format_amount(amount_cents))
        
        # 5. VALOR DA NF-e (se parcelamento)
        if nfe_amount_cents and nfe_amount_cents != amount_cents:
            parts.append(self.format_amount(nfe_amount_cents))
        
        # 6. INFO DE PARCELA (se aplicavel)
        if installment_info:
            parts.append(installment_info)
        
        # 7. TIPO DO DOCUMENTO
        parts.append(self.get_doc_type_suffix(doc_type))
        
        # 8. NUMERO DO DOCUMENTO
        parts.append(self.normalize_doc_number(doc_number))
        
        # 9. NUMEROS RELACIONADOS (para agrupamentos)
        if related_numbers:
            for num in related_numbers:
                parts.append(self.normalize_doc_number(num))
        
        # Junta com underscore e adiciona extensao
        filename = "_".join(parts) + ".pdf"
        
        return self.sanitize_filename(filename)
    
    # ==========================================================================
    # BUSCA DE DOCUMENTOS
    # ==========================================================================
    
    def get_reconciled_documents(self) -> List[Dict[str, Any]]:
        """Retorna documentos reconciliados prontos para renomeacao."""
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.original_path, d.doc_type, d.entity_tag, d.doc_number,
                t.amount_cents, t.emission_date, t.due_date, t.supplier_clean
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.status = 'RECONCILED'
            ORDER BY d.entity_tag, t.supplier_clean, t.due_date
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_match_info(self, doc_id: int) -> Dict[str, Any]:
        """
        Busca informacoes de matching para um documento.
        
        Returns:
            Dict com boleto_amount, nfe_amounts, installment_info, related_ids
        """
        cursor = self.db.connection.cursor()
        
        # Busca se e pai (boleto) ou filho (nota) em matches
        cursor.execute("""
            SELECT 
                m.id, m.parent_doc_id, m.child_doc_id, m.match_type, m.confidence,
                p.doc_type as parent_type, c.doc_type as child_type
            FROM matches m
            JOIN documentos p ON m.parent_doc_id = p.id
            JOIN documentos c ON m.child_doc_id = c.id
            WHERE m.parent_doc_id = ? OR m.child_doc_id = ?
        """, (doc_id, doc_id))
        
        matches = [dict(row) for row in cursor.fetchall()]
        
        info = {
            'is_parent': False,
            'is_child': False,
            'related_ids': [],
            'boleto_amount': 0,
            'nfe_amount': 0,
            'installment_info': None,
        }
        
        for match in matches:
            if match['parent_doc_id'] == doc_id:
                info['is_parent'] = True
                info['related_ids'].append(match['child_doc_id'])
            elif match['child_doc_id'] == doc_id:
                info['is_child'] = True
                info['related_ids'].append(match['parent_doc_id'])
        
        # Busca valores do boleto pai
        if info['is_child'] and info['related_ids']:
            parent_id = info['related_ids'][0]
            cursor.execute("""
                SELECT amount_cents FROM transacoes WHERE doc_id = ?
            """, (parent_id,))
            row = cursor.fetchone()
            if row:
                info['boleto_amount'] = row['amount_cents']
        
        # Verifica se e parcelamento (busca duplicatas)
        cursor.execute("""
            SELECT seq_num, amount_cents FROM duplicatas 
            WHERE nfe_id = ? OR boleto_id = ?
            ORDER BY seq_num
        """, (doc_id, doc_id))
        
        duplicatas = cursor.fetchall()
        if duplicatas:
            total_parts = len(duplicatas)
            # Tenta identificar qual parcela
            for dup in duplicatas:
                if dup['boleto_id'] == doc_id:
                    seq = dup['seq_num']
                    info['installment_info'] = f"PARC_{seq}-{total_parts}"
                    break
        
        return info
    
    # ==========================================================================
    # GERACAO DE OPERACOES
    # ==========================================================================
    
    def prepare_operations(self) -> List[RenameOperation]:
        """
        Prepara lista de operacoes de renomeacao.
        
        Returns:
            Lista de RenameOperation prontas para execucao ou dry-run
        """
        operations = []
        docs = self.get_reconciled_documents()
        
        for doc in docs:
            doc_id = doc['id']
            original_path = doc['original_path']
            doc_type = DocumentType(doc['doc_type']) if doc['doc_type'] else DocumentType.UNKNOWN
            entity = EntityTag(doc['entity_tag']) if doc['entity_tag'] else None
            
            # Dados da transacao
            amount = doc.get('amount_cents', 0)
            date_str = doc.get('due_date') or doc.get('emission_date') or ""
            supplier = doc.get('supplier_clean', "")
            doc_number = doc.get('doc_number', "")
            
            # Busca info de matching
            match_info = self.get_match_info(doc_id)
            
            # Define valores para o nome
            boleto_amount = match_info.get('boleto_amount', amount)
            nfe_amount = amount if match_info['is_child'] else None
            installment_info = match_info.get('installment_info')
            
            # Busca numeros relacionados
            related_numbers = []
            if match_info['related_ids']:
                cursor = self.db.connection.cursor()
                for rel_id in match_info['related_ids']:
                    cursor.execute(
                        "SELECT doc_number FROM documentos WHERE id = ?",
                        (rel_id,)
                    )
                    row = cursor.fetchone()
                    if row and row['doc_number']:
                        related_numbers.append(row['doc_number'])
            
            # Gera o novo nome
            new_name = self.build_filename(
                doc_type=doc_type,
                entity=entity,
                date_str=date_str,
                supplier=supplier,
                amount_cents=boleto_amount if boleto_amount else amount,
                doc_number=doc_number,
                nfe_amount_cents=nfe_amount,
                installment_info=installment_info,
                related_numbers=related_numbers if doc_type == DocumentType.BOLETO else None
            )
            
            # Define caminho de destino
            new_path = str(self.output_folder / new_name)
            
            # Cria operacao
            op = RenameOperation(
                doc_id=doc_id,
                original_path=original_path,
                new_name=new_name,
                new_path=new_path,
                doc_type=doc_type,
                entity=entity,
                date_str=self.format_date(date_str),
                supplier=self.normalize_supplier(supplier),
                amount_display=self.format_amount(amount),
                doc_number=self.normalize_doc_number(doc_number)
            )
            
            operations.append(op)
        
        return operations
    
    # ==========================================================================
    # EXECUCAO
    # ==========================================================================
    
    def execute_rename(
        self,
        operation: RenameOperation,
        copy_mode: bool = True
    ) -> bool:
        """
        Executa uma operacao de renomeacao.
        
        Args:
            operation: Operacao a executar
            copy_mode: Se True, copia o arquivo. Se False, move.
            
        Returns:
            True se sucesso, False se falha
        """
        try:
            if not os.path.exists(operation.original_path):
                operation.error = "Arquivo original nao encontrado"
                return False
            
            # Verifica se destino ja existe
            if os.path.exists(operation.new_path):
                # Adiciona sufixo numerico
                base, ext = os.path.splitext(operation.new_path)
                counter = 1
                while os.path.exists(f"{base}_{counter}{ext}"):
                    counter += 1
                operation.new_path = f"{base}_{counter}{ext}"
                operation.new_name = os.path.basename(operation.new_path)
            
            # Executa copia ou movimento
            if copy_mode:
                shutil.copy2(operation.original_path, operation.new_path)
            else:
                shutil.move(operation.original_path, operation.new_path)
            
            operation.executed = True
            
            # Atualiza status no banco
            self.db.update_document_status(
                doc_id=operation.doc_id,
                status=DocumentStatus.RENAMED
            )
            
            return True
            
        except Exception as e:
            operation.error = str(e)
            return False
    
    def run(
        self,
        dry_run: bool = True,
        copy_mode: bool = True
    ) -> RenameResult:
        """
        Executa o processo de renomeacao.
        
        Args:
            dry_run: Se True, apenas simula sem alterar arquivos
            copy_mode: Se True, copia arquivos. Se False, move.
            
        Returns:
            RenameResult com estatisticas
        """
        print("=" * 60)
        print("SRDA-Rural Renamer - Iniciando Renomeacao")
        print("=" * 60)
        print(f"Pasta de saida: {self.output_folder.absolute()}")
        print(f"Modo: {'DRY-RUN (simulacao)' if dry_run else 'EXECUCAO REAL'}")
        print(f"Operacao: {'COPIAR' if copy_mode else 'MOVER'}")
        print("-" * 60)
        
        result = RenameResult()
        
        # Prepara operacoes
        operations = self.prepare_operations()
        result.total_operations = len(operations)
        result.operations = operations
        
        if not operations:
            print("\n[INFO] Nenhum documento pronto para renomeacao.")
            print("Execute primeiro o scanner.py e matcher.py")
            return result
        
        print(f"\n{len(operations)} documento(s) para renomear")
        print("-" * 60)
        
        # Processa cada operacao
        for i, op in enumerate(operations, 1):
            entity_str = op.entity.value if op.entity else "??"
            print(f"\n[{i}/{len(operations)}] {op.doc_type.value} | {entity_str}")
            print(f"  Original: {Path(op.original_path).name}")
            print(f"  Novo:     {op.new_name}")
            
            if not op.is_valid:
                print("  -> [SKIP] Operacao invalida")
                result.skipped += 1
                continue
            
            if dry_run:
                print("  -> [DRY-RUN] Simulado com sucesso")
                result.successful += 1
            else:
                if self.execute_rename(op, copy_mode):
                    print("  -> [OK] Arquivo renomeado")
                    result.successful += 1
                else:
                    print(f"  -> [ERRO] {op.error}")
                    result.failed += 1
        
        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO DA RENOMEACAO")
        print("=" * 60)
        print(f"Total de operacoes:  {result.total_operations}")
        print(f"Sucesso:             {result.successful}")
        print(f"Falhas:              {result.failed}")
        print(f"Ignorados:           {result.skipped}")
        
        if dry_run:
            print("\n[AVISO] Modo DRY-RUN: nenhum arquivo foi alterado.")
            print("Execute com dry_run=False para aplicar as mudancas.")
        
        return result


# ==============================================================================
# FUNCOES UTILITARIAS
# ==============================================================================

def preview_renames(operations: List[RenameOperation]):
    """Exibe preview das renomeacoes planejadas."""
    if not operations:
        print("\nNenhuma operacao pendente.")
        return
    
    print("\n" + "=" * 60)
    print("PREVIEW DAS RENOMEACOES")
    print("=" * 60)
    
    for op in operations:
        entity = op.entity.value if op.entity else "??"
        print(f"\n[{op.doc_type.value}] {entity} - {op.supplier}")
        print(f"  Data: {op.date_str}")
        print(f"  Valor: R$ {op.amount_display}")
        print(f"  Numero: {op.doc_number}")
        print(f"  Original: {Path(op.original_path).name}")
        print(f"  -> Novo:  {op.new_name}")


# ==============================================================================
# EXEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Cria instancia do renomeador
    renamer = DocumentRenamer(output_folder="Output")
    
    # Executa em modo dry-run primeiro
    result = renamer.run(dry_run=True, copy_mode=True)
    
    # Exibe preview
    preview_renames(result.operations)
    
    # Estatisticas do banco
    print("\n" + "-" * 60)
    print("Estatisticas do Banco:")
    stats = renamer.db.get_statistics()
    print(f"  Total de documentos: {stats['total_documents']}")
    print(f"  Por status: {stats.get('by_status', {})}")
    
    renamer.db.close()
    print("\n[OK] Processo concluido!")
