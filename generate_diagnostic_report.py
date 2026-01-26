"""
SRDA-Rural Diagnostic Report Generator
=======================================
Gera relatorio detalhado de cada arquivo com diagnostico do erro.

Executa em TODOS os arquivos e gera CSV com:
- Arquivo
- Valor esperado (do nome)
- Valor extraido
- Diferenca
- Categoria do erro
- Detalhes de diagnostico
"""

import os
import re
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import traceback

# Adicionar diretorio pai ao path
sys.path.insert(0, str(Path(__file__).parent))

from scanner import CognitiveScanner
from database import SRDADatabase, DocumentType, EntityTag
from spatial_extractor import SpatialExtractor


@dataclass
class DiagnosticResult:
    """Resultado de diagnostico detalhado para um arquivo."""
    filename: str
    filepath: str
    
    # Ground Truth (do nome do arquivo)
    gt_date: str
    gt_entity: str
    gt_supplier: str
    gt_amount_str: str
    gt_amount_cents: int
    gt_total_amount_str: Optional[str]
    gt_total_amount_cents: Optional[int]
    gt_doc_type: str
    gt_doc_number: Optional[str]
    gt_installment: Optional[str]
    
    # Extracao
    ext_amount_cents: int
    ext_entity: str
    ext_supplier: str
    ext_doc_type: str
    ext_doc_number: str
    ext_confidence: float
    extraction_time_ms: int
    
    # Analise
    amount_match: bool
    difference_cents: int
    difference_pct: float
    error_category: str
    error_details: str
    raw_text_preview: str


def parse_amount_to_cents(amount_str: str) -> int:
    """Converte string de valor brasileiro para centavos."""
    if not amount_str:
        return 0
    try:
        clean = amount_str.replace('R$', '').replace(' ', '')
        clean = clean.replace('.', '').replace(',', '.')
        return int(float(clean) * 100)
    except:
        return 0


def parse_filename_detailed(filename: str) -> Dict[str, Any]:
    """
    Parseia o nome do arquivo com detalhes completos.
    Retorna dict com todos os campos extraidos do nome.
    """
    result = {
        'date': '',
        'entity': '',
        'supplier': '',
        'amount': '',
        'amount_cents': 0,
        'total_amount': None,
        'total_amount_cents': None,
        'doc_type': 'UNKNOWN',
        'doc_number': None,
        'installment': None,
        'parse_success': False,
        'parse_error': None
    }
    
    try:
        name = Path(filename).stem
        parts = name.split('_')
        
        if len(parts) < 4:
            result['parse_error'] = f"Poucos campos no nome ({len(parts)})"
            return result
        
        # Data
        result['date'] = parts[0]
        
        # Entidade
        result['entity'] = parts[1] if parts[1] in ['VG', 'MV', 'SEFAZ', 'SOS'] else 'VG'
        
        # Fornecedor
        result['supplier'] = parts[2]
        
        # Tipos de documento conhecidos
        doc_types = ['NFE', 'BOLETO', 'NFSE', 'FATURA', 'CTE', 'DAR', 'APOLICE', 'CC', 'CONTRATO', 'PIX', 'NFSSE']
        
        amounts = []
        for i, part in enumerate(parts[3:], start=3):
            part_upper = part.upper()
            
            # Documento tipo
            if any(part_upper.startswith(dt) for dt in doc_types):
                result['doc_type'] = part_upper.split()[0]  # Pega primeira palavra
                # Numero do documento
                if i + 1 < len(parts):
                    remaining = '_'.join(parts[i+1:])
                    numbers = re.findall(r'\d+', remaining)
                    if numbers:
                        result['doc_number'] = numbers[0]
                break
            
            # Parcela
            if 'PARC' in part_upper:
                result['installment'] = part
                continue
            
            # Valor (numeros com virgula)
            if re.match(r'^[\d\.,]+$', part.replace('.', '').replace(',', '')):
                # Verificar se parece um valor valido (tem virgula para centavos)
                if ',' in part or (part.isdigit() and len(part) <= 6):
                    amounts.append(part)
        
        # Atribuir valores
        if len(amounts) >= 1:
            result['amount'] = amounts[0]
            result['amount_cents'] = parse_amount_to_cents(amounts[0])
        
        if len(amounts) >= 2:
            result['total_amount'] = amounts[1]
            result['total_amount_cents'] = parse_amount_to_cents(amounts[1])
        
        result['parse_success'] = True
        
    except Exception as e:
        result['parse_error'] = str(e)
    
    return result


def categorize_error(gt_cents: int, ext_cents: int, gt_total_cents: int, 
                     installment: str, doc_type: str, ext_confidence: float,
                     raw_text_len: int) -> Tuple[str, str]:
    """
    Categoriza o tipo de erro para facilitar analise.
    
    Returns:
        (categoria, detalhes)
    """
    diff = ext_cents - gt_cents
    diff_pct = abs(diff) / gt_cents * 100 if gt_cents > 0 else 0
    
    # Erro zero - nenhuma extracao
    if ext_cents == 0:
        if raw_text_len < 100:
            return "EXTRACAO_VAZIA", "Texto muito curto ou PDF escaneado sem OCR"
        return "EXTRACAO_VAZIA", "Nenhum valor extraido, pode ser layout nao reconhecido"
    
    # Match exato
    if abs(diff) <= max(int(gt_cents * 0.01), 10):
        return "OK", "Valor correto"
    
    # Extraiu o total em vez da parcela
    if gt_total_cents and abs(ext_cents - gt_total_cents) <= max(int(gt_total_cents * 0.01), 10):
        return "TOTAL_VS_PARCELA", f"Extraiu total ({gt_total_cents/100:.2f}) ao inves da parcela ({gt_cents/100:.2f})"
    
    # Valor proximo do dobro (pode ser soma de documentos)
    if 1.9 <= ext_cents / gt_cents <= 2.1:
        return "DOBRO_DO_VALOR", "Extraiu aproximadamente o dobro, possivel soma de documentos"
    
    # Valor proximo da metade
    if 0.48 <= ext_cents / gt_cents <= 0.52:
        return "METADE_DO_VALOR", "Extraiu aproximadamente metade, pode ser valor de um item"
    
    # Valor muito maior (>3x)
    if ext_cents > gt_cents * 3:
        return "VALOR_MUITO_MAIOR", f"Valor extraido {ext_cents/gt_cents:.1f}x maior, pode ser total de fatura"
    
    # Valor muito menor (<1/3)
    if ext_cents < gt_cents / 3:
        return "VALOR_MUITO_MENOR", f"Valor extraido {gt_cents/ext_cents:.1f}x menor"
    
    # Diferenca moderada
    if diff_pct <= 20:
        return "DIFERENCA_PEQUENA", f"Diferenca de {diff_pct:.1f}%, pode ser arredondamento ou desconto"
    
    # Outros casos
    return "DIFERENCA_GRANDE", f"Diferenca de {diff_pct:.1f}% nao categorizada"


class DiagnosticGenerator:
    """Gera relatorio diagnostico completo."""
    
    def __init__(self, folder_path: str, output_csv: str):
        self.folder = Path(folder_path)
        self.output_csv = output_csv
        self.scanner = CognitiveScanner()
        self.results: List[DiagnosticResult] = []
        
    def collect_files(self) -> List[Path]:
        """Coleta todos os PDFs da pasta."""
        files = []
        for ext in ['*.pdf', '*.PDF']:
            files.extend(self.folder.glob(ext))
        return sorted(files)
    
    def process_file(self, filepath: Path) -> DiagnosticResult:
        """Processa um unico arquivo e gera diagnostico."""
        start_time = time.time()
        
        # Parse do nome
        gt = parse_filename_detailed(filepath.name)
        
        # Valores padrao
        ext_amount = 0
        ext_entity = ''
        ext_supplier = ''
        ext_doc_type = ''
        ext_doc_number = ''
        ext_confidence = 0.0
        raw_text_preview = ''
        error_details = ''
        
        try:
            # Extrair
            result = self.scanner.hierarchical_extract(filepath, 0, DocumentType.UNKNOWN)
            
            ext_amount = result.get('amount_cents', 0)
            
            entity = result.get('entity_tag')
            if entity:
                ext_entity = entity.name if hasattr(entity, 'name') else str(entity)
            
            ext_supplier = result.get('supplier_clean', '') or ''
            ext_doc_type = str(result.get('doc_type', ''))
            ext_doc_number = str(result.get('doc_number', ''))
            ext_confidence = result.get('confidence', 0.0)
            
            # Preview do texto raw
            raw_text = result.get('raw_text', '')
            if raw_text:
                raw_text_preview = raw_text[:200].replace('\n', ' ').replace('\r', '')
            
        except Exception as e:
            error_details = f"EXCEPTION: {str(e)}"
            raw_text_preview = f"ERRO: {traceback.format_exc()[:200]}"
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Calcular diferenca
        gt_cents = gt['amount_cents']
        diff_cents = ext_amount - gt_cents
        diff_pct = abs(diff_cents) / gt_cents * 100 if gt_cents > 0 else 0
        amount_match = abs(diff_cents) <= max(int(gt_cents * 0.01), 10)
        
        # Categorizar erro
        error_cat, error_det = categorize_error(
            gt_cents, ext_amount, 
            gt.get('total_amount_cents') or 0,
            gt.get('installment'),
            gt.get('doc_type'),
            ext_confidence,
            len(raw_text_preview)
        )
        
        if error_details:  # Exception
            error_cat = "EXCEPTION"
            error_det = error_details
        
        return DiagnosticResult(
            filename=filepath.name,
            filepath=str(filepath),
            gt_date=gt['date'],
            gt_entity=gt['entity'],
            gt_supplier=gt['supplier'],
            gt_amount_str=gt['amount'],
            gt_amount_cents=gt_cents,
            gt_total_amount_str=gt.get('total_amount'),
            gt_total_amount_cents=gt.get('total_amount_cents'),
            gt_doc_type=gt['doc_type'],
            gt_doc_number=gt.get('doc_number'),
            gt_installment=gt.get('installment'),
            ext_amount_cents=ext_amount,
            ext_entity=ext_entity,
            ext_supplier=ext_supplier,
            ext_doc_type=ext_doc_type,
            ext_doc_number=ext_doc_number,
            ext_confidence=ext_confidence,
            extraction_time_ms=elapsed_ms,
            amount_match=amount_match,
            difference_cents=diff_cents,
            difference_pct=diff_pct,
            error_category=error_cat,
            error_details=error_det,
            raw_text_preview=raw_text_preview
        )
    
    def run(self, max_files: int = None, skip_ok: bool = False):
        """
        Executa diagnostico em todos os arquivos.
        
        Args:
            max_files: Limitar numero de arquivos
            skip_ok: Se True, nao inclui arquivos OK no CSV
        """
        files = self.collect_files()
        
        if max_files:
            files = files[:max_files]
        
        total = len(files)
        print(f"\n{'='*70}")
        print(f"DIAGNOSTICO COMPLETO - {total} ARQUIVOS")
        print(f"{'='*70}")
        print(f"Pasta: {self.folder}")
        print(f"Saida: {self.output_csv}")
        print(f"{'='*70}\n")
        
        stats = defaultdict(int)
        
        for i, filepath in enumerate(files, 1):
            # Progress a cada 10 arquivos
            if i % 10 == 0 or i == 1:
                print(f"[{i:4d}/{total}] Processando...", flush=True)
            
            result = self.process_file(filepath)
            self.results.append(result)
            
            stats[result.error_category] += 1
            stats['total'] += 1
        
        # Escrever CSV
        self.write_csv(skip_ok)
        
        # Imprimir resumo
        self.print_summary(stats)
    
    def write_csv(self, skip_ok: bool = False):
        """Escreve resultados em CSV."""
        results_to_write = self.results
        if skip_ok:
            results_to_write = [r for r in self.results if r.error_category != 'OK']
        
        with open(self.output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            # utf-8-sig para abrir corretamente no Excel brasileiro
            writer = csv.writer(f, delimiter=';')  # Ponto-virgula para Excel PT-BR
            
            # Header
            writer.writerow([
                'ARQUIVO',
                'GT_VALOR_STR',
                'GT_VALOR_CENTS',
                'EXT_VALOR_CENTS',
                'DIFERENCA_CENTS',
                'DIFERENCA_%',
                'MATCH',
                'CATEGORIA_ERRO',
                'DETALHES_ERRO',
                'GT_TIPO_DOC',
                'GT_NUM_DOC',
                'GT_PARCELA',
                'GT_TOTAL_STR',
                'GT_TOTAL_CENTS',
                'GT_FORNECEDOR',
                'GT_ENTIDADE',
                'EXT_FORNECEDOR',
                'EXT_ENTIDADE',
                'EXT_TIPO_DOC',
                'EXT_NUM_DOC',
                'CONFIANCA',
                'TEMPO_MS',
                'TEXTO_PREVIEW'
            ])
            
            # Data
            for r in results_to_write:
                writer.writerow([
                    r.filename,
                    r.gt_amount_str,
                    r.gt_amount_cents,
                    r.ext_amount_cents,
                    r.difference_cents,
                    f"{r.difference_pct:.1f}",
                    'SIM' if r.amount_match else 'NAO',
                    r.error_category,
                    r.error_details,
                    r.gt_doc_type,
                    r.gt_doc_number or '',
                    r.gt_installment or '',
                    r.gt_total_amount_str or '',
                    r.gt_total_amount_cents or '',
                    r.gt_supplier,
                    r.gt_entity,
                    r.ext_supplier,
                    r.ext_entity,
                    r.ext_doc_type,
                    r.ext_doc_number,
                    f"{r.ext_confidence:.2f}",
                    r.extraction_time_ms,
                    r.raw_text_preview[:100]
                ])
        
        print(f"\nCSV salvo: {self.output_csv}")
        print(f"Total de linhas: {len(results_to_write)}")
    
    def print_summary(self, stats: Dict[str, int]):
        """Imprime resumo das categorias de erro."""
        print(f"\n{'='*70}")
        print("RESUMO POR CATEGORIA DE ERRO")
        print(f"{'='*70}")
        
        total = stats['total']
        ok_count = stats.get('OK', 0)
        error_count = total - ok_count
        
        print(f"Total processado: {total}")
        print(f"  OK (valor correto):     {ok_count:4d} ({ok_count/total*100:.1f}%)")
        print(f"  Com erro:               {error_count:4d} ({error_count/total*100:.1f}%)")
        print()
        
        print("Detalhamento dos erros:")
        for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
            if cat not in ['total', 'OK']:
                print(f"  {cat:25s} {count:4d} ({count/total*100:.1f}%)")
        
        print(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerar relatorio diagnostico SRDA-Rural')
    parser.add_argument('--folder', '-f', 
                        default='11.2025_NOVEMBRO_1.547',
                        help='Pasta com arquivos PDF')
    parser.add_argument('--output', '-o', 
                        default='diagnostico_extracao.csv',
                        help='Arquivo CSV de saida')
    parser.add_argument('--max-files', '-n', type=int, default=None,
                        help='Limitar numero de arquivos')
    parser.add_argument('--errors-only', action='store_true',
                        help='Incluir apenas arquivos com erro no CSV')
    
    args = parser.parse_args()
    
    # Verificar pasta
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"ERRO: Pasta nao encontrada: {folder_path}")
        sys.exit(1)
    
    # Gerar diagnostico
    generator = DiagnosticGenerator(str(folder_path), args.output)
    generator.run(max_files=args.max_files, skip_ok=args.errors_only)
    
    print(f"\nAbra o arquivo '{args.output}' no Excel para analise detalhada.")
    print("O CSV usa ponto-virgula como separador para compatibilidade com Excel PT-BR.")


if __name__ == "__main__":
    main()
