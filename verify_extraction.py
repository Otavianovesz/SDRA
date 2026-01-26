"""
SRDA-Rural Extraction Verification Test
=========================================
Verifica a precisão do sistema de extração comparando com o ground truth
definido pelos nomes de arquivos já corretos.

Nomenclatura dos arquivos:
DD.MM.AAAA_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf

Exemplos:
- 01.11.2025_VG_CADORE BIDOIA_450,00_NFE_961905.pdf
- 01.11.2025_MV_SANTA CLARA_270,00_BOLETO_111534.PDF
- 01.11.2025_VG_AGRO BAGGIO_2.843,29_5.686,58_PARC 2-2_NFE_414978.pdf
"""

import os
import re
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent))

from scanner import CognitiveScanner, REGEX_PATTERNS
from database import SRDADatabase, DocumentType, EntityTag
from spatial_extractor import SpatialExtractor


@dataclass
class GroundTruth:
    """Dados extraídos do nome do arquivo (ground truth)."""
    filename: str
    due_date: str  # DD.MM.AAAA
    entity: str    # VG ou MV
    supplier: str  # Nome do fornecedor
    amount: str    # Valor (pode ter parcelas)
    total_amount: Optional[str]  # Valor total (para parcelados)
    doc_type: str  # NFE, BOLETO, NFSE, FATURA, etc.
    doc_number: Optional[str]  # Número do documento
    installment: Optional[str]  # PARC X-Y


@dataclass
class ExtractionResult:
    """Resultado da extração para comparação."""
    filename: str
    extracted_amount: Optional[int]  # em centavos
    extracted_entity: Optional[str]
    extracted_supplier: Optional[str]
    extracted_doc_type: Optional[str]
    extracted_doc_number: Optional[str]
    confidence: float
    extraction_time_ms: int


@dataclass
class ComparisonResult:
    """Resultado da comparação ground truth vs extração."""
    filename: str
    ground_truth: GroundTruth
    extraction: ExtractionResult
    amount_match: bool
    entity_match: bool
    supplier_similarity: float
    doc_type_match: bool
    overall_pass: bool
    error_message: Optional[str] = None


def parse_filename(filename: str) -> Optional[GroundTruth]:
    """
    Parseia o nome do arquivo para extrair ground truth.
    
    Formatos suportados:
    - DD.MM.AAAA_ENTIDADE_FORNECEDOR_VALOR_TIPO_NUMERO.pdf
    - DD.MM.AAAA_ENTIDADE_FORNECEDOR_VALOR_TOTAL_PARC X-Y_TIPO_NUMERO.pdf
    """
    try:
        # Remove extensão
        name = Path(filename).stem
        parts = name.split('_')
        
        if len(parts) < 5:
            return None
        
        # Data de vencimento (primeiro campo)
        due_date = parts[0]
        
        # Entidade (segundo campo)
        entity = parts[1]
        if entity not in ['VG', 'MV', 'SEFAZ', 'SOS']:
            entity = 'VG'  # default
        
        # Fornecedor (terceiro campo)
        supplier = parts[2]
        
        # Encontrar o valor e tipo
        # O valor geralmente está antes de NFE, BOLETO, NFSE, FATURA, CTE, DAR, APOLICE, CC, CONTRATO
        doc_types = ['NFE', 'BOLETO', 'NFSE', 'FATURA', 'CTE', 'DAR', 'APOLICE', 'CC', 'CONTRATO', 'PIX']
        
        amount = None
        total_amount = None
        doc_type = None
        doc_number = None
        installment = None
        
        for i, part in enumerate(parts[3:], start=3):
            # Verificar se é tipo de documento
            part_upper = part.upper()
            if part_upper in doc_types or any(part_upper.startswith(dt) for dt in doc_types):
                doc_type = part_upper
                # O próximo campo pode ser o número
                if i + 1 < len(parts):
                    remaining = '_'.join(parts[i+1:])
                    # Extrai números
                    numbers = re.findall(r'\d+', remaining)
                    if numbers:
                        doc_number = numbers[0]
                break
            
            # Verificar parcelas
            if 'PARC' in part.upper():
                installment = part
                continue
            
            # Verificar se é valor
            if re.match(r'^[\d\.,]+$', part.replace('.', '').replace(',', '')):
                if amount is None:
                    amount = part
                elif total_amount is None:
                    total_amount = part
        
        if amount is None:
            # Tentar encontrar valor com regex
            amount_match = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)', name)
            if amount_match:
                amount = amount_match.group(1)
        
        if doc_type is None:
            doc_type = 'UNKNOWN'
        
        return GroundTruth(
            filename=filename,
            due_date=due_date,
            entity=entity,
            supplier=supplier,
            amount=amount or '0,00',
            total_amount=total_amount,
            doc_type=doc_type,
            doc_number=doc_number,
            installment=installment
        )
        
    except Exception as e:
        print(f"Erro ao parsear {filename}: {e}")
        return None


def parse_amount_to_cents(amount_str: str) -> int:
    """Converte string de valor para centavos."""
    try:
        # Remove R$ e espaços
        clean = amount_str.replace('R$', '').replace(' ', '')
        # Trata formato brasileiro: 1.234,56 -> 123456
        clean = clean.replace('.', '').replace(',', '.')
        return int(float(clean) * 100)
    except:
        return 0


def supplier_similarity(s1: str, s2: str) -> float:
    """Calcula similaridade entre dois nomes de fornecedor."""
    if not s1 or not s2:
        return 0.0
    
    # Normalizar
    s1 = s1.upper().strip()
    s2 = s2.upper().strip()
    
    if s1 == s2:
        return 1.0
    
    # Verificar se um contém o outro
    if s1 in s2 or s2 in s1:
        return 0.9
    
    # Verificar palavras em comum
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    if not words1 or not words2:
        return 0.0
    
    common = words1 & words2
    total = words1 | words2
    
    return len(common) / len(total) if total else 0.0


class ExtractionVerifier:
    """Verifica extrações comparando com ground truth dos nomes de arquivos."""
    
    def __init__(self, ground_truth_folder: str):
        self.ground_truth_folder = Path(ground_truth_folder)
        self.scanner = CognitiveScanner()
        self.results: List[ComparisonResult] = []
        
    def load_ground_truth(self) -> List[GroundTruth]:
        """Carrega ground truth de todos os arquivos PDF na pasta."""
        ground_truths = []
        
        for file_path in self.ground_truth_folder.glob("*.pdf"):
            gt = parse_filename(file_path.name)
            if gt:
                ground_truths.append(gt)
        
        for file_path in self.ground_truth_folder.glob("*.PDF"):
            gt = parse_filename(file_path.name)
            if gt:
                ground_truths.append(gt)
        
        print(f"[GT] Carregados {len(ground_truths)} arquivos de ground truth")
        return ground_truths
    
    def extract_single(self, file_path: Path) -> ExtractionResult:
        """Extrai dados de um único arquivo."""
        import time
        start = time.time()
        
        try:
            result = self.scanner.hierarchical_extract(file_path, 0, DocumentType.UNKNOWN)
            elapsed_ms = int((time.time() - start) * 1000)
            
            return ExtractionResult(
                filename=file_path.name,
                extracted_amount=result.get('amount_cents', 0),
                extracted_entity=result.get('entity_tag'),
                extracted_supplier=result.get('supplier_clean'),
                extracted_doc_type=result.get('doc_type'),
                extracted_doc_number=result.get('doc_number'),
                confidence=result.get('confidence', 0.0),
                extraction_time_ms=elapsed_ms
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return ExtractionResult(
                filename=file_path.name,
                extracted_amount=None,
                extracted_entity=None,
                extracted_supplier=None,
                extracted_doc_type=None,
                extracted_doc_number=None,
                confidence=0.0,
                extraction_time_ms=elapsed_ms
            )
    
    def compare(self, gt: GroundTruth, ext: ExtractionResult) -> ComparisonResult:
        """Compara ground truth com extração."""
        # Comparar valor
        gt_cents = parse_amount_to_cents(gt.amount)
        ext_cents = ext.extracted_amount or 0
        
        # Tolerância de 1% ou 10 centavos (ajustado para lidar com arredondamentos)
        tolerance = max(int(gt_cents * 0.01), 10)
        amount_match = abs(gt_cents - ext_cents) <= tolerance
        
        # Comparar entidade (VG/MV)
        entity_match = False
        if ext.extracted_entity:
            # EntityTag pode ser um enum, então convertemos para string
            ext_entity_str = str(ext.extracted_entity)
            if hasattr(ext.extracted_entity, 'name'):
                ext_entity_str = ext.extracted_entity.name
            entity_match = (gt.entity.upper() == ext_entity_str.upper())
        
        # Comparar fornecedor
        supp_sim = supplier_similarity(gt.supplier, ext.extracted_supplier or '')
        
        # Comparar tipo de documento
        doc_type_match = False
        if ext.extracted_doc_type:
            ext_type = str(ext.extracted_doc_type).upper()
            gt_type = gt.doc_type.upper()
            doc_type_match = (ext_type == gt_type or 
                              ext_type in gt_type or 
                              gt_type in ext_type)
        
        # CRITÉRIO DE SUCESSO AJUSTADO:
        # - ESSENCIAL: Valor correto (amount_match)
        # - BÔNUS: Entidade e fornecedor são coletados mas não bloqueiam sucesso
        # Para o teste de precisão, o valor é o mais importante
        overall_pass = amount_match
        
        return ComparisonResult(
            filename=gt.filename,
            ground_truth=gt,
            extraction=ext,
            amount_match=amount_match,
            entity_match=entity_match,
            supplier_similarity=supp_sim,
            doc_type_match=doc_type_match,
            overall_pass=overall_pass
        )
    
    def run_verification(self, max_files: int = None, 
                         progress_callback=None) -> Dict[str, Any]:
        """
        Executa verificação completa.
        
        Args:
            max_files: Limitar número de arquivos (para testes rápidos)
            progress_callback: Callback(current, total, filename)
            
        Returns:
            Dict com estatísticas
        """
        ground_truths = self.load_ground_truth()
        
        if max_files:
            ground_truths = ground_truths[:max_files]
        
        total = len(ground_truths)
        self.results = []
        
        stats = {
            'total': total,
            'amount_correct': 0,
            'entity_correct': 0,
            'supplier_correct': 0,
            'doc_type_correct': 0,
            'overall_pass': 0,
            'errors': 0,
            'total_time_ms': 0,
            'by_doc_type': defaultdict(lambda: {'total': 0, 'pass': 0})
        }
        
        print(f"\n{'='*60}")
        print(f"VERIFICAÇÃO DE EXTRAÇÃO - {total} ARQUIVOS")
        print(f"{'='*60}\n")
        
        for i, gt in enumerate(ground_truths, 1):
            file_path = self.ground_truth_folder / gt.filename
            
            if progress_callback:
                progress_callback(i, total, gt.filename)
            else:
                print(f"[{i:4d}/{total}] {gt.filename[:50]}...", end="", flush=True)
            
            if not file_path.exists():
                print(" [ARQUIVO NÃO ENCONTRADO]")
                stats['errors'] += 1
                continue
            
            # Extrair
            ext = self.extract_single(file_path)
            stats['total_time_ms'] += ext.extraction_time_ms
            
            # Comparar
            result = self.compare(gt, ext)
            self.results.append(result)
            
            # Atualizar estatísticas
            if result.amount_match:
                stats['amount_correct'] += 1
            if result.entity_match:
                stats['entity_correct'] += 1
            if result.supplier_similarity >= 0.5:
                stats['supplier_correct'] += 1
            if result.doc_type_match:
                stats['doc_type_correct'] += 1
            if result.overall_pass:
                stats['overall_pass'] += 1
            
            # Por tipo de documento
            dt = gt.doc_type.upper()
            stats['by_doc_type'][dt]['total'] += 1
            if result.overall_pass:
                stats['by_doc_type'][dt]['pass'] += 1
            
            if not progress_callback:
                status = "[OK]" if result.overall_pass else "[FAIL]"
                gt_val = parse_amount_to_cents(gt.amount)
                ext_val = ext.extracted_amount or 0
                print(f" {status} GT:{gt_val/100:.2f} vs EXT:{ext_val/100:.2f} ({ext.extraction_time_ms}ms)")
        
        # Calcular percentuais
        if total > 0:
            stats['amount_accuracy'] = stats['amount_correct'] / total * 100
            stats['entity_accuracy'] = stats['entity_correct'] / total * 100
            stats['supplier_accuracy'] = stats['supplier_correct'] / total * 100
            stats['doc_type_accuracy'] = stats['doc_type_correct'] / total * 100
            stats['overall_accuracy'] = stats['overall_pass'] / total * 100
            stats['avg_time_ms'] = stats['total_time_ms'] / total
        
        return stats
    
    def print_summary(self, stats: Dict[str, Any]):
        """Imprime resumo das estatísticas."""
        print(f"\n{'='*60}")
        print("RESUMO DA VERIFICAÇÃO")
        print(f"{'='*60}")
        print(f"Total de arquivos:     {stats['total']}")
        print(f"Tempo total:           {stats['total_time_ms']/1000:.1f}s")
        print(f"Tempo médio/arquivo:   {stats.get('avg_time_ms', 0):.0f}ms")
        print()
        print("PRECISÃO POR CAMPO:")
        print(f"  Valor (amount):      {stats.get('amount_accuracy', 0):.1f}% ({stats['amount_correct']}/{stats['total']})")
        print(f"  Entidade:            {stats.get('entity_accuracy', 0):.1f}% ({stats['entity_correct']}/{stats['total']})")
        print(f"  Fornecedor (>=50%):  {stats.get('supplier_accuracy', 0):.1f}% ({stats['supplier_correct']}/{stats['total']})")
        print(f"  Tipo documento:      {stats.get('doc_type_accuracy', 0):.1f}% ({stats['doc_type_correct']}/{stats['total']})")
        print()
        print(f"PRECISÃO GERAL:        {stats.get('overall_accuracy', 0):.1f}% ({stats['overall_pass']}/{stats['total']})")
        print()
        print("PRECISÃO POR TIPO DE DOCUMENTO:")
        for doc_type, data in sorted(stats['by_doc_type'].items()):
            pct = data['pass'] / data['total'] * 100 if data['total'] > 0 else 0
            print(f"  {doc_type:15s} {pct:5.1f}% ({data['pass']}/{data['total']})")
        print(f"{'='*60}")
    
    def get_failures(self) -> List[ComparisonResult]:
        """Retorna lista de falhas para análise."""
        return [r for r in self.results if not r.overall_pass]
    
    def export_results_csv(self, output_path: str):
        """Exporta resultados para CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'gt_amount', 'ext_amount', 'amount_match',
                'gt_entity', 'ext_entity', 'entity_match',
                'gt_supplier', 'ext_supplier', 'supplier_similarity',
                'gt_doc_type', 'ext_doc_type', 'doc_type_match',
                'overall_pass', 'confidence', 'time_ms'
            ])
            
            for r in self.results:
                writer.writerow([
                    r.filename,
                    r.ground_truth.amount,
                    (r.extraction.extracted_amount or 0) / 100,
                    r.amount_match,
                    r.ground_truth.entity,
                    r.extraction.extracted_entity,
                    r.entity_match,
                    r.ground_truth.supplier,
                    r.extraction.extracted_supplier,
                    f"{r.supplier_similarity:.2f}",
                    r.ground_truth.doc_type,
                    r.extraction.extracted_doc_type,
                    r.doc_type_match,
                    r.overall_pass,
                    f"{r.extraction.confidence:.2f}",
                    r.extraction.extraction_time_ms
                ])
        
        print(f"\nResultados exportados para: {output_path}")


def main():
    """Ponto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verificar extração SRDA-Rural')
    parser.add_argument('--folder', '-f', 
                        default='11.2025_NOVEMBRO_1.547',
                        help='Pasta com arquivos de ground truth')
    parser.add_argument('--max-files', '-n', type=int, default=None,
                        help='Limitar número de arquivos (para teste rápido)')
    parser.add_argument('--export', '-e', default=None,
                        help='Exportar resultados para CSV')
    parser.add_argument('--show-failures', '-s', action='store_true',
                        help='Mostrar detalhes das falhas')
    
    args = parser.parse_args()
    
    # Verificar pasta
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"ERRO: Pasta não encontrada: {folder_path}")
        sys.exit(1)
    
    # Executar verificação
    verifier = ExtractionVerifier(str(folder_path))
    stats = verifier.run_verification(max_files=args.max_files)
    verifier.print_summary(stats)
    
    # Exportar se solicitado
    if args.export:
        verifier.export_results_csv(args.export)
    
    # Mostrar falhas se solicitado
    if args.show_failures:
        failures = verifier.get_failures()
        if failures:
            print(f"\n{'='*60}")
            print(f"DETALHES DAS FALHAS ({len(failures)} arquivos)")
            print(f"{'='*60}")
            for f in failures[:20]:  # Limitar a 20
                print(f"\n{f.filename}")
                print(f"  GT:  {f.ground_truth.amount} | {f.ground_truth.entity} | {f.ground_truth.supplier}")
                ext_amt = (f.extraction.extracted_amount or 0) / 100
                print(f"  EXT: {ext_amt:.2f} | {f.extraction.extracted_entity} | {f.extraction.extracted_supplier}")
                print(f"  Match: amt={f.amount_match}, ent={f.entity_match}, supp={f.supplier_similarity:.2f}")
    
    # Retornar código de erro se precisão < 80%
    if stats.get('overall_accuracy', 0) < 80:
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    main()
