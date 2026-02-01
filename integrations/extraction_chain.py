"""
SRDA V4.0 - Extraction Chain of Responsibility
==============================================
The 'Brain' of the extraction process.
Orchestrates the tiered extraction logic using a Chain of Responsibility pattern.

Levels:
1. Sidecar Metadata (JSON from Gmail/etc) - Instant (Ground Truth)
2. Digital Extraction (PDF Text + Spatial) - Fast
3. Visual OCR (Paddle/Surya) - Heavy
4. Oracle (Gemini Flash) - Last Resort
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from database import DocumentType, SRDADatabase
# Import existing tools (we will wrap them)
from spatial_extractor import get_spatial_extractor
from srda_rural_core import parse_brl_currency

# Setup Logger
logger = logging.getLogger('srda.chain')

@dataclass
class ExtractionContext:
    file_path: Path
    text_content: str = ""
    doc_type: str = "UNKNOWN"
    
    # Accumulated Results
    amount_cents: int = 0
    supplier_name: Optional[str] = None
    due_date: Optional[str] = None
    emission_date: Optional[str] = None
    cnpj: Optional[str] = None
    
    # Metadata
    confidence: float = 0.0
    extraction_path: str = "pending"
    needs_review: bool = False
    
    def is_complete(self) -> bool:
        """Check if we have the minimum viable data (Amount + Supplier + Date)."""
        return (
            self.amount_cents > 0 and 
            self.supplier_name is not None and 
            (self.due_date is not None or self.emission_date is not None)
            and self.confidence > 0.8
        )

class ExtractionChain:
    def __init__(self, db: SRDADatabase):
        self.db = db
        # Lazy load voters here if needed
        self._surya = None
        self._gemini = None

    def process(self, file_path: Path) -> Dict[str, Any]:
        """Main entry point."""
        ctx = ExtractionContext(file_path=file_path)
        
        # 1. Sidecar (Metadata)
        self._step_sidecar(ctx)
        if ctx.is_complete():
            return self._finalize(ctx)
            
        # 2. Digital (Text)
        self._step_digital(ctx)
        if ctx.is_complete():
            return self._finalize(ctx)
            
        # 3. OCR (Visual)
        # Only if we missing critical data
        if not ctx.amount_cents or not ctx.supplier_name:
             self._step_ocr(ctx)
        
        # 4. Oracle (Gemini)
        # If still failing or strictly requested
        if not ctx.amount_cents or not ctx.supplier_name or ctx.confidence < 0.6:
            self._step_oracle(ctx)
            
        return self._finalize(ctx)

    def _step_sidecar(self, ctx: ExtractionContext):
        """Check for .json or .meta sidecar files."""
        try:
            # Look for filename.pdf.json or filename.json
            candidates = [
                ctx.file_path.with_suffix('.json'),
                ctx.file_path.with_name(ctx.file_path.name + ".json")
            ]
            
            for sidecar in candidates:
                if sidecar.exists():
                    logger.info(f"[CHAIN] Found Metadata Sidecar: {sidecar.name}")
                    with open(sidecar, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Mapping
                        if "amount" in data:
                            # Handle "150.00" or 15000
                            val = data["amount"]
                            if isinstance(val, (int, float)):
                                ctx.amount_cents = int(val * 100) if val < 1000000 else int(val) # Heuristic
                            elif isinstance(val, str):
                                ctx.amount_cents = parse_brl_currency(val) * 100
                                
                        if "supplier_name" in data: ctx.supplier_name = data["supplier_name"]
                        if "date" in data: ctx.due_date = data["date"]
                        
                        ctx.extraction_path = "sidecar_metadata"
                        ctx.confidence = 1.0 # Trust sidecar implicitely
                    break
        except Exception as e:
            logger.warning(f"Sidecar error: {e}")


    def _step_digital(self, ctx: ExtractionContext):
        """Use Regex & Spatial Extractor (Fast Path)."""
        import fitz
        try:
            doc = fitz.open(str(ctx.file_path))
            page = doc[0]
            ctx.text_content = page.get_text()
            
            # 1. Classification
            # TODO: Use a proper classifier, for now simple heuristics
            upper_text = ctx.text_content.upper()
            if "DANFE" in upper_text: ctx.doc_type = "NFE"
            elif "BOLETO" in upper_text: ctx.doc_type = "BOLETO"
            
            # 2. Digital Regex
            # We can instantiate a lightweight RegexVoter here or use static patterns
            # For robustness, we will try to use the SpatialExtractor first if available
            spatial = get_spatial_extractor(str(ctx.file_path))
            if spatial:
                # Digital Amount
                if ctx.doc_type == "NFE":
                    val = spatial.extract_value_right("VALOR TOTAL DA NOTA")
                else:
                    val = spatial.extract_value_below("VALOR DO DOCUMENTO")
                    
                if val:
                    ctx.amount_cents = int(parse_brl_currency(val) * 100)
                    ctx.confidence = 0.9
                    ctx.extraction_path = "digital_spatial"
                    
                # Supplier Header
                supp = spatial.extract_supplier_header()
                if supp: ctx.supplier_name = supp
                
            doc.close()
        except Exception as e:
            logger.error(f"Digital step failed: {e}")

    def _step_ocr(self, ctx: ExtractionContext):
        """Use Ensemble (Paddle/Surya) via EnsembleExtractor."""
        try:
            from ensemble_extractor import EnsembleExtractor
            # We assume EnsembleExtractor is robust enough to handle instantiation
            # Ideally we should inject it, but for now we instantiate on demand (or use singleton)
            ensemble = EnsembleExtractor(high_accuracy=True) 
            
            # This runs the full pipeline (Vision + OCR + Regex)
            # It's a bit heavy, but it's Level 3.
            res = ensemble.extract_from_pdf(str(ctx.file_path))
            
            if res.amount_cents > 0:
                ctx.amount_cents = res.amount_cents
            if res.fornecedor:
                ctx.supplier_name = res.fornecedor
            if res.due_date:
                ctx.due_date = res.due_date
            if res.emission_date:
                ctx.emission_date = res.emission_date
                
            # Update confidence/metadata
            if res.confidence > ctx.confidence:
                ctx.confidence = res.confidence
                ctx.extraction_path = "ensemble_ocr"
                
        except ImportError:
            logger.warning("EnsembleExtractor not available.")
        except Exception as e:
            logger.error(f"OCR step failed: {e}")

    def _step_oracle(self, ctx: ExtractionContext):
        """Google Gemini Flash (Level 4)."""
        try:
            from voters.gemini_voter import get_gemini_voter
            voter = get_gemini_voter()
            if voter.is_available():
                logger.info(f"[CHAIN] Invoking Oracle (Gemini) for {ctx.file_path.name}")
                res = voter.extract(str(ctx.file_path))
                
                if res.success:
                    data = res.data
                    if data.get('amount'): 
                        ctx.amount_cents = int(data['amount'] * 100)
                    if data.get('supplier_name'): 
                        ctx.supplier_name = data['supplier_name']
                    if data.get('due_date'): 
                        ctx.due_date = data['due_date']
                        
                    ctx.confidence = 0.95
                    ctx.extraction_path = "gemini_oracle"
        except ImportError:
            logger.warning("GeminiVoter not available.")
        except Exception as e:
            logger.error(f"Oracle step failed: {e}")

        
    def _finalize(self, ctx: ExtractionContext) -> Dict[str, Any]:
        """Convert Context to Dict result."""
        
        # V4.0 Feature: Supplier Name Resolution (Knowledge Base)
        if ctx.supplier_name:
             # Look up in aliases table (e.g. 'UBER TRIP' -> 'UBER')
             resolved = self.db.get_resolved_supplier(ctx.supplier_name)
             if resolved != ctx.supplier_name:
                 logger.info(f"[CHAIN] Resolved Supplier: '{ctx.supplier_name}' -> '{resolved}'")
                 ctx.supplier_name = resolved

        return {
            "amount_cents": int(ctx.amount_cents),
            "supplier_name": ctx.supplier_name,
            "due_date": ctx.due_date,
            "emission_date": ctx.emission_date,
            "doc_type": ctx.doc_type,
            "confidence": ctx.confidence,
            "extraction_path": ctx.extraction_path
        }

# Global Chain Instance
_chain_instance = None
def get_extraction_chain(db=None) -> ExtractionChain:
    global _chain_instance
    if not _chain_instance:
        if db is None: db = SRDADatabase()
        _chain_instance = ExtractionChain(db)
    return _chain_instance
