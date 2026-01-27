"""
Pattern Learner - Automatic regex pattern learning from user corrections.

Implements evolutionary pattern learning:
1. Captures user corrections (e.g., correct supplier name)
2. Analyzes raw text to find matching context
3. Generates regex pattern candidate
4. Saves to extraction_patterns.yaml for future use

Usage:
    learner = PatternLearner()
    learner.learn_from_correction(raw_text, field="fornecedor", corrected_value="EMPRESA XYZ")
"""

import re
import logging
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """A pattern learned from user correction."""
    field: str           # e.g., "fornecedor", "valor"
    doc_type: str        # e.g., "BOLETO", "NFE"
    pattern: str         # The regex pattern
    confidence: float    # Suggested confidence
    example_value: str   # The value it was learned from
    context: str         # Surrounding text context
    created_at: str      # Timestamp


class PatternLearner:
    """
    Learn regex patterns from user corrections.
    
    When a user manually corrects an extracted value, this class:
    1. Finds where that value appears in the raw text
    2. Analyzes the surrounding context
    3. Generates a regex pattern that would capture it
    4. Saves the pattern for future use
    """
    
    def __init__(self, patterns_file: str = "extraction_patterns.yaml"):
        """
        Initialize pattern learner.
        
        Args:
            patterns_file: YAML file to save learned patterns
        """
        self.patterns_file = Path(__file__).parent / patterns_file
        self.learned_patterns: List[LearnedPattern] = []
        
    def learn_from_correction(
        self, 
        raw_text: str, 
        field: str,
        corrected_value: str,
        doc_type: str = "GENERIC"
    ) -> Optional[LearnedPattern]:
        """
        Learn a pattern from a user correction.
        
        Args:
            raw_text: The full document text
            field: Field that was corrected ("fornecedor", "valor", etc.)
            corrected_value: The correct value provided by user
            doc_type: Document type for pattern categorization
            
        Returns:
            LearnedPattern if successful, None otherwise
        """
        if not corrected_value or not raw_text:
            return None
        
        # Find the corrected value in the text
        location = self._find_value_in_text(raw_text, corrected_value)
        if location is None:
            logger.warning(f"Corrected value '{corrected_value}' not found in text")
            return None
        
        start_idx, end_idx = location
        
        # Extract context around the value
        context = self._extract_context(raw_text, start_idx, end_idx)
        
        # Generate regex pattern
        pattern = self._generate_pattern(context, corrected_value, field)
        
        if not pattern:
            return None
        
        # Create learned pattern
        learned = LearnedPattern(
            field=field,
            doc_type=doc_type.upper(),
            pattern=pattern,
            confidence=0.85,  # User-learned patterns get high confidence
            example_value=corrected_value,
            context=context[:100],
            created_at=datetime.now().isoformat()
        )
        
        self.learned_patterns.append(learned)
        logger.info(f"Learned pattern for {field}: {pattern}")
        
        return learned
    
    def _find_value_in_text(self, text: str, value: str) -> Optional[Tuple[int, int]]:
        """Find the value in the text, return (start, end) indices."""
        # Try exact match first
        value_upper = value.upper()
        text_upper = text.upper()
        
        idx = text_upper.find(value_upper)
        if idx >= 0:
            return (idx, idx + len(value))
        
        # Try fuzzy match - find any word from the value
        words = value_upper.split()
        if len(words) >= 2:
            # Try to find the first two words together
            search_term = f"{words[0]}\\s+{words[1]}"
            match = re.search(search_term, text_upper)
            if match:
                return (match.start(), match.end())
        
        return None
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around the found value."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]
    
    def _generate_pattern(self, context: str, value: str, field: str) -> Optional[str]:
        """Generate a regex pattern from context and value."""
        # Clean value for regex
        value_escaped = re.escape(value)
        
        # Look for common prefix patterns
        prefixes = {
            "fornecedor": [
                r"(?:BENEFICI[AÁ]RIO|CEDENTE|FAVORECIDO)[:\s]*",
                r"(?:PRESTADOR)[:\s]*",
                r"(?:EMPRESA|RAZAO\s*SOCIAL)[:\s]*",
            ],
            "valor": [
                r"(?:VALOR\s*(?:TOTAL|DOCUMENTO)?)[:\s]*R?\$?\s*",
                r"(?:TOTAL)[:\s]*R?\$?\s*",
            ],
            "data": [
                r"(?:VENCIMENTO|VENC\.?)[:\s]*",
                r"(?:DATA)[:\s]*",
            ]
        }
        
        # Try to find which prefix precedes the value
        best_pattern = None
        
        for prefix in prefixes.get(field, []):
            # Check if prefix appears before value in context
            pattern_test = prefix + r"([^\n]{5,50})"
            match = re.search(pattern_test, context, re.IGNORECASE)
            if match:
                # Verify our value is captured
                if value.upper() in match.group(1).upper():
                    best_pattern = prefix + r"([^\n,]{5,50})"
                    break
        
        # Fallback: generic pattern based on field
        if not best_pattern:
            if field == "fornecedor":
                best_pattern = r"(?<=\n)([A-Z][A-Z\s\.\-]{4,40})(?=\s*(?:CNPJ|CPF|\n))"
            elif field == "valor":
                best_pattern = r"([\d\.]+,\d{2})"
            else:
                return None
        
        return best_pattern
    
    def save_to_yaml(self, pattern: LearnedPattern) -> bool:
        """
        Append learned pattern to extraction_patterns.yaml.
        
        Args:
            pattern: The pattern to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Load existing patterns
            data = {}
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            
            # Create section if needed
            if pattern.field not in data:
                data[pattern.field] = {}
            if pattern.doc_type not in data[pattern.field]:
                data[pattern.field][pattern.doc_type] = []
            
            # Add new pattern entry
            new_entry = {
                'pattern': pattern.pattern,
                'confidence': pattern.confidence,
                'learned': True,
                'example': pattern.example_value,
                'created': pattern.created_at
            }
            
            # Check for duplicates
            existing = data[pattern.field][pattern.doc_type]
            if not any(e.get('pattern') == pattern.pattern for e in existing):
                data[pattern.field][pattern.doc_type].append(new_entry)
            
            # Save
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"Saved learned pattern to {self.patterns_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
            return False
    
    def suggest_patterns(self, raw_text: str, field: str) -> List[str]:
        """
        Suggest regex patterns based on common document structures.
        
        This is useful for bootstrapping - analyzing a document and
        suggesting patterns that the user can validate.
        
        Args:
            raw_text: Document text
            field: Field to find patterns for
            
        Returns:
            List of suggested regex patterns
        """
        suggestions = []
        
        if field == "fornecedor":
            # Look for patterns after known labels
            labels = ["BENEFICIÁRIO", "CEDENTE", "PRESTADOR", "FAVORECIDO", "EMPRESA"]
            for label in labels:
                pattern = rf'{label}[:\s]*([A-Z][A-Z\s\.\-]{{5,40}})'
                matches = re.findall(pattern, raw_text, re.IGNORECASE)
                if matches:
                    suggestions.append((pattern, matches[0]))
                    
        elif field == "valor":
            # Look for monetary patterns with context
            patterns = [
                (r'VALOR\s*(?:TOTAL)?[:\s]*R?\$?\s*([\d\.]+,\d{2})', "VALOR TOTAL"),
                (r'\(=\)\s*(?:Valor|TOTAL)[:\s]*([\d\.]+,\d{2})', "Valor Documento"),
            ]
            for pattern, name in patterns:
                matches = re.findall(pattern, raw_text, re.IGNORECASE)
                if matches:
                    suggestions.append((pattern, matches[0]))
        
        return suggestions
    
    def get_learned_patterns(self) -> List[LearnedPattern]:
        """Return all learned patterns from this session."""
        return self.learned_patterns.copy()


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    learner = PatternLearner()
    
    # Simulated document text
    sample_text = """
    BOLETO BANCÁRIO
    
    BENEFICIÁRIO: EMPRESA TESTE LTDA
    CNPJ: 12.345.678/0001-90
    
    VALOR: R$ 1.234,56
    VENCIMENTO: 15/12/2025
    """
    
    # Simulate learning from correction
    result = learner.learn_from_correction(
        raw_text=sample_text,
        field="fornecedor",
        corrected_value="EMPRESA TESTE",
        doc_type="BOLETO"
    )
    
    if result:
        print(f"Learned pattern: {result.pattern}")
        print(f"Context: {result.context[:50]}...")
        
        # Optionally save
        # learner.save_to_yaml(result)
