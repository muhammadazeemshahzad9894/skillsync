"""
SkillSync Evaluation Module

Exports:
- ExtractionEvaluator: Smart evaluation with fuzzy matching
- ExtractionMetrics: Metrics data class
- EXTRACTION_TEST_SET: Built-in test cases
"""

from .metrics import (
    ExtractionEvaluator,
    ExtractionMetrics,
    EXTRACTION_TEST_SET,
    are_skills_equivalent,
    are_roles_similar
)

__all__ = [
    "ExtractionEvaluator",
    "ExtractionMetrics",
    "EXTRACTION_TEST_SET",
    "are_skills_equivalent",
    "are_roles_similar"
]
