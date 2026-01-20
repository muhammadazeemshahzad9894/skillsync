{
type: "file_content",
fileName: "src/evaluation/__init__.py",
content: """
Evaluation Module

Exports metrics, evaluators, and tracking tools.
"""

from .metrics import (
    TeamEvaluator,
    TeamQualityMetrics,
    LatencyTracker,
    ExtractionEvaluator,
    ExtractionMetrics,
    EXTRACTION_TEST_SET,
    format_score_with_icon,
    get_overall_status
)

__all__ = [
    'TeamEvaluator',
    'TeamQualityMetrics',
    'LatencyTracker',
    'ExtractionEvaluator',
    'ExtractionMetrics',
    'EXTRACTION_TEST_SET',
    'format_score_with_icon',
    'get_overall_status'
]
"""
}