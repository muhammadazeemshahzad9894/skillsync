{
type: "file_content",
fileName: "src/extraction/__init__.py",
content: """
Extraction Module

Exports the LLM extractor and related configuration classes.
"""

from .llm_extractor import (
    ChainedLLMExtractor,
    LLMExtractor,
    ProjectRequirements,
    ExtractionConfig,
    ALLOWED_ROLES,
    DOMAINS
)

__all__ = [
    'ChainedLLMExtractor',
    'LLMExtractor',
    'ProjectRequirements',
    'ExtractionConfig',
    'ALLOWED_ROLES',
    'DOMAINS'
]
"""
}