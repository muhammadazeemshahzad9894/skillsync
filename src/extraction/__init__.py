"""
SkillSync Extraction Module

Exports:
- LLMExtractor: Main extraction class
- ProjectRequirements: Data class for extracted requirements
"""

from .llm_extractor import LLMExtractor, ProjectRequirements, ALLOWED_ROLES, DOMAINS

__all__ = [
    "LLMExtractor",
    "ProjectRequirements",
    "ALLOWED_ROLES",
    "DOMAINS"
]
