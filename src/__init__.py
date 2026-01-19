"""
SkillSync AI v2 - Enhanced Team Formation System

Features:
- Chain-of-prompts LLM extraction
- StackOverflow CSV support (28 columns)
- Availability filtering
- Comprehensive evaluation with test set
- Non-tech friendly UI
"""

__version__ = "2.0.0"
__author__ = "Group 45"

from .engine import SkillSyncEngine

__all__ = ["SkillSyncEngine", "__version__", "__author__"]
