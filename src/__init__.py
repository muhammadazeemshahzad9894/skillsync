"""
SkillSync - AI-Powered Team Formation System

A comprehensive system for intelligent team assembly using:
- LLM-based requirement extraction
- Semantic similarity matching
- Multiple formation strategies
- Constraint validation
- Quality evaluation

Modules:
    preprocessing: Data cleaning, normalization, and parsing
    extraction: LLM-based information extraction
    matching: Embedding generation and candidate retrieval
    team_formation: Team assembly strategies and constraints
    evaluation: Quality metrics and benchmarking

Example:
    from src import SkillSyncEngine
    
    engine = SkillSyncEngine()
    results = engine.form_teams("Build a fintech mobile app", team_size=4)
"""

__version__ = "1.0.0"
__author__ = "Group 45"

from .engine import SkillSyncEngine

__all__ = ["SkillSyncEngine", "__version__", "__author__"]
