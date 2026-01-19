"""
Matching Module

Handles candidate embedding generation, similarity search, and retrieval.
"""

from .embeddings import EmbeddingManager
from .retrieval import CandidateRetriever

__all__ = ["EmbeddingManager", "CandidateRetriever"]
