"""
Candidate Retrieval Module

Handles semantic search and ranking of candidates based on project requirements.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingManager
from ..preprocessing.normalizer import normalizer

# Configure logging
logger = logging.getLogger(__name__)


class CandidateRetriever:
    """
    Retrieves and ranks candidates based on semantic similarity
    to project requirements.
    
    Example:
        retriever = CandidateRetriever(profiles, embedding_manager)
        candidates = retriever.search("Python backend developer", top_k=10)
    """
    
    def __init__(
        self,
        profiles: List[Dict[str, Any]],
        embedding_manager: EmbeddingManager = None
    ):
        """
        Initialize retriever with candidate profiles.
        
        Args:
            profiles: List of candidate profile dictionaries
            embedding_manager: Optional embedding manager instance
        """
        self.profiles = profiles
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._embeddings: Optional[np.ndarray] = None
    
    @property
    def embeddings(self) -> np.ndarray:
        """Lazily compute and cache profile embeddings."""
        if self._embeddings is None:
            logger.info(f"Computing embeddings for {len(self.profiles)} profiles...")
            self._embeddings = self.embedding_manager.encode_profiles(self.profiles)
        return self._embeddings
    
    def refresh_embeddings(self) -> None:
        """Force recomputation of embeddings."""
        self._embeddings = None
        _ = self.embeddings
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for candidates matching a query.
        
        Args:
            query: Search query (natural language or keywords)
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold
            
        Returns:
            List of candidate profiles with match_score added
        """
        if len(self.embeddings) == 0:
            logger.warning("No embeddings available for search")
            return []
        
        # Encode query
        query_embedding = self.embedding_manager.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Build results
        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                break
            
            candidate = self.profiles[idx].copy()
            candidate["match_score"] = score
            results.append(candidate)
        
        logger.info(f"Found {len(results)} candidates for query")
        return results
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 20,
        min_experience: float = None,
        max_experience: float = None,
        required_skills: List[str] = None,
        target_roles: List[str] = None,
        industries: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with additional filtering constraints.
        
        Args:
            query: Search query
            top_k: Maximum results
            min_experience: Minimum years of experience
            max_experience: Maximum years of experience
            required_skills: Skills that must be present
            target_roles: Preferred role types
            industries: Preferred industries
            
        Returns:
            Filtered and ranked candidate list
        """
        # Get initial results (fetch more to account for filtering)
        initial_k = min(top_k * 3, len(self.profiles))
        candidates = self.search(query, top_k=initial_k)
        
        # Apply filters
        filtered = []
        for candidate in candidates:
            # Experience filter
            if min_experience is not None or max_experience is not None:
                try:
                    exp = float(candidate.get("metadata", {}).get("work_experience_years", 0))
                    if min_experience and exp < min_experience:
                        continue
                    if max_experience and exp > max_experience:
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Skills filter
            if required_skills:
                candidate_skills = set(
                    s.lower() for s in 
                    candidate.get("technical", {}).get("skills", [])
                )
                normalized_required = set(
                    normalizer.normalize_skill(s).lower() 
                    for s in required_skills
                )
                # Require at least partial match
                if not candidate_skills & normalized_required:
                    continue
            
            # Role filter
            if target_roles:
                candidate_role = candidate.get("role", "").lower()
                if not any(r.lower() in candidate_role for r in target_roles):
                    # Check dev_type as well
                    dev_type = candidate.get("metadata", {}).get("dev_type", "").lower()
                    if not any(r.lower() in dev_type for r in target_roles):
                        continue
            
            # Industry filter
            if industries:
                candidate_industry = candidate.get("metadata", {}).get("industry", "General").lower()
                if not any(ind.lower() in candidate_industry for ind in industries):
                    continue
            
            filtered.append(candidate)
            
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def get_similar_candidates(
        self,
        candidate_id: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find candidates similar to a given candidate.
        
        Args:
            candidate_id: ID of the reference candidate
            top_k: Number of similar candidates to return
            exclude_self: Whether to exclude the reference candidate
            
        Returns:
            List of similar candidates
        """
        # Find the reference candidate
        ref_idx = None
        for i, profile in enumerate(self.profiles):
            if profile.get("id") == candidate_id:
                ref_idx = i
                break
        
        if ref_idx is None:
            logger.warning(f"Candidate {candidate_id} not found")
            return []
        
        # Get embedding for reference
        ref_embedding = self.embeddings[ref_idx].reshape(1, -1)
        
        # Compute similarities
        scores = cosine_similarity(ref_embedding, self.embeddings)[0]
        
        # Rank
        ranked_indices = np.argsort(scores)[::-1]
        
        # Build results
        results = []
        for idx in ranked_indices:
            if exclude_self and idx == ref_idx:
                continue
            
            candidate = self.profiles[idx].copy()
            candidate["similarity_score"] = float(scores[idx])
            results.append(candidate)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def calculate_team_cohesion(
        self,
        team_members: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate average pairwise similarity within a team.
        
        Higher cohesion means team members have similar backgrounds.
        
        Args:
            team_members: List of team member profiles
            
        Returns:
            Average cohesion score (0-1)
        """
        if len(team_members) < 2:
            return 1.0
        
        # Get embeddings for team members
        member_embeddings = self.embedding_manager.encode_profiles(team_members)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(member_embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(team_members)
        total_sim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarities[i, j]
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    def calculate_skill_coverage(
        self,
        team_members: List[Dict[str, Any]],
        required_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate how well a team covers required skills.
        
        Args:
            team_members: List of team member profiles
            required_skills: List of required skills
            
        Returns:
            Tuple of (coverage_ratio, covered_skills, missing_skills)
        """
        # Collect all team skills
        team_skills = set()
        for member in team_members:
            skills = member.get("technical", {}).get("skills", [])
            normalized = [normalizer.normalize_skill(s).lower() for s in skills]
            team_skills.update(normalized)
        
        # Normalize required skills
        required_normalized = set(
            normalizer.normalize_skill(s).lower() 
            for s in required_skills
        )
        
        # Calculate coverage
        covered = team_skills & required_normalized
        missing = required_normalized - team_skills
        
        coverage_ratio = len(covered) / len(required_normalized) if required_normalized else 1.0
        
        return coverage_ratio, list(covered), list(missing)
