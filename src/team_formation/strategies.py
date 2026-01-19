"""
Team Formation Strategies Module

Implements multiple team formation strategies with constraint awareness
and optimization.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..preprocessing.normalizer import normalizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TeamFormationResult:
    """Result of a team formation strategy."""
    strategy_name: str
    rationale: str
    members: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_analysis: str = ""
    
    @property
    def average_match_score(self) -> float:
        scores = [m.get("match_score", 0) for m in self.members]
        return sum(scores) / len(scores) if scores else 0
    
    @property
    def total_experience(self) -> float:
        total = 0
        for m in self.members:
            try:
                total += float(m.get("metadata", {}).get("work_experience_years", 0))
            except (ValueError, TypeError):
                pass
        return total


class BaseStrategy(ABC):
    """Abstract base class for team formation strategies."""
    
    name: str = "Base Strategy"
    rationale: str = "Base strategy rationale"
    
    @abstractmethod
    def select_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None
    ) -> TeamFormationResult:
        """
        Select team members from candidate pool.
        
        Args:
            candidate_pool: Ranked list of candidates
            team_size: Number of team members to select
            requirements: Optional project requirements
            
        Returns:
            TeamFormationResult with selected members
        """
        pass


class ExpertStrategy(BaseStrategy):
    """
    Selects top candidates by match score.
    
    Best for: Projects requiring deep expertise in specific skills.
    """
    
    name = "The Expert Team"
    rationale = "Selected based on highest technical match scores for maximum expertise alignment."
    
    def select_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None
    ) -> TeamFormationResult:
        # Already sorted by match_score from retrieval
        selected = candidate_pool[:team_size]
        
        return TeamFormationResult(
            strategy_name=self.name,
            rationale=self.rationale,
            members=selected,
            metadata={"strategy_type": "expert"}
        )


class BalancedStrategy(BaseStrategy):
    """
    Balances team with mix of experience levels.
    
    Best for: Projects needing knowledge transfer and mentorship.
    """
    
    name = "The Balanced Team"
    rationale = "Mix of senior and junior members for knowledge transfer and fresh perspectives."
    
    def select_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None
    ) -> TeamFormationResult:
        
        def get_experience(candidate):
            try:
                return float(candidate.get("metadata", {}).get("work_experience_years", 0))
            except (ValueError, TypeError):
                return 0.0
        
        # Sort by experience
        sorted_pool = sorted(candidate_pool, key=get_experience, reverse=True)
        
        selected = []
        used_indices = set()
        
        # Add most senior
        if sorted_pool:
            selected.append(sorted_pool[0])
            used_indices.add(0)
        
        # Add most junior (for balance)
        if len(sorted_pool) > 1 and len(selected) < team_size:
            selected.append(sorted_pool[-1])
            used_indices.add(len(sorted_pool) - 1)
        
        # Fill remaining from middle, prioritizing match score
        remaining = [
            (i, c) for i, c in enumerate(sorted_pool) 
            if i not in used_indices
        ]
        remaining.sort(key=lambda x: x[1].get("match_score", 0), reverse=True)
        
        for i, candidate in remaining:
            if len(selected) >= team_size:
                break
            selected.append(candidate)
        
        return TeamFormationResult(
            strategy_name=self.name,
            rationale=self.rationale,
            members=selected,
            metadata={
                "strategy_type": "balanced",
                "experience_range": {
                    "min": min(get_experience(c) for c in selected) if selected else 0,
                    "max": max(get_experience(c) for c in selected) if selected else 0
                }
            }
        )


class DiverseStrategy(BaseStrategy):
    """
    Maximizes diversity in team roles and backgrounds.
    
    Best for: Innovation projects needing varied perspectives.
    """
    
    name = "The Complementary Team"
    rationale = "Diverse Belbin team roles and backgrounds for complementary collaboration."
    
    def select_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None
    ) -> TeamFormationResult:
        selected = []
        used_belbin_roles = set()
        used_dev_types = set()
        
        # First pass: maximize role diversity
        for candidate in candidate_pool:
            if len(selected) >= team_size:
                break
            
            belbin = candidate.get("personality", {}).get("Belbin_team_role", "Unknown")
            dev_type = candidate.get("metadata", {}).get("dev_type", "Unknown")
            
            # Prefer candidates with unique roles
            is_unique = belbin not in used_belbin_roles or dev_type not in used_dev_types
            
            if is_unique:
                selected.append(candidate)
                used_belbin_roles.add(belbin)
                used_dev_types.add(dev_type)
        
        # Fill remaining with best match scores
        if len(selected) < team_size:
            remaining = [c for c in candidate_pool if c not in selected]
            remaining.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            for candidate in remaining:
                if len(selected) >= team_size:
                    break
                selected.append(candidate)
        
        return TeamFormationResult(
            strategy_name=self.name,
            rationale=self.rationale,
            members=selected,
            metadata={
                "strategy_type": "diverse",
                "belbin_roles": list(used_belbin_roles),
                "dev_types": list(used_dev_types)
            }
        )


class SkillCoverageStrategy(BaseStrategy):
    """
    Optimizes for maximum coverage of required skills.
    
    Best for: Projects with specific technical requirements.
    """
    
    name = "The Skill Coverage Team"
    rationale = "Optimized to cover maximum required skills with minimum overlap."
    
    def select_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None
    ) -> TeamFormationResult:
        required_skills = set()
        if requirements:
            keywords = requirements.get("technical_keywords", [])
            required_skills = set(normalizer.normalize_skill(s).lower() for s in keywords)
        
        selected = []
        covered_skills = set()
        
        while len(selected) < team_size and candidate_pool:
            best_candidate = None
            best_new_skills = 0
            best_idx = -1
            
            for i, candidate in enumerate(candidate_pool):
                if candidate in selected:
                    continue
                
                # Get candidate skills
                skills = candidate.get("technical", {}).get("skills", [])
                tools = candidate.get("technical", {}).get("tools", [])
                candidate_skills = set(
                    normalizer.normalize_skill(s).lower() 
                    for s in skills + tools
                )
                
                # Count new skills this candidate brings
                if required_skills:
                    new_skills = len((candidate_skills & required_skills) - covered_skills)
                else:
                    new_skills = len(candidate_skills - covered_skills)
                
                # Tie-breaker: match score
                if new_skills > best_new_skills or (
                    new_skills == best_new_skills and 
                    best_candidate and
                    candidate.get("match_score", 0) > best_candidate.get("match_score", 0)
                ):
                    best_candidate = candidate
                    best_new_skills = new_skills
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                skills = best_candidate.get("technical", {}).get("skills", [])
                tools = best_candidate.get("technical", {}).get("tools", [])
                covered_skills.update(
                    normalizer.normalize_skill(s).lower() 
                    for s in skills + tools
                )
        
        return TeamFormationResult(
            strategy_name=self.name,
            rationale=self.rationale,
            members=selected,
            metadata={
                "strategy_type": "skill_coverage",
                "skills_covered": len(covered_skills & required_skills) if required_skills else len(covered_skills),
                "skills_required": len(required_skills)
            }
        )


class TeamFormationEngine:
    """
    Main engine for forming teams using multiple strategies.
    
    Example:
        engine = TeamFormationEngine()
        results = engine.form_teams(candidates, team_size, requirements)
    """
    
    def __init__(self):
        """Initialize with default strategies."""
        self.strategies = {
            "expert": ExpertStrategy(),
            "balanced": BalancedStrategy(),
            "diverse": DiverseStrategy(),
            "skill_coverage": SkillCoverageStrategy()
        }
    
    def add_strategy(self, key: str, strategy: BaseStrategy) -> None:
        """Add a custom strategy."""
        self.strategies[key] = strategy
    
    def form_teams(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int,
        requirements: Dict[str, Any] = None,
        strategy_keys: List[str] = None
    ) -> Dict[str, TeamFormationResult]:
        """
        Form teams using all (or specified) strategies.
        
        Args:
            candidate_pool: Ranked candidates from retrieval
            team_size: Desired team size
            requirements: Project requirements
            strategy_keys: Optional list of strategy keys to use
            
        Returns:
            Dictionary of strategy names to TeamFormationResult
        """
        if len(candidate_pool) < team_size:
            logger.warning(f"Candidate pool ({len(candidate_pool)}) smaller than team size ({team_size})")
        
        strategies_to_use = strategy_keys or list(self.strategies.keys())[:3]  # Default to first 3
        results = {}
        
        for key in strategies_to_use:
            if key not in self.strategies:
                logger.warning(f"Unknown strategy: {key}")
                continue
            
            strategy = self.strategies[key]
            logger.info(f"Applying strategy: {strategy.name}")
            
            result = strategy.select_team(candidate_pool, team_size, requirements)
            
            # Use readable key for display
            display_key = f"Option {chr(65 + len(results))}: {strategy.name}"
            results[display_key] = result
        
        return results


# Singleton engine instance
team_formation_engine = TeamFormationEngine()
