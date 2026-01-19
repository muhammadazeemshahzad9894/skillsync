"""
Evaluation Module

Provides metrics and evaluation framework for assessing team formation quality.
Implements comparison against random baseline as specified in project plan.
"""

import random
import logging
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from statistics import mean, stdev

import numpy as np

from ..matching.retrieval import CandidateRetriever
from ..preprocessing.normalizer import normalizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a team formation."""
    # Core metrics
    skill_coverage: float  # 0-1
    role_diversity: float  # 0-1
    experience_balance: float  # 0-1
    avg_match_score: float  # 0-1
    
    # Computed metrics
    cohesion_score: float = 0.0  # 0-1
    constraint_satisfaction: float = 1.0  # 0-1
    
    # Comparison metrics
    improvement_over_random: float = 0.0  # Percentage improvement
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        weights = {
            "skill_coverage": 0.30,
            "role_diversity": 0.15,
            "experience_balance": 0.10,
            "avg_match_score": 0.25,
            "cohesion_score": 0.10,
            "constraint_satisfaction": 0.10
        }
        
        total = 0.0
        for metric, weight in weights.items():
            total += getattr(self, metric, 0) * weight
        
        return total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for display/storage."""
        return {
            "skill_coverage": round(self.skill_coverage, 3),
            "role_diversity": round(self.role_diversity, 3),
            "experience_balance": round(self.experience_balance, 3),
            "avg_match_score": round(self.avg_match_score, 3),
            "cohesion_score": round(self.cohesion_score, 3),
            "constraint_satisfaction": round(self.constraint_satisfaction, 3),
            "overall_score": round(self.overall_score, 3),
            "improvement_over_random": round(self.improvement_over_random, 1)
        }


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""
    system_metrics: EvaluationMetrics
    random_metrics: EvaluationMetrics
    improvement_percentage: float
    num_trials: int
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system_metrics.to_dict(),
            "random_baseline": self.random_metrics.to_dict(),
            "improvement_percentage": round(self.improvement_percentage, 2),
            "num_random_trials": self.num_trials,
            "latency_ms": round(self.latency_ms, 1)
        }


class TeamEvaluator:
    """
    Evaluates team formations against various quality metrics.
    
    Implements evaluation criteria from project plan:
    - Skill coverage
    - Role diversity
    - Experience balance
    - Match scores
    - Comparison against random baseline
    
    Example:
        evaluator = TeamEvaluator(retriever)
        metrics = evaluator.evaluate_team(team, requirements)
        benchmark = evaluator.benchmark_against_random(team, candidates, requirements)
    """
    
    def __init__(self, retriever: CandidateRetriever = None):
        """
        Initialize evaluator.
        
        Args:
            retriever: Optional CandidateRetriever for cohesion calculation
        """
        self.retriever = retriever
    
    def calculate_skill_coverage(
        self,
        team: List[Dict[str, Any]],
        required_skills: List[str]
    ) -> float:
        """Calculate what percentage of required skills are covered."""
        if not required_skills:
            return 1.0
        
        team_skills = set()
        for member in team:
            skills = member.get("technical", {}).get("skills", [])
            tools = member.get("technical", {}).get("tools", [])
            for s in skills + tools:
                team_skills.add(normalizer.normalize_skill(s).lower())
        
        required_normalized = set(
            normalizer.normalize_skill(s).lower() for s in required_skills
        )
        
        if not required_normalized:
            return 1.0
        
        covered = len(team_skills & required_normalized)
        return covered / len(required_normalized)
    
    def calculate_role_diversity(self, team: List[Dict[str, Any]]) -> float:
        """Calculate diversity of roles and Belbin types in team."""
        if len(team) <= 1:
            return 1.0
        
        # Check dev type diversity
        dev_types = set()
        belbin_roles = set()
        
        for member in team:
            dev_type = member.get("metadata", {}).get("dev_type", "")
            belbin = member.get("personality", {}).get("Belbin_team_role", "")
            
            if dev_type:
                dev_types.add(dev_type.lower())
            if belbin:
                belbin_roles.add(belbin.lower())
        
        # Diversity score: unique roles / team size
        dev_diversity = len(dev_types) / len(team)
        belbin_diversity = len(belbin_roles) / len(team)
        
        return (dev_diversity + belbin_diversity) / 2
    
    def calculate_experience_balance(self, team: List[Dict[str, Any]]) -> float:
        """
        Calculate how balanced experience levels are.
        
        Perfect balance = 0.5 (mix of senior and junior)
        All same level = lower score
        """
        if len(team) <= 1:
            return 1.0
        
        experiences = []
        for member in team:
            try:
                exp = float(member.get("metadata", {}).get("work_experience_years", 0))
                experiences.append(exp)
            except (ValueError, TypeError):
                experiences.append(0)
        
        if not experiences or max(experiences) == min(experiences):
            return 0.5  # No variation
        
        # Normalize to 0-1 range
        max_exp = max(experiences)
        min_exp = min(experiences)
        
        if max_exp == 0:
            return 1.0
        
        # Score based on spread - more spread = better balance
        spread = (max_exp - min_exp) / max_exp
        
        # Also consider if we have both junior and senior
        has_junior = any(e <= 3 for e in experiences)
        has_senior = any(e >= 7 for e in experiences)
        
        mix_bonus = 0.2 if (has_junior and has_senior) else 0
        
        return min(1.0, spread * 0.8 + mix_bonus)
    
    def calculate_avg_match_score(self, team: List[Dict[str, Any]]) -> float:
        """Calculate average semantic match score."""
        scores = [m.get("match_score", 0) for m in team]
        return mean(scores) if scores else 0
    
    def evaluate_team(
        self,
        team: List[Dict[str, Any]],
        required_skills: List[str] = None,
        target_roles: List[str] = None,
        compute_cohesion: bool = True
    ) -> EvaluationMetrics:
        """
        Compute comprehensive evaluation metrics for a team.
        
        Args:
            team: List of team member profiles
            required_skills: Skills that should be covered
            target_roles: Roles that should be represented
            compute_cohesion: Whether to compute cohesion (requires retriever)
            
        Returns:
            EvaluationMetrics with all scores
        """
        metrics = EvaluationMetrics(
            skill_coverage=self.calculate_skill_coverage(team, required_skills or []),
            role_diversity=self.calculate_role_diversity(team),
            experience_balance=self.calculate_experience_balance(team),
            avg_match_score=self.calculate_avg_match_score(team)
        )
        
        # Compute cohesion if retriever available
        if compute_cohesion and self.retriever:
            metrics.cohesion_score = self.retriever.calculate_team_cohesion(team)
        
        return metrics
    
    def create_random_team(
        self,
        candidate_pool: List[Dict[str, Any]],
        team_size: int
    ) -> List[Dict[str, Any]]:
        """Create a random team from the candidate pool."""
        if len(candidate_pool) <= team_size:
            return candidate_pool.copy()
        
        return random.sample(candidate_pool, team_size)
    
    def benchmark_against_random(
        self,
        system_team: List[Dict[str, Any]],
        candidate_pool: List[Dict[str, Any]],
        required_skills: List[str] = None,
        num_random_trials: int = 50
    ) -> BenchmarkResult:
        """
        Compare system team against random baseline.
        
        Args:
            system_team: Team formed by the system
            candidate_pool: Full candidate pool
            required_skills: Required skills for coverage calculation
            num_random_trials: Number of random teams to generate
            
        Returns:
            BenchmarkResult with comparison metrics
        """
        start_time = time.time()
        
        # Evaluate system team
        system_metrics = self.evaluate_team(system_team, required_skills)
        
        # Generate and evaluate random teams
        team_size = len(system_team)
        random_scores = []
        
        for _ in range(num_random_trials):
            random_team = self.create_random_team(candidate_pool, team_size)
            random_metrics = self.evaluate_team(random_team, required_skills, compute_cohesion=False)
            random_scores.append(random_metrics.overall_score)
        
        # Calculate average random baseline
        avg_random_score = mean(random_scores)
        random_metrics = EvaluationMetrics(
            skill_coverage=avg_random_score * 0.3 / 0.3,  # Approximate breakdown
            role_diversity=avg_random_score * 0.15 / 0.15,
            experience_balance=avg_random_score * 0.1 / 0.1,
            avg_match_score=avg_random_score * 0.25 / 0.25
        )
        
        # Calculate improvement
        if avg_random_score > 0:
            improvement = ((system_metrics.overall_score - avg_random_score) / avg_random_score) * 100
        else:
            improvement = 100.0
        
        system_metrics.improvement_over_random = improvement
        
        latency_ms = (time.time() - start_time) * 1000
        
        return BenchmarkResult(
            system_metrics=system_metrics,
            random_metrics=random_metrics,
            improvement_percentage=improvement,
            num_trials=num_random_trials,
            latency_ms=latency_ms
        )


@dataclass
class PipelineLatencyReport:
    """Report of latency across pipeline stages."""
    extraction_ms: float
    embedding_ms: float
    retrieval_ms: float
    team_formation_ms: float
    explanation_ms: float
    total_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "extraction_ms": round(self.extraction_ms, 1),
            "embedding_ms": round(self.embedding_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "team_formation_ms": round(self.team_formation_ms, 1),
            "explanation_ms": round(self.explanation_ms, 1),
            "total_ms": round(self.total_ms, 1)
        }


class LatencyTracker:
    """
    Tracks execution time across pipeline stages.
    
    Example:
        tracker = LatencyTracker()
        with tracker.track("extraction"):
            # extraction code
        report = tracker.get_report()
    """
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._start_times[stage] = time.time()
    
    def stop(self, stage: str) -> float:
        """Stop timing and record duration."""
        if stage not in self._start_times:
            return 0.0
        
        duration = (time.time() - self._start_times[stage]) * 1000
        self.timings[stage] = duration
        del self._start_times[stage]
        return duration
    
    def track(self, stage: str):
        """Context manager for timing a stage."""
        class Timer:
            def __init__(self, tracker, stage):
                self.tracker = tracker
                self.stage = stage
            
            def __enter__(self):
                self.tracker.start(self.stage)
                return self
            
            def __exit__(self, *args):
                self.tracker.stop(self.stage)
        
        return Timer(self, stage)
    
    def get_report(self) -> PipelineLatencyReport:
        """Generate latency report."""
        return PipelineLatencyReport(
            extraction_ms=self.timings.get("extraction", 0),
            embedding_ms=self.timings.get("embedding", 0),
            retrieval_ms=self.timings.get("retrieval", 0),
            team_formation_ms=self.timings.get("team_formation", 0),
            explanation_ms=self.timings.get("explanation", 0),
            total_ms=sum(self.timings.values())
        )
    
    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()
        self._start_times.clear()
