"""
Comprehensive Evaluation Module

Implements evaluation framework for:
1. Extraction accuracy (precision/recall)
2. Team quality metrics
3. Comparison against random baseline
4. Latency tracking
5. User-friendly score display

Includes built-in test set for extraction evaluation.
"""

import random
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TEST SET FOR EXTRACTION EVALUATION
# ============================================================================

EXTRACTION_TEST_SET = [
    {
        "id": "test_001",
        "description": "We need a team to build a Fintech mobile app with React Native for iOS and Android. Backend should be Python with FastAPI, deployed on AWS. Need experience with payment integrations and security.",
        "expected": {
            "technical_keywords": ["React Native", "Python", "FastAPI", "AWS", "iOS", "Android"],
            "tools": [],
            "target_roles": ["Developer, mobile", "Developer, back-end", "Cloud infrastructure engineer"],
            "domain": "Fintech"
        }
    },
    {
        "id": "test_002", 
        "description": "Looking for a data science team to build ML models for healthcare diagnostics. Must know TensorFlow, PyTorch, and have experience with medical imaging. Python required, Jupyter for notebooks.",
        "expected": {
            "technical_keywords": ["TensorFlow", "PyTorch", "Python", "medical imaging"],
            "tools": ["Jupyter"],
            "target_roles": ["Data scientist or machine learning specialist"],
            "domain": "Healthcare"
        }
    },
    {
        "id": "test_003",
        "description": "E-commerce platform rebuild using React frontend, Node.js backend, PostgreSQL database. Need DevOps engineer for Kubernetes and Docker deployment on GCP.",
        "expected": {
            "technical_keywords": ["React", "Node.js", "PostgreSQL", "Kubernetes", "Docker", "GCP"],
            "tools": [],
            "target_roles": ["Developer, front-end", "Developer, back-end", "DevOps specialist", "Cloud infrastructure engineer"],
            "domain": "E-commerce"
        }
    },
    {
        "id": "test_004",
        "description": "IoT project for smart agriculture. Need embedded systems developers with C++ and Rust experience. MQTT protocol for device communication. AWS IoT for cloud backend.",
        "expected": {
            "technical_keywords": ["C++", "Rust", "MQTT", "AWS IoT", "IoT"],
            "tools": [],
            "target_roles": ["Developer, embedded applications or devices", "Cloud infrastructure engineer"],
            "domain": "Agriculture"
        }
    },
    {
        "id": "test_005",
        "description": "Full-stack web application for education platform. TypeScript everywhere - Next.js frontend, NestJS backend. MongoDB for data, deployed on Vercel and Railway.",
        "expected": {
            "technical_keywords": ["TypeScript", "Next.js", "NestJS", "MongoDB", "Vercel", "Railway"],
            "tools": [],
            "target_roles": ["Developer, full-stack"],
            "domain": "Education"
        }
    },
    {
        "id": "test_006",
        "description": "Security-focused project: build a penetration testing tool using Python. Need security professionals with experience in OWASP, vulnerability scanning. CI/CD with GitHub Actions.",
        "expected": {
            "technical_keywords": ["Python", "OWASP", "vulnerability scanning"],
            "tools": ["GitHub Actions"],
            "target_roles": ["Security professional", "Developer, back-end", "DevOps specialist"],
            "domain": "Cybersecurity"
        }
    },
    {
        "id": "test_007",
        "description": "Real-time analytics dashboard for manufacturing. Apache Kafka for streaming, Apache Spark for processing, Grafana for visualization. Need data engineers with 5+ years experience.",
        "expected": {
            "technical_keywords": ["Apache Kafka", "Apache Spark", "Grafana", "streaming"],
            "tools": [],
            "target_roles": ["Data engineer"],
            "domain": "Manufacturing",
            "min_experience": 5
        }
    },
    {
        "id": "test_008",
        "description": "Mobile game development using Unity and C#. Need 3D artists and game developers. Multiplayer backend with Photon. Target iOS and Android.",
        "expected": {
            "technical_keywords": ["Unity", "C#", "Photon", "3D", "iOS", "Android"],
            "tools": [],
            "target_roles": ["Developer, mobile"],
            "domain": "Gaming"
        }
    },
    {
        "id": "test_009",
        "description": "AI chatbot for customer service using GPT-4, LangChain, and vector databases. Python backend, React frontend. Need ML specialists and full-stack developers.",
        "expected": {
            "technical_keywords": ["GPT-4", "LangChain", "vector databases", "Python", "React"],
            "tools": [],
            "target_roles": ["Data scientist or machine learning specialist", "Developer, AI", "Developer, full-stack"],
            "domain": "General"
        }
    },
    {
        "id": "test_010",
        "description": "Site reliability engineering team for cloud infrastructure. Terraform for IaC, Prometheus and Grafana for monitoring. AWS and Azure multi-cloud. 24/7 on-call rotation.",
        "expected": {
            "technical_keywords": ["Terraform", "Prometheus", "Grafana", "AWS", "Azure", "IaC"],
            "tools": [],
            "target_roles": ["Engineer, site reliability", "Cloud infrastructure engineer", "DevOps specialist"],
            "domain": "Technology"
        }
    }
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExtractionMetrics:
    """Metrics for extraction evaluation."""
    precision: float
    recall: float
    f1_score: float
    domain_accuracy: float
    role_accuracy: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
            "domain_accuracy": round(self.domain_accuracy, 3),
            "role_accuracy": round(self.role_accuracy, 3)
        }


@dataclass
class TeamQualityMetrics:
    """Metrics for team quality evaluation."""
    skill_coverage: float  # 0-1
    role_diversity: float  # 0-1
    experience_balance: float  # 0-1
    avg_match_score: float  # 0-1
    availability_fit: float  # 0-1
    cohesion_score: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Weighted overall score."""
        weights = {
            "skill_coverage": 0.30,
            "role_diversity": 0.15,
            "experience_balance": 0.15,
            "avg_match_score": 0.20,
            "availability_fit": 0.10,
            "cohesion_score": 0.10
        }
        return sum(getattr(self, k, 0) * v for k, v in weights.items())
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "skill_coverage": round(self.skill_coverage, 3),
            "role_diversity": round(self.role_diversity, 3),
            "experience_balance": round(self.experience_balance, 3),
            "avg_match_score": round(self.avg_match_score, 3),
            "availability_fit": round(self.availability_fit, 3),
            "cohesion_score": round(self.cohesion_score, 3),
            "overall_score": round(self.overall_score, 3)
        }
    
    def get_status_icons(self) -> Dict[str, str]:
        """Get status icons for each metric."""
        def icon(val):
            if val >= 0.8:
                return "✅"
            elif val >= 0.5:
                return "⚠️"
            else:
                return "❌"
        
        return {
            "skill_coverage": icon(self.skill_coverage),
            "role_diversity": icon(self.role_diversity),
            "experience_balance": icon(self.experience_balance),
            "avg_match_score": icon(self.avg_match_score),
            "availability_fit": icon(self.availability_fit),
            "overall": icon(self.overall_score)
        }


@dataclass
class BenchmarkResult:
    """Results from benchmark against random baseline."""
    system_score: float
    random_avg_score: float
    random_std: float
    improvement_percentage: float
    num_trials: int
    p_value_estimate: str  # "significant", "marginal", "not significant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_score": round(self.system_score, 3),
            "random_avg_score": round(self.random_avg_score, 3),
            "random_std": round(self.random_std, 3),
            "improvement_percentage": round(self.improvement_percentage, 1),
            "num_trials": self.num_trials,
            "significance": self.p_value_estimate
        }


@dataclass
class LatencyReport:
    """Latency tracking across pipeline stages."""
    stages: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())
    
    def to_dict(self) -> Dict[str, float]:
        result = {f"{k}_ms": round(v, 1) for k, v in self.stages.items()}
        result["total_ms"] = round(self.total_ms, 1)
        return result
    
    def get_breakdown_display(self) -> List[Tuple[str, float, str]]:
        """Get display-friendly breakdown with percentages."""
        total = self.total_ms
        if total == 0:
            return []
        
        return [
            (stage, ms, f"{(ms/total)*100:.0f}%")
            for stage, ms in self.stages.items()
        ]


# ============================================================================
# EXTRACTION EVALUATOR
# ============================================================================

class ExtractionEvaluator:
    """
    Evaluates extraction quality against ground truth.
    """
    
    def __init__(self, test_set: List[Dict] = None):
        self.test_set = test_set or EXTRACTION_TEST_SET
    
    def _normalize_list(self, items: List[str]) -> set:
        """Normalize list items for comparison."""
        return set(item.lower().strip() for item in items if item)
    
    def _calculate_set_metrics(
        self,
        predicted: List[str],
        expected: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for list comparison."""
        pred_set = self._normalize_list(predicted)
        exp_set = self._normalize_list(expected)
        
        if not pred_set and not exp_set:
            return 1.0, 1.0, 1.0
        if not pred_set:
            return 0.0, 0.0, 0.0
        if not exp_set:
            return 0.0, 1.0, 0.0
        
        true_positives = len(pred_set & exp_set)
        
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(exp_set) if exp_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def evaluate_single(
        self,
        predicted: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate single extraction against expected."""
        results = {}
        
        # Skills/keywords
        p, r, f1 = self._calculate_set_metrics(
            predicted.get("technical_keywords", []),
            expected.get("technical_keywords", [])
        )
        results["skills"] = {"precision": p, "recall": r, "f1": f1}
        
        # Tools
        p, r, f1 = self._calculate_set_metrics(
            predicted.get("tools", []),
            expected.get("tools", [])
        )
        results["tools"] = {"precision": p, "recall": r, "f1": f1}
        
        # Roles
        p, r, f1 = self._calculate_set_metrics(
            predicted.get("target_roles", []),
            expected.get("target_roles", [])
        )
        results["roles"] = {"precision": p, "recall": r, "f1": f1}
        
        # Domain accuracy
        pred_domain = (predicted.get("domain") or "").lower()
        exp_domain = (expected.get("domain") or "").lower()
        results["domain_match"] = 1.0 if pred_domain == exp_domain or exp_domain in pred_domain else 0.0
        
        return results
    
    def evaluate_extractor(self, extractor) -> ExtractionMetrics:
        """
        Run full evaluation on test set.
        
        Args:
            extractor: Object with extract_requirements(description) method
            
        Returns:
            ExtractionMetrics with aggregate scores
        """
        all_results = []
        
        for test_case in self.test_set:
            try:
                # Run extraction
                result = extractor.extract_requirements(test_case["description"])
                
                # Convert to dict if needed
                if hasattr(result, "to_dict"):
                    predicted = result.to_dict()
                else:
                    predicted = result
                
                # Evaluate
                metrics = self.evaluate_single(predicted, test_case["expected"])
                metrics["test_id"] = test_case["id"]
                all_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Test {test_case['id']} failed: {e}")
                continue
        
        if not all_results:
            return ExtractionMetrics(0, 0, 0, 0, 0)
        
        # Aggregate metrics
        avg_precision = mean([r["skills"]["precision"] for r in all_results])
        avg_recall = mean([r["skills"]["recall"] for r in all_results])
        avg_f1 = mean([r["skills"]["f1"] for r in all_results])
        domain_acc = mean([r["domain_match"] for r in all_results])
        role_acc = mean([r["roles"]["f1"] for r in all_results])
        
        return ExtractionMetrics(
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            domain_accuracy=domain_acc,
            role_accuracy=role_acc,
            details={"per_test": all_results}
        )


# ============================================================================
# TEAM EVALUATOR
# ============================================================================

class TeamEvaluator:
    """
    Evaluates team formation quality.
    """
    
    def __init__(self, normalizer=None):
        self.normalizer = normalizer
    
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
                team_skills.add(s.lower() if isinstance(s, str) else str(s).lower())
        
        required_normalized = set(s.lower() for s in required_skills)
        
        # Count matches (including partial matches)
        covered = 0
        for req in required_normalized:
            if req in team_skills:
                covered += 1
            elif any(req in ts or ts in req for ts in team_skills):
                covered += 0.5  # Partial credit
        
        return min(1.0, covered / len(required_normalized))
    
    def calculate_role_diversity(self, team: List[Dict[str, Any]]) -> float:
        """Calculate diversity of roles in team."""
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
        
        dev_diversity = len(dev_types) / len(team)
        belbin_diversity = len(belbin_roles) / len(team)
        
        return (dev_diversity * 0.6 + belbin_diversity * 0.4)
    
    def calculate_experience_balance(self, team: List[Dict[str, Any]]) -> float:
        """Calculate how balanced experience levels are."""
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
            return 0.5
        
        # Score based on spread and mix
        spread = (max(experiences) - min(experiences)) / max(max(experiences), 1)
        has_junior = any(e <= 3 for e in experiences)
        has_senior = any(e >= 7 for e in experiences)
        
        mix_bonus = 0.3 if (has_junior and has_senior) else 0
        
        return min(1.0, spread * 0.7 + mix_bonus)
    
    def calculate_availability_fit(
        self,
        team: List[Dict[str, Any]],
        min_required_hours: int = None
    ) -> float:
        """Calculate how well team meets availability requirements."""
        if not min_required_hours:
            return 1.0
        
        fits = 0
        for member in team:
            constraints = member.get("constraints", {})
            max_hours = constraints.get("max_hours", 40)
            if max_hours >= min_required_hours:
                fits += 1
        
        return fits / len(team) if team else 0
    
    def calculate_avg_match_score(self, team: List[Dict[str, Any]]) -> float:
        """Calculate average semantic match score."""
        scores = [m.get("match_score", 0) for m in team]
        return mean(scores) if scores else 0
    
    def evaluate_team(
        self,
        team: List[Dict[str, Any]],
        required_skills: List[str] = None,
        min_availability_hours: int = None
    ) -> TeamQualityMetrics:
        """
        Compute comprehensive quality metrics for a team.
        """
        return TeamQualityMetrics(
            skill_coverage=self.calculate_skill_coverage(team, required_skills or []),
            role_diversity=self.calculate_role_diversity(team),
            experience_balance=self.calculate_experience_balance(team),
            avg_match_score=self.calculate_avg_match_score(team),
            availability_fit=self.calculate_availability_fit(team, min_availability_hours)
        )
    
    def benchmark_against_random(
        self,
        system_team: List[Dict[str, Any]],
        candidate_pool: List[Dict[str, Any]],
        required_skills: List[str] = None,
        num_trials: int = 50
    ) -> BenchmarkResult:
        """Compare system team against random baseline."""
        # Evaluate system team
        system_metrics = self.evaluate_team(system_team, required_skills)
        system_score = system_metrics.overall_score
        
        # Generate random teams
        team_size = len(system_team)
        random_scores = []
        
        for _ in range(num_trials):
            if len(candidate_pool) <= team_size:
                random_team = candidate_pool.copy()
            else:
                random_team = random.sample(candidate_pool, team_size)
            
            metrics = self.evaluate_team(random_team, required_skills)
            random_scores.append(metrics.overall_score)
        
        avg_random = mean(random_scores)
        std_random = stdev(random_scores) if len(random_scores) > 1 else 0
        
        improvement = ((system_score - avg_random) / avg_random * 100) if avg_random > 0 else 100
        
        # Simple significance estimate
        if std_random > 0:
            z_score = (system_score - avg_random) / std_random
            if z_score > 2:
                significance = "significant"
            elif z_score > 1:
                significance = "marginal"
            else:
                significance = "not significant"
        else:
            significance = "significant" if improvement > 10 else "not significant"
        
        return BenchmarkResult(
            system_score=system_score,
            random_avg_score=avg_random,
            random_std=std_random,
            improvement_percentage=improvement,
            num_trials=num_trials,
            p_value_estimate=significance
        )


# ============================================================================
# LATENCY TRACKER
# ============================================================================

class LatencyTracker:
    """Track latency across pipeline stages."""
    
    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
    
    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._starts[stage] = time.time()
    
    def stop(self, stage: str) -> float:
        """Stop timing and return duration in ms."""
        if stage not in self._starts:
            return 0.0
        
        duration = (time.time() - self._starts[stage]) * 1000
        self._timings[stage] = duration
        del self._starts[stage]
        return duration
    
    def track(self, stage: str):
        """Context manager for timing."""
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
    
    def get_report(self) -> LatencyReport:
        """Get latency report."""
        return LatencyReport(stages=self._timings.copy())
    
    def reset(self) -> None:
        """Reset all timings."""
        self._timings.clear()
        self._starts.clear()


# ============================================================================
# SCORE DISPLAY HELPERS
# ============================================================================

def format_score_with_icon(score: float, label: str) -> str:
    """Format score with icon for display."""
    if score >= 0.8:
        icon = "✅"
        color = "green"
    elif score >= 0.5:
        icon = "⚠️"
        color = "orange"
    else:
        icon = "❌"
        color = "red"
    
    return f"{icon} {label}: {score:.0%}"


def get_overall_status(metrics: TeamQualityMetrics) -> Tuple[str, str, str]:
    """Get overall status icon, label, and description."""
    score = metrics.overall_score
    
    if score >= 0.8:
        return "✅", "Excellent", "This team is well-optimized for the project"
    elif score >= 0.6:
        return "⚠️", "Good", "This team should work well with minor gaps"
    elif score >= 0.4:
        return "⚠️", "Fair", "Consider adjusting team composition"
    else:
        return "❌", "Needs Work", "Significant gaps in team coverage"
