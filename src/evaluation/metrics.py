"""
SkillSync Evaluation Module - Smart Fuzzy Matching

This module implements intelligent evaluation with:
- Fuzzy string matching for technical terms
- Synonym recognition (e.g., "AWS" = "Amazon Web Services")
- Normalization handling (e.g., "React.js" = "React" = "ReactJS")
- Semantic similarity for role matching
- Comprehensive metrics and reporting

Author: SkillSync Team
Version: 3.0 (Production-Ready)
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from statistics import mean
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# SYNONYM MAPPINGS for Technical Terms
# ============================================================================

SKILL_SYNONYMS = {
    # Cloud platforms
    "aws": {"aws", "amazon web services", "amazon web services (aws)"},
    "gcp": {"gcp", "google cloud", "google cloud platform", "google cloud platform (gcp)"},
    "azure": {"azure", "microsoft azure", "azure cloud"},
    
    # Frontend frameworks
    "react": {"react", "react.js", "reactjs"},
    "vue": {"vue", "vue.js", "vuejs"},
    "angular": {"angular", "angularjs", "angular.js"},
    "next.js": {"next.js", "nextjs", "next"},
    "svelte": {"svelte", "sveltejs", "svelte.js"},
    
    # Backend frameworks
    "express": {"express", "express.js", "expressjs"},
    "nestjs": {"nestjs", "nest.js", "nest"},
    "fastapi": {"fastapi", "fast api"},
    "django": {"django"},
    "flask": {"flask"},
    
    # Databases
    "postgresql": {"postgresql", "postgres", "psql"},
    "mongodb": {"mongodb", "mongo"},
    "mysql": {"mysql"},
    "redis": {"redis"},
    
    # Languages
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "python": {"python"},
    "java": {"java"},
    "c++": {"c++", "cpp"},
    "c#": {"c#", "csharp", "c sharp"},
    
    # DevOps
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "ci/cd": {"ci/cd", "cicd", "ci", "cd", "continuous integration", "continuous deployment"},
    "terraform": {"terraform"},
    
    # Mobile
    "react native": {"react native", "react-native", "reactnative"},
    "flutter": {"flutter"},
    "ios": {"ios"},
    "android": {"android"},
    
    # ML/AI
    "tensorflow": {"tensorflow", "tf"},
    "pytorch": {"pytorch", "torch"},
    "machine learning": {"machine learning", "ml"},
    "artificial intelligence": {"artificial intelligence", "ai"},
    
    # Other
    "node.js": {"node.js", "nodejs", "node"},
    "graphql": {"graphql"},
    "rest": {"rest", "restful", "rest api"},
    "websockets": {"websockets", "websocket", "ws"},
}

def normalize_skill(skill: str) -> str:
    """Normalize a skill name for matching."""
    normalized = skill.lower().strip()
    normalized = re.sub(r'[^\w\s\-\.\+#]', '', normalized)  # Keep alphanumeric, space, -, ., +, #
    normalized = re.sub(r'\s+', ' ', normalized)  # Normalize spaces
    return normalized

def are_skills_equivalent(skill1: str, skill2: str) -> bool:
    """
    Check if two skill names are semantically equivalent.
    
    Uses synonym mapping and fuzzy matching to handle variations like:
    - "AWS" and "Amazon Web Services"
    - "React" and "React.js"
    - "Node.js" and "NodeJS"
    """
    norm1 = normalize_skill(skill1)
    norm2 = normalize_skill(skill2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Check synonyms
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if norm1 in synonyms and norm2 in synonyms:
            return True
    
    # Fuzzy matching - check substring overlap for longer terms
    if len(norm1) >= 4 and len(norm2) >= 4:
        # Check if one is contained in the other (for compound names)
        if norm1 in norm2 or norm2 in norm1:
            return True
    
    return False


# ============================================================================
# ROLE SIMILARITY
# ============================================================================

# Role groupings for similarity matching
ROLE_GROUPS = {
    "frontend": ["Developer, front-end"],
    "backend": ["Developer, back-end"],
    "fullstack": ["Developer, full-stack"],
    "mobile": ["Developer, mobile"],
    "embedded": ["Developer, embedded applications or devices"],
    "desktop": ["Developer, desktop or enterprise applications"],
    "ai": ["Developer, AI", "Data scientist or machine learning specialist"],
    "data": ["Data engineer", "Data scientist or machine learning specialist", "Data or business analyst"],
    "devops": ["DevOps specialist", "Engineer, site reliability", "Cloud infrastructure engineer", "System administrator"],
    "qa": ["Developer, QA or test"],
    "security": ["Security professional"],
    "management": ["Product manager", "Project manager"],
    "research": ["Research & Development role"],
}

def are_roles_similar(role1: str, role2: str) -> bool:
    """
    Check if two roles are similar or in the same family.
    """
    # Exact match
    if role1.lower().strip() == role2.lower().strip():
        return True
    
    # Check if in same role group
    for group, roles in ROLE_GROUPS.items():
        roles_lower = [r.lower() for r in roles]
        if role1.lower() in roles_lower and role2.lower() in roles_lower:
            return True
    
    return False


# ============================================================================
# TEST SET for Evaluation
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
            "domain": "Manufacturing"
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
    },
    {
        "id": "test_011_marketplace",
        "description": "Redesign our online marketplace. Modern React frontend with TypeScript, Node.js backend with Express, MongoDB for products, PostgreSQL for orders. Payment integration with Stripe. Host on AWS using Docker containers. Need CI/CD automation. Team of 5 full-stack developers.",
        "expected": {
            "technical_keywords": ["React", "TypeScript", "Node.js", "Express", "MongoDB", "PostgreSQL", "Stripe", "AWS", "Docker", "CI/CD"],
            "tools": [],
            "target_roles": ["Developer, full-stack", "Developer, front-end", "Developer, back-end", "DevOps specialist", "Cloud infrastructure engineer"],
            "domain": "E-commerce"
        }
    }
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExtractionMetrics:
    """Comprehensive metrics for extraction evaluation."""
    precision: float
    recall: float
    f1_score: float
    domain_accuracy: float
    role_f1: float
    
    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_interpretation(self) -> str:
        """Get human-readable interpretation of results."""
        if self.f1_score >= 0.9:
            return "🎉 Excellent (≥90%)"
        elif self.f1_score >= 0.7:
            return "✅ Good (70-90%)"
        elif self.f1_score >= 0.5:
            return "⚠️ Needs Improvement (50-70%)"
        else:
            return "❌ Poor (<50%)"


# ============================================================================
# EXTRACTOR EVALUATOR
# ============================================================================

class ExtractionEvaluator:
    """
    Evaluates extraction quality using smart fuzzy matching.
    
    Features:
    - Synonym-aware matching (AWS = Amazon Web Services)
    - Fuzzy string matching for technical terms
    - Role similarity detection
    - Detailed per-test breakdowns
    - Comprehensive metrics
    """
    
    def __init__(self, test_set: List[Dict[str, Any]] = None):
        """
        Initialize evaluator.
        
        Args:
            test_set: List of test cases with 'description' and 'expected' fields.
                     If None, uses built-in EXTRACTION_TEST_SET.
        """
        self.test_set = test_set or EXTRACTION_TEST_SET
    
    def _calculate_fuzzy_set_metrics(
        self,
        predicted: List[str],
        expected: List[str],
        use_fuzzy: bool = True
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Calculate precision, recall, F1 with fuzzy matching.
        
        Args:
            predicted: List of predicted items
            expected: List of expected items
            use_fuzzy: Whether to use fuzzy/synonym matching
            
        Returns:
            (precision, recall, f1, details_dict)
        """
        if not predicted and not expected:
            return 1.0, 1.0, 1.0, {"matched": [], "missed": [], "hallucinated": []}
        
        if not predicted:
            return 0.0, 0.0, 0.0, {"matched": [], "missed": expected, "hallucinated": []}
        
        if not expected:
            return 0.0, 1.0, 0.0, {"matched": [], "missed": [], "hallucinated": predicted}
        
        # Track matching
        matched_predicted = set()
        matched_expected = set()
        match_pairs = []
        
        # Find matches
        for pred in predicted:
            for exp in expected:
                if exp in matched_expected:
                    continue
                
                is_match = False
                if use_fuzzy:
                    is_match = are_skills_equivalent(pred, exp)
                else:
                    is_match = normalize_skill(pred) == normalize_skill(exp)
                
                if is_match:
                    matched_predicted.add(pred)
                    matched_expected.add(exp)
                    match_pairs.append((pred, exp))
                    break
        
        # Calculate metrics
        true_positives = len(matched_predicted)
        false_positives = len(predicted) - true_positives
        false_negatives = len(expected) - len(matched_expected)
        
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(expected) if expected else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Detailed breakdown
        missed = [e for e in expected if e not in matched_expected]
        hallucinated = [p for p in predicted if p not in matched_predicted]
        
        details = {
            "matched": match_pairs,
            "missed": missed,
            "hallucinated": hallucinated,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
        
        return precision, recall, f1, details
    
    def _calculate_role_metrics(
        self,
        predicted: List[str],
        expected: List[str]
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Calculate role matching metrics with similarity detection.
        """
        if not predicted and not expected:
            return 1.0, 1.0, 1.0, {"matched": [], "missed": [], "hallucinated": []}
        
        if not predicted:
            return 0.0, 0.0, 0.0, {"matched": [], "missed": expected, "hallucinated": []}
        
        if not expected:
            return 0.0, 1.0, 0.0, {"matched": [], "missed": [], "hallucinated": predicted}
        
        # Track matching
        matched_predicted = set()
        matched_expected = set()
        match_pairs = []
        
        # Find matches
        for pred in predicted:
            for exp in expected:
                if exp in matched_expected:
                    continue
                
                if are_roles_similar(pred, exp):
                    matched_predicted.add(pred)
                    matched_expected.add(exp)
                    match_pairs.append((pred, exp))
                    break
        
        # Calculate metrics
        true_positives = len(matched_predicted)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(expected) if expected else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        details = {
            "matched": match_pairs,
            "missed": [e for e in expected if e not in matched_expected],
            "hallucinated": [p for p in predicted if p not in matched_predicted]
        }
        
        return precision, recall, f1, details
    
    def evaluate_single(
        self,
        predicted: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single extraction against expected values.
        
        Args:
            predicted: Predicted extraction results
            expected: Expected extraction results
            
        Returns:
            Dictionary with detailed metrics
        """
        results = {}
        
        # Technical keywords (with fuzzy matching)
        p, r, f1, details = self._calculate_fuzzy_set_metrics(
            predicted.get("technical_keywords", []),
            expected.get("technical_keywords", []),
            use_fuzzy=True
        )
        results["skills"] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "details": details
        }
        
        # Tools (with fuzzy matching)
        p, r, f1, details = self._calculate_fuzzy_set_metrics(
            predicted.get("tools", []),
            expected.get("tools", []),
            use_fuzzy=True
        )
        results["tools"] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "details": details
        }
        
        # Roles (with similarity matching)
        p, r, f1, details = self._calculate_role_metrics(
            predicted.get("target_roles", []),
            expected.get("target_roles", [])
        )
        results["roles"] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "details": details
        }
        
        # Domain (exact match with flexibility)
        pred_domain = (predicted.get("domain") or "").lower().strip()
        exp_domain = (expected.get("domain") or "").lower().strip()
        
        # Exact match or substring match
        domain_match = pred_domain == exp_domain or exp_domain in pred_domain or pred_domain in exp_domain
        results["domain_match"] = 1.0 if domain_match else 0.0
        results["domain_details"] = {
            "predicted": predicted.get("domain"),
            "expected": expected.get("domain")
        }
        
        return results
    
    def evaluate_extractor(self, extractor) -> ExtractionMetrics:
        """
        Run full evaluation on test set.
        
        Args:
            extractor: Extractor instance with extract_requirements() method
            
        Returns:
            ExtractionMetrics with aggregate scores and details
        """
        logger.info(f"Running evaluation on {len(self.test_set)} test cases...")
        
        all_results = []
        
        for test_case in self.test_set:
            try:
                # Run extraction
                result = extractor.extract_requirements(test_case["description"])
                
                # Convert to dict
                if hasattr(result, "to_dict"):
                    predicted = result.to_dict()
                else:
                    predicted = result
                
                # Evaluate
                metrics = self.evaluate_single(predicted, test_case["expected"])
                metrics["test_id"] = test_case["id"]
                metrics["description"] = test_case["description"][:100] + "..."
                all_results.append(metrics)
                
                logger.debug(f"Test {test_case['id']}: F1={metrics['skills']['f1']:.2f}")
                
            except Exception as e:
                logger.error(f"Test {test_case['id']} failed: {e}")
                # Add failed test with zero scores
                all_results.append({
                    "test_id": test_case["id"],
                    "skills": {"precision": 0, "recall": 0, "f1": 0},
                    "roles": {"precision": 0, "recall": 0, "f1": 0},
                    "domain_match": 0,
                    "error": str(e)
                })
        
        if not all_results:
            logger.error("No test results - all tests failed")
            return ExtractionMetrics(0, 0, 0, 0, 0, details={"per_test": []})
        
        # Aggregate metrics
        avg_precision = mean([r["skills"]["precision"] for r in all_results])
        avg_recall = mean([r["skills"]["recall"] for r in all_results])
        avg_f1 = mean([r["skills"]["f1"] for r in all_results])
        domain_acc = mean([r["domain_match"] for r in all_results])
        role_f1 = mean([r["roles"]["f1"] for r in all_results])
        
        metrics = ExtractionMetrics(
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            domain_accuracy=domain_acc,
            role_f1=role_f1,
            details={"per_test": all_results}
        )
        
        logger.info(
            f"✅ Evaluation complete: "
            f"Precision={avg_precision:.1%}, Recall={avg_recall:.1%}, "
            f"F1={avg_f1:.1%}, Domain={domain_acc:.1%}"
        )
        
        return metrics
    
    def generate_recommendations(self, metrics: ExtractionMetrics) -> List[str]:
        """
        Generate actionable recommendations based on evaluation results.
        """
        recommendations = []
        
        if metrics.precision < 0.7:
            recommendations.append(
                "🎯 **Reduce Hallucinations**: The system is extracting items not in the text. "
                "Consider strengthening validation rules or adjusting temperature to 0.0."
            )
        
        if metrics.recall < 0.7:
            recommendations.append(
                "📊 **Improve Coverage**: The system is missing items. "
                "Enhance extraction prompts to be more comprehensive and explicit."
            )
        
        if metrics.domain_accuracy < 0.7:
            recommendations.append(
                "🏷️ **Better Domain Classification**: Improve domain detection by providing "
                "more examples or constraints in the prompt."
            )
        
        if metrics.role_f1 < 0.7:
            recommendations.append(
                "👥 **Role Matching**: Refine role inference rules or expand role detection logic."
            )
        
        if metrics.f1_score >= 0.9:
            recommendations.append(
                "🎉 **Excellent Performance**: The extraction system is performing very well! "
                "Consider fine-tuning for edge cases or specific domains."
            )
        
        return recommendations
