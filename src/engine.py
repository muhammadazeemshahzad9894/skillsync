"""
SkillSync Engine

Main orchestration layer that coordinates all system components
for end-to-end team formation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI

from config.settings import settings
from .preprocessing import normalizer, csv_parser, PDFParser
from .extraction import LLMExtractor, ProjectRequirements
from .matching import EmbeddingManager, CandidateRetriever
from .team_formation import (
    team_formation_engine,
    TeamFormationResult,
    constraint_validator,
    ValidationResult
)
from .evaluation import TeamEvaluator, LatencyTracker, EvaluationMetrics
from .utils import load_data, save_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SkillSyncEngine:
    """
    Main engine for AI-powered team formation.
    
    Orchestrates the complete pipeline:
    1. Load and preprocess candidate data
    2. Generate embeddings for semantic search
    3. Extract structured requirements from natural language
    4. Retrieve matching candidates
    5. Form teams using multiple strategies
    6. Validate against constraints
    7. Generate explanations
    8. Evaluate quality
    
    Example:
        engine = SkillSyncEngine()
        results, requirements = engine.form_teams(
            "Build a fintech mobile app with React Native and Python backend",
            team_size=4
        )
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the SkillSync engine.
        
        Args:
            data_path: Optional custom path to employee data
        """
        logger.info("⚙️ Initializing SkillSync Engine...")
        
        self.data_path = data_path or settings.paths.employees_path
        self.latency_tracker = LatencyTracker()
        
        # Initialize components
        self._init_data()
        self._init_embeddings()
        self._init_llm()
        
        logger.info("✅ SkillSync Engine ready.")
    
    def _init_data(self) -> None:
        """Load and preprocess candidate data."""
        logger.info(f"📂 Loading data from {self.data_path}...")
        
        self.raw_data = load_data(self.data_path)
        
        if not self.raw_data:
            logger.warning("No employee data found. Use add_candidates() to populate.")
        else:
            logger.info(f"📊 Loaded {len(self.raw_data)} candidate profiles")
    
    def _init_embeddings(self) -> None:
        """Initialize embedding manager and retriever."""
        logger.info(f"🧠 Loading embedding model ({settings.embedding.model_name})...")
        
        self.embedding_manager = EmbeddingManager(
            model_name=settings.embedding.model_name,
            show_progress=settings.embedding.show_progress
        )
        
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        
        # Pre-compute embeddings if we have data
        if self.raw_data:
            _ = self.retriever.embeddings
    
    def _init_llm(self) -> None:
        """Initialize LLM client and extractor."""
        logger.info("🔌 Connecting to LLM API...")
        
        self.llm_client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url
        )
        
        self.extractor = LLMExtractor(
            client=self.llm_client,
            model=settings.llm.model,
            extra_headers=settings.llm_headers
        )
        
        self.pdf_parser = PDFParser(
            llm_client=self.llm_client,
            llm_model=settings.llm.model
        )
        
        self.evaluator = TeamEvaluator(retriever=self.retriever)
    
    def reload_data(self) -> None:
        """Reload data from disk and refresh embeddings."""
        logger.info("🔄 Reloading data...")
        self._init_data()
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        self.embedding_manager.clear_cache()
        _ = self.retriever.embeddings
        logger.info("✅ Data reloaded.")
    
    def add_candidates_from_csv(self, file_or_path: Any) -> int:
        """
        Add candidates from CSV file.
        
        Args:
            file_or_path: CSV file path or file-like object
            
        Returns:
            Number of candidates added
        """
        new_profiles = csv_parser.parse_csv(file_or_path)
        
        # Add to data and save
        self.raw_data.extend(new_profiles)
        save_data(self.data_path, self.raw_data)
        
        # Refresh retriever
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        
        logger.info(f"✅ Added {len(new_profiles)} candidates from CSV")
        return len(new_profiles)
    
    def add_candidates_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Add candidate from PDF resume.
        
        Args:
            pdf_bytes: PDF file content
            
        Returns:
            Parsed profile dictionary
        """
        profile = self.pdf_parser.parse_resume(
            pdf_bytes,
            extra_headers=settings.llm_headers
        )
        
        profile_dict = profile.to_dict()
        
        # Add to data and save
        self.raw_data.append(profile_dict)
        save_data(self.data_path, self.raw_data)
        
        # Refresh retriever
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        
        logger.info(f"✅ Added candidate from PDF: {profile.name}")
        return profile_dict
    
    def form_teams(
        self,
        project_description: str,
        team_size: int = None,
        include_evaluation: bool = True
    ) -> Tuple[Dict[str, TeamFormationResult], ProjectRequirements]:
        """
        Form teams based on project description.
        
        Args:
            project_description: Natural language project description
            team_size: Desired team size (default from settings)
            include_evaluation: Whether to compute evaluation metrics
            
        Returns:
            Tuple of (strategies dict, requirements)
        """
        team_size = team_size or settings.team.default_team_size
        self.latency_tracker.reset()
        
        # 1. Extract Requirements
        with self.latency_tracker.track("extraction"):
            logger.info("🔍 Extracting requirements...")
            requirements = self.extractor.extract_project_requirements(project_description)
        
        # 2. Retrieve Candidates
        with self.latency_tracker.track("retrieval"):
            logger.info("🎯 Searching candidates...")
            pool_size = team_size * settings.team.candidate_pool_multiplier
            
            # Use filtered search if we have constraints
            candidate_pool = self.retriever.search_with_filters(
                query=requirements.to_search_query(),
                top_k=pool_size,
                min_experience=requirements.min_experience,
                max_experience=requirements.max_experience,
                required_skills=requirements.technical_keywords[:5] if requirements.technical_keywords else None
            )
        
        if len(candidate_pool) < team_size:
            logger.error(f"Not enough candidates: {len(candidate_pool)} < {team_size}")
            return {"error": f"Not enough candidates found. Found {len(candidate_pool)}, need {team_size}."}, requirements
        
        # 3. Form Teams
        with self.latency_tracker.track("team_formation"):
            logger.info("🏗️ Forming teams...")
            strategies = team_formation_engine.form_teams(
                candidate_pool=candidate_pool,
                team_size=team_size,
                requirements=requirements.to_dict(),
                strategy_keys=["expert", "balanced", "diverse"]
            )
        
        # 4. Validate and Enrich Results
        for name, result in strategies.items():
            # Validate
            validation = constraint_validator.validate(
                team_members=result.members,
                required_skills=requirements.technical_keywords,
                target_roles=requirements.target_roles,
                required_team_size=team_size
            )
            result.metadata["validation"] = {
                "is_valid": validation.is_valid,
                "warnings": validation.warnings,
                "coverage_score": validation.coverage_score
            }
            
            # Evaluate if requested
            if include_evaluation:
                metrics = self.evaluator.evaluate_team(
                    result.members,
                    required_skills=requirements.technical_keywords
                )
                result.metadata["evaluation"] = metrics.to_dict()
        
        # 5. Generate Explanations
        with self.latency_tracker.track("explanation"):
            logger.info("🤖 Generating explanations...")
            for name, result in strategies.items():
                explanation = self.extractor.generate_team_explanation(
                    team_members=result.members,
                    project_summary=requirements.summary,
                    strategy_name=result.strategy_name,
                    strategy_rationale=result.rationale
                )
                result.llm_analysis = explanation
        
        # Log latency
        latency_report = self.latency_tracker.get_report()
        logger.info(f"⏱️ Total latency: {latency_report.total_ms:.0f}ms")
        
        return strategies, requirements
    
    def get_team_evaluation(
        self,
        team: List[Dict[str, Any]],
        required_skills: List[str] = None,
        compare_to_random: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed evaluation for a team.
        
        Args:
            team: Team member profiles
            required_skills: Skills to check coverage for
            compare_to_random: Whether to benchmark against random
            
        Returns:
            Evaluation results dictionary
        """
        result = {}
        
        # Basic metrics
        metrics = self.evaluator.evaluate_team(team, required_skills)
        result["metrics"] = metrics.to_dict()
        
        # Validation
        validation = constraint_validator.validate(
            team_members=team,
            required_skills=required_skills
        )
        result["validation"] = {
            "is_valid": validation.is_valid,
            "warnings": validation.warnings
        }
        
        # Benchmark
        if compare_to_random and len(self.raw_data) > len(team):
            benchmark = self.evaluator.benchmark_against_random(
                system_team=team,
                candidate_pool=self.raw_data,
                required_skills=required_skills
            )
            result["benchmark"] = benchmark.to_dict()
        
        return result
    
    @property
    def candidate_count(self) -> int:
        """Get number of candidates in pool."""
        return len(self.raw_data)
    
    def get_candidate_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Get candidate profile by ID."""
        for candidate in self.raw_data:
            if candidate.get("id") == candidate_id:
                return candidate
        return None
    
    def search_candidates(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for candidates matching a query.
        
        Args:
            query: Search query
            top_k: Maximum results
            
        Returns:
            List of matching candidates
        """
        return self.retriever.search(query, top_k=top_k)
