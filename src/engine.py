"""
SkillSync Engine v2

Enhanced orchestration layer with:
- Chained LLM extraction
- Availability filtering
- Comprehensive evaluation
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI

from config.settings import settings
from .preprocessing import normalizer
from .preprocessing.csv_parser import parse_csv_auto, StackOverflowCSVParser
from .extraction import ChainedLLMExtractor, ProjectRequirements, ExtractionConfig
from .matching import EmbeddingManager, CandidateRetriever
from .team_formation import team_formation_engine, TeamFormationResult, constraint_validator
from .evaluation import TeamEvaluator, LatencyTracker, TeamQualityMetrics
from .utils import load_data, save_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SkillSyncEngine:
    """
    Main engine for AI-powered team formation.
    
    Enhanced features:
    - Chain-of-prompts extraction
    - Availability filtering
    - StackOverflow CSV support
    - Comprehensive metrics
    """
    
    def __init__(self, data_path: str = None):
        """Initialize the SkillSync engine."""
        logger.info("⚙️ Initializing SkillSync Engine v2...")
        
        self.data_path = data_path or settings.paths.employees_path
        self.latency_tracker = LatencyTracker()
        
        self._init_data()
        self._init_embeddings()
        self._init_llm()
        
        logger.info("✅ SkillSync Engine v2 ready.")
    
    def _init_data(self) -> None:
        """Load candidate data."""
        logger.info(f"📂 Loading data from {self.data_path}...")
        self.raw_data = load_data(self.data_path)
        
        if not self.raw_data:
            logger.warning("No employee data found.")
        else:
            logger.info(f"📊 Loaded {len(self.raw_data)} candidate profiles")
    
    def _init_embeddings(self) -> None:
        """Initialize embedding manager."""
        logger.info(f"🧠 Loading embedding model...")
        
        self.embedding_manager = EmbeddingManager(
            model_name=settings.embedding.model_name,
            show_progress=settings.embedding.show_progress
        )
        
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        
        if self.raw_data:
            _ = self.retriever.embeddings
    
    def _init_llm(self) -> None:
        """Initialize LLM client and extractors."""
        logger.info("🔌 Connecting to LLM API...")
        
        self.llm_client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url
        )
        
        # Use chained extractor for better results
        self.extractor = ChainedLLMExtractor(
            client=self.llm_client,
            config=ExtractionConfig(
                model=settings.llm.model,
                temperature_extract=0.0,
                temperature_validate=0.0,
                temperature_enhance=0.1
            ),
            extra_headers=settings.llm_headers
        )
        
        self.evaluator = TeamEvaluator()
    
    def reload_data(self) -> None:
        """Reload data from disk."""
        logger.info("🔄 Reloading data...")
        self._init_data()
        self.retriever = CandidateRetriever(
            profiles=self.raw_data,
            embedding_manager=self.embedding_manager
        )
        self.embedding_manager.clear_cache()
        _ = self.retriever.embeddings
        logger.info("✅ Data reloaded.")
    
    def add_candidates_from_csv(
        self,
        file_or_path: Any,
        min_availability_hours: int = None
    ) -> int:
        """
        Add candidates from CSV file.
        
        Supports both simple and StackOverflow formats.
        """
        new_profiles = parse_csv_auto(
            file_or_path,
            llm_client=self.llm_client,
            llm_model=settings.llm.model,
            min_availability_hours=min_availability_hours
        )
        
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
        """Add candidate from PDF resume."""
        # Import PDF parser here to avoid import issues if PyMuPDF not installed
        from .preprocessing.pdf_parser import PDFParser
        
        pdf_parser = PDFParser(
            llm_client=self.llm_client,
            llm_model=settings.llm.model
        )
        
        profile = pdf_parser.parse_resume(
            pdf_bytes,
            extra_headers=settings.llm_headers
        )
        
        profile_dict = profile.to_dict()
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
        include_evaluation: bool = True,
        min_availability_hours: int = None
    ) -> Tuple[Dict[str, TeamFormationResult], ProjectRequirements]:
        """
        Form teams based on project description.
        
        Returns:
            Tuple of (strategies dict, requirements)
        """
        team_size = team_size or settings.team.default_team_size
        self.latency_tracker.reset()
        
        # 1. Extract Requirements (chained)
        with self.latency_tracker.track("extraction"):
            logger.info("🔍 Extracting requirements (chain approach)...")
            requirements = self.extractor.extract_requirements(project_description)
        
        # Use availability from requirements if specified
        if requirements.min_availability_hours and not min_availability_hours:
            min_availability_hours = requirements.min_availability_hours
        
        # 2. Retrieve Candidates
        with self.latency_tracker.track("retrieval"):
            logger.info("🎯 Searching candidates...")
            pool_size = team_size * settings.team.candidate_pool_multiplier
            
            # Filter by availability if specified
            candidate_pool = self.retriever.search_with_filters(
                query=requirements.to_search_query(),
                top_k=pool_size,
                min_experience=requirements.min_experience,
                max_experience=requirements.max_experience,
                required_skills=requirements.technical_keywords[:5] if requirements.technical_keywords else None
            )
            
            # Additional availability filtering
            if min_availability_hours:
                candidate_pool = [
                    c for c in candidate_pool
                    if c.get("constraints", {}).get("max_hours", 40) >= min_availability_hours
                ]
        
        if len(candidate_pool) < team_size:
            logger.error(f"Not enough candidates: {len(candidate_pool)} < {team_size}")
            return {"error": f"Not enough candidates. Found {len(candidate_pool)}, need {team_size}."}, requirements
        
        # 3. Form Teams
        with self.latency_tracker.track("team_formation"):
            logger.info("🏗️ Forming teams...")
            strategies = team_formation_engine.form_teams(
                candidate_pool=candidate_pool,
                team_size=team_size,
                requirements=requirements.to_dict(),
                strategy_keys=["expert", "balanced", "diverse"]
            )
        
        # 4. Validate and Evaluate
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
            
            # Evaluate
            if include_evaluation:
                metrics = self.evaluator.evaluate_team(
                    result.members,
                    required_skills=requirements.technical_keywords,
                    min_availability_hours=min_availability_hours
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
                    strategy_rationale=result.rationale,
                    required_skills=requirements.technical_keywords
                )
                result.llm_analysis = explanation.to_markdown()
        
        latency_report = self.latency_tracker.get_report()
        logger.info(f"⏱️ Total latency: {latency_report.total_ms:.0f}ms")
        
        return strategies, requirements
    
    def get_team_evaluation(
        self,
        team: List[Dict[str, Any]],
        required_skills: List[str] = None,
        compare_to_random: bool = True
    ) -> Dict[str, Any]:
        """Get detailed evaluation for a team."""
        result = {}
        
        metrics = self.evaluator.evaluate_team(team, required_skills)
        result["metrics"] = metrics.to_dict()
        
        validation = constraint_validator.validate(
            team_members=team,
            required_skills=required_skills
        )
        result["validation"] = {
            "is_valid": validation.is_valid,
            "warnings": validation.warnings
        }
        
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
        return len(self.raw_data)
    
    def search_candidates(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.retriever.search(query, top_k=top_k)
