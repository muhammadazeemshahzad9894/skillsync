"""
Enhanced LLM Extraction Module

Implements a robust, single-pass extraction strategy using OpenAI's JSON mode.
Focuses on reliability, speed, and strict schema adherence.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for extraction."""
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1500


# ============================================================================
# ALLOWED ROLES
# ============================================================================

ALLOWED_ROLES = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "Developer, embedded applications or devices",
    "DevOps specialist",
    "Engineer, site reliability",
    "Cloud infrastructure engineer",
    "Data engineer",
    "Data scientist or machine learning specialist",
    "Developer, AI",
    "Developer, QA or test",
    "Developer, desktop or enterprise applications",
    "Product manager",
    "Project manager",
    "Security professional",
    "Data or business analyst",
    "Research & Development role",
    "System administrator",
]


# ============================================================================
# PROMPTS
# ============================================================================

EXTRACT_SYSTEM_PROMPT = """You are an expert technical recruiter and systems architect.
Your task is to extract structured project requirements from natural language descriptions.

OUTPUT FORMAT:
You must return a valid JSON object matching this schema exactly:
{
  "technical_keywords": ["list", "of", "skills"],
  "tools": ["list", "of", "tools"],
  "target_roles": ["list", "of", "standardized_roles"],
  "domain": "string",
  "seniority_level": "Junior" | "Mid" | "Senior" | "Mixed",
  "min_experience": number | null,
  "max_experience": number | null,
  "min_availability_hours": number | null,
  "summary": "string"
}

GUIDELINES:
1. **Technical Keywords**: Include languages (Python, Java), frameworks (React, FastAPI), databases (PostgreSQL), and cloud platforms (AWS, Azure). Normalize names (e.g., "React.js" -> "React").
2. **Tools**: Include strictly developer tools (Jira, GitHub, Docker, Kubernetes, Jenkins).
3. **Roles**: Map the needs to the ALLOWED ROLES list below. If a role is implied (e.g., "React" -> "Developer, front-end"), include it.
4. **Inference**: You MAY infer required skills that are standard for the mentioned stack (e.g., if "React" is requested, "JavaScript/TypeScript" is implied).
5. **Seniority**: Infer based on keywords like "Lead", "Architect" (Senior) or "maintain", "junior" (Junior). Default to "Mixed".

ALLOWED ROLES LIST (Choose from these ONLY):
""" + "\n".join(f"- {role}" for role in ALLOWED_ROLES)

TEAM_EXPLANATION_SYSTEM_PROMPT = """You are SkillSync, an expert team formation assistant.
Analyze the provided team composition against the project requirements.
Return valid JSON only."""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProjectRequirements:
    """Structured project requirements."""
    technical_keywords: List[str]
    tools: List[str]
    target_roles: List[str]
    domain: str
    seniority_level: str
    summary: str
    min_experience: Optional[float] = None
    max_experience: Optional[float] = None
    min_availability_hours: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_search_query(self) -> str:
        """Generate optimized search query."""
        parts = [self.summary]
        parts.extend(self.technical_keywords[:12])
        parts.extend(self.target_roles[:5])
        return " ".join(parts)


@dataclass
class TeamExplanation:
    """Structured team explanation."""
    analysis: str
    strengths: List[str]
    risks: List[str]
    skill_coverage: Dict[str, List[str]]
    dynamics_summary: str


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class ChainedLLMExtractor:
    """
    Robust LLM extractor using Single-Pass strategy with JSON Mode.
    (Kept class name 'ChainedLLMExtractor' for backward compatibility).
    """
    
    def __init__(
        self,
        client: OpenAI,
        config: ExtractionConfig = None,
        extra_headers: Dict[str, str] = None
    ):
        self.client = client
        self.config = config or ExtractionConfig()
        self.extra_headers = extra_headers or {}
        
        # Track extraction stages for debugging/evaluation
        self.last_extraction_stages: Dict[str, Any] = {}
    
    def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Make LLM API call enforcing JSON mode."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                extra_headers=self.extra_headers,
                response_format={"type": "json_object"}  # CRITICAL FIX
            )
            
            content = response.choices[0].message.content.strip()
            
            # Robust parsing in case the model returns markdown code blocks despite JSON mode
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def extract_requirements(
        self,
        description: str,
        skip_enhance: bool = False  # Argument kept for compatibility, ignored
    ) -> ProjectRequirements:
        """
        Extract structured requirements from description.
        """
        self.last_extraction_stages = {"description": description}
        
        try:
            logger.info("🔍 Extracting requirements (Single Pass)...")
            
            user_prompt = f"Project Description:\n{description}\n\nExtract technical requirements into JSON."
            
            extracted_data = self._call_llm_json(
                EXTRACT_SYSTEM_PROMPT,
                user_prompt,
                temperature=self.config.temperature
            )
            
            self.last_extraction_stages["raw_output"] = extracted_data
            
            # Sanitize and Validate Data Types
            keywords = extracted_data.get("technical_keywords", [])
            if isinstance(keywords, str): keywords = [keywords]
            
            tools = extracted_data.get("tools", [])
            if isinstance(tools, str): tools = [tools]
            
            roles = extracted_data.get("target_roles", [])
            if isinstance(roles, str): roles = [roles]
            
            # Create object with safe defaults
            reqs = ProjectRequirements(
                technical_keywords=[str(k) for k in keywords if k],
                tools=[str(t) for t in tools if t],
                target_roles=[str(r) for r in roles if r],
                domain=str(extracted_data.get("domain", "General")),
                seniority_level=str(extracted_data.get("seniority_level", "Mixed")),
                summary=str(extracted_data.get("summary", description[:200])),
                min_experience=self._safe_float(extracted_data.get("min_experience")),
                max_experience=self._safe_float(extracted_data.get("max_experience")),
                min_availability_hours=self._safe_int(extracted_data.get("min_availability_hours"))
            )
            
            logger.info(f"✅ Extraction success: {len(reqs.technical_keywords)} skills, {len(reqs.target_roles)} roles")
            return reqs
            
        except Exception as e:
            logger.error(f"❌ Extraction failed completely: {e}")
            # Fallback to prevent app crash, but log heavily
            return ProjectRequirements(
                technical_keywords=[],
                tools=[],
                target_roles=[],
                domain="General",
                seniority_level="Mixed",
                summary=description
            )

    def generate_team_explanation(
        self,
        team_members: List[Dict[str, Any]],
        project_summary: str,
        strategy_name: str,
        strategy_rationale: str,
        required_skills: List[str] = None
    ) -> TeamExplanation:
        """Generate structured team explanation."""
        
        # Prepare context
        team_summary = []
        for m in team_members:
            name = m.get("name", "Unknown")
            role = m.get("role", "Unknown")
            skills = ", ".join(m.get("technical", {}).get("skills", [])[:5])
            team_summary.append(f"- {name} ({role}): {skills}")
        
        user_prompt = f"""
        Project: {project_summary}
        Strategy: {strategy_name} ({strategy_rationale})
        Required Skills: {', '.join(required_skills or [])}
        
        Team Members:
        {chr(10).join(team_summary)}
        
        Provide a JSON analysis with keys: 
        analysis (string), strengths (list), risks (list), 
        skill_coverage (object with covered/missing lists), 
        dynamics_summary (string).
        """
        
        try:
            data = self._call_llm_json(
                TEAM_EXPLANATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.7
            )
            
            return TeamExplanation(
                analysis=data.get("analysis", "Analysis unavailable."),
                strengths=data.get("strengths", []),
                risks=data.get("risks", []),
                skill_coverage=data.get("skill_coverage", {"covered": [], "missing": []}),
                dynamics_summary=data.get("dynamics_summary", "")
            )
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return TeamExplanation(
                analysis="Analysis unavailable due to error.",
                strengths=[],
                risks=[],
                skill_coverage={"covered": [], "missing": []},
                dynamics_summary=""
            )

    def get_extraction_stages(self) -> Dict[str, Any]:
        return self.last_extraction_stages

    def _safe_float(self, val: Any) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    def _safe_int(self, val: Any) -> Optional[int]:
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None