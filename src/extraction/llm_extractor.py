"""
LLM Extraction Module

Handles LLM-based extraction of structured data from natural language inputs,
including project requirements and profile enhancement.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProjectRequirements:
    """Structured project requirements extracted from natural language."""
    technical_keywords: List[str]
    target_roles: List[str]
    seniority_level: str
    domain: str
    summary: str
    min_experience: Optional[float] = None
    max_experience: Optional[float] = None
    required_availability: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_search_query(self) -> str:
        """Generate optimized search query for candidate retrieval."""
        parts = [self.summary]
        parts.extend(self.technical_keywords)
        parts.extend(self.target_roles)
        return " ".join(parts)


class LLMExtractor:
    """
    Extracts structured information from natural language using LLM.
    
    Handles:
    - Project requirements extraction
    - Profile enhancement
    - Team explanation generation
    
    Example:
        extractor = LLMExtractor(client, model)
        requirements = extractor.extract_project_requirements(description)
    """
    
    PROJECT_EXTRACTION_PROMPT = '''
Extract structured requirements from this project description.

PROJECT DESCRIPTION:
"""
{description}
"""

Return a JSON object with these fields:
{{
    "technical_keywords": ["list", "of", "skills", "tools", "technologies", "mentioned"],
    "target_roles": ["list", "of", "job", "titles", "needed"],
    "seniority_level": "Junior | Mid | Senior | Mixed",
    "domain": "Industry domain (e.g., Fintech, Healthcare, E-commerce, Education, General)",
    "summary": "A clear, professional 1-2 sentence summary of what the project needs",
    "min_experience": null or minimum years of experience required (number),
    "max_experience": null or maximum years of experience (number),
    "required_availability": null or minimum weekly hours needed (number)
}}

Rules:
- Extract ALL technical skills, frameworks, languages, and tools mentioned
- Infer roles even if not explicitly stated (e.g., "mobile app" implies "Mobile Developer")
- Be comprehensive with technical_keywords - include both explicit and implicit requirements
- For seniority, default to "Mixed" if not specified
- Return ONLY valid JSON, no markdown
'''

    TEAM_EXPLANATION_PROMPT = '''
PROJECT CONTEXT: "{project_summary}"
STRATEGY: "{strategy_name}" - {strategy_rationale}

SELECTED TEAM:
{team_details}

Write a professional 2-3 sentence analysis explaining:
1. Why this team composition is effective for the project
2. The key strength this team brings

Be specific about how team members complement each other. Focus on concrete skills and experience.
'''

    def __init__(
        self,
        client: OpenAI,
        model: str,
        extra_headers: Dict[str, str] = None
    ):
        """
        Initialize LLM extractor.
        
        Args:
            client: OpenAI-compatible client
            model: Model identifier
            extra_headers: Optional headers for API calls (e.g., for OpenRouter)
        """
        self.client = client
        self.model = model
        self.extra_headers = extra_headers or {}
    
    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """Make LLM API call with error handling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=self.extra_headers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        # Clean markdown formatting
        cleaned = response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise ValueError("Failed to parse LLM response as JSON")
    
    def extract_project_requirements(
        self,
        description: str
    ) -> ProjectRequirements:
        """
        Extract structured requirements from project description.
        
        Args:
            description: Natural language project description
            
        Returns:
            ProjectRequirements object with extracted data
        """
        logger.info("Extracting project requirements...")
        
        prompt = self.PROJECT_EXTRACTION_PROMPT.format(description=description)
        
        try:
            response = self._call_llm(prompt, temperature=0.1)
            data = self._parse_json_response(response)
            
            return ProjectRequirements(
                technical_keywords=data.get("technical_keywords", []),
                target_roles=data.get("target_roles", []),
                seniority_level=data.get("seniority_level", "Mixed"),
                domain=data.get("domain", "General"),
                summary=data.get("summary", description),
                min_experience=data.get("min_experience"),
                max_experience=data.get("max_experience"),
                required_availability=data.get("required_availability")
            )
            
        except Exception as e:
            logger.warning(f"Extraction failed, using fallback: {e}")
            return ProjectRequirements(
                technical_keywords=[],
                target_roles=[],
                seniority_level="Mixed",
                domain="General",
                summary=description
            )
    
    def generate_team_explanation(
        self,
        team_members: List[Dict[str, Any]],
        project_summary: str,
        strategy_name: str,
        strategy_rationale: str
    ) -> str:
        """
        Generate human-readable explanation for team selection.
        
        Args:
            team_members: List of team member profiles
            project_summary: Brief project description
            strategy_name: Name of the selection strategy
            strategy_rationale: Brief rationale for the strategy
            
        Returns:
            Generated explanation text
        """
        # Format team details
        team_details = []
        for member in team_members:
            skills = member.get("technical", {}).get("skills", [])[:4]
            exp = member.get("metadata", {}).get("work_experience_years", "?")
            role = member.get("role", "Unknown")
            name = member.get("name", "Unknown")
            
            team_details.append(
                f"- {name}: {role}, {exp} years exp, Skills: {', '.join(skills)}"
            )
        
        prompt = self.TEAM_EXPLANATION_PROMPT.format(
            project_summary=project_summary,
            strategy_name=strategy_name,
            strategy_rationale=strategy_rationale,
            team_details="\n".join(team_details)
        )
        
        try:
            return self._call_llm(prompt, temperature=0.7, max_tokens=200)
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return "Team analysis unavailable."
    
    def enhance_profile(
        self,
        profile: Dict[str, Any],
        additional_text: str = None
    ) -> Dict[str, Any]:
        """
        Enhance existing profile with LLM-inferred attributes.
        
        Args:
            profile: Existing profile dictionary
            additional_text: Optional additional context
            
        Returns:
            Enhanced profile dictionary
        """
        # This can be extended to infer missing fields
        # For now, return profile as-is
        return profile
