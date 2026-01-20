"""
Enhanced LLM Extraction Module

Implements chain-of-prompts approach:
1. Extract - Initial extraction from project description
2. Validate - Check for hallucinations, enforce constraints
3. Enhance - Add inferred skills, normalize terminology

Merged with teammate's proj_extractor.py best practices.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for extraction chain."""
    model: str = "openai/gpt-4o-mini"
    temperature_extract: float = 0.0
    temperature_validate: float = 0.0
    temperature_enhance: float = 0.1
    max_tokens: int = 1500


# ============================================================================
# ALLOWED ROLES (from teammate's proj_extractor.py)
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
# PROMPTS - STAGE 1: EXTRACT
# ============================================================================

EXTRACT_SYSTEM_PROMPT = """You are a technical requirements extraction assistant for a team formation system.

YOUR JOB:
1. Extract ALL technologies, frameworks, languages, and tools explicitly mentioned
2. Identify the professional roles needed based on the technologies
3. Classify the industry domain
4. Return structured JSON

BE THOROUGH but ACCURATE:
- Include every technology mentioned by name
- Infer roles based on technology patterns (React → front-end developer)
- Be specific about domains (E-commerce, Fintech, Healthcare, not just "Technology")
- Return ONLY valid JSON (no markdown, no extra text)

CATEGORY RULES:
- "technical_keywords": programming languages, frameworks, libraries, databases, cloud platforms
- "tools": developer tools ONLY (Git, Docker, Figma, Jira, etc.)
- "target_roles": job roles inferred from the tech stack
- Cloud platforms (AWS, GCP, Azure) are ALWAYS "technical_keywords", NEVER "tools"
"""

EXTRACT_USER_PROMPT = """Extract ALL technical requirements from this project description.

STEP-BY-STEP EXTRACTION:

1. IDENTIFY TECHNOLOGIES (for "technical_keywords"):
   - Programming languages: Python, JavaScript, Java, C++, etc.
   - Frameworks: React, Django, Spring, Flask, FastAPI, etc.
   - Databases: PostgreSQL, MongoDB, MySQL, SQL, NoSQL, etc.
   - Cloud: AWS, Azure, GCP, cloud services
   - ML/AI: TensorFlow, PyTorch, Machine Learning, AI
   - Other: Any tech mentioned (GraphQL, REST, Docker, etc.)
   → Extract EVERY technology mentioned by name

2. IDENTIFY TOOLS (for "tools"):
   - Development: Git, GitHub, VS Code, IntelliJ
   - Design: Figma, Sketch, Adobe XD
   - Project: Jira, Trello, Slack
   - DevOps: Docker, Kubernetes, Jenkins, CI/CD tools
   
3. INFER ROLES (for "target_roles") - Use EXACT names from this list:
   - "Developer, front-end" ← IF React, Vue, Angular, frontend mentioned
   - "Developer, back-end" ← IF Django, Flask, FastAPI, Node.js, Spring, API, backend mentioned
   - "Developer, full-stack" ← IF both frontend AND backend tech mentioned
   - "Developer, mobile" ← IF React Native, Flutter, iOS, Android, mobile mentioned
   - "Data scientist or machine learning specialist" ← IF ML, TensorFlow, PyTorch, AI mentioned
   - "DevOps specialist" ← IF Docker, Kubernetes, CI/CD mentioned
   - "Cloud infrastructure engineer" ← IF AWS, Azure, GCP deployment mentioned
   - "Data engineer" ← IF data pipelines, ETL, SQL, analytics mentioned
   → Include ALL relevant roles based on tech stack

4. CLASSIFY DOMAIN (for "domain"):
   - "Fintech" ← payment, banking, trading, financial
   - "Healthcare" ← medical, health, patient, diagnostic
   - "E-commerce" ← shopping, store, marketplace, retail, e-commerce
   - "Education" ← learning, teaching, courses, educational
   - "Manufacturing" ← factory, production, industrial
   - "Gaming" ← game, gaming, entertainment
   - Use best match, avoid generic "Technology" or "General"

5. SET SENIORITY (for "seniority_level"):
   - "Junior" if "junior" or "entry-level" mentioned
   - "Senior" if "senior" or "experienced" or "5+ years" mentioned
   - "Mixed" if not specified (default)

RETURN THIS EXACT JSON SCHEMA:
{{
  "technical_keywords": ["list every technology mentioned"],
  "tools": ["list developer/design tools"],
  "target_roles": ["roles from the approved list above"],
  "domain": "specific industry domain",
  "seniority_level": "Junior|Mid|Senior|Mixed",
  "min_experience": null,
  "max_experience": null,
  "min_availability_hours": null,
  "summary": "1-2 sentence professional summary"
}}

PROJECT DESCRIPTION:
{description}

Return ONLY the JSON object (no markdown, no backticks, no extra text):"""


# ============================================================================
# PROMPTS - STAGE 2: VALIDATE
# ============================================================================

VALIDATE_SYSTEM_PROMPT = """You are a validation assistant that checks and improves extracted requirements.

Your job:
1. Verify all items are reasonable given the project description
2. Ensure roles match the ALLOWED ROLES list (fix any mismatches)
3. Keep all technologies that are mentioned or reasonably implied
4. Fix obvious errors but don't remove items unnecessarily

Be helpful, not overly strict - we want comprehensive extraction."""

VALIDATE_USER_PROMPT = """Check and improve this extraction against the original project description.

ORIGINAL DESCRIPTION:
"{description}"

EXTRACTED DATA:
{extracted_json}

ALLOWED ROLES (use EXACT names from this list):
{allowed_roles}

VALIDATION TASKS:
1. Verify all technical_keywords are reasonable (keep items that are mentioned or clearly implied)
2. Check tools are actual developer/design tools (not cloud platforms)
3. Map any roles NOT in ALLOWED ROLES to the closest match
4. If "Developer, full-stack" is present, you can keep front-end/back-end if they're also needed
5. Verify domain is specific (not "Technology" or "General" unless truly applicable)
6. Check seniority level is reasonable

IMPORTANT: Don't remove items unless they're clearly wrong. We want comprehensive extraction.

Return the CORRECTED JSON with same schema. Add a "validation_notes" field listing any changes made.
Return JSON only:"""


# ============================================================================
# PROMPTS - STAGE 3: ENHANCE
# ============================================================================

ENHANCE_SYSTEM_PROMPT = """You are an enhancement assistant that improves extracted requirements.

Your job:
1. Normalize technology names (e.g., "AWS" → "Amazon Web Services (AWS)")
2. Add commonly paired technologies if highly likely
3. Expand acronyms for clarity
4. Improve the summary for professional use

Be conservative - only add what's very likely implied."""

ENHANCE_USER_PROMPT = """Enhance this validated extraction for better matching.

VALIDATED DATA:
{validated_json}

ENHANCEMENT TASKS:
1. Normalize cloud platforms:
   - "AWS" → "Amazon Web Services (AWS)"
   - "GCP" → "Google Cloud"
   - "OCI" → "Oracle Cloud Infrastructure (OCI)"

2. Normalize frameworks:
   - "PyTorch" or "Torch" → "Torch/PyTorch"
   - ".NET Core" → "ASP.NET CORE"
   - "Bash" → "Bash/Shell (all shells)"

3. Add implicit skills ONLY if very obvious:
   - React mentioned → likely JavaScript/TypeScript
   - Django mentioned → likely Python
   - Spring Boot → likely Java

4. Improve summary to be more specific and actionable

Return enhanced JSON with same schema plus "enhancements_made" field.
Return JSON only:"""


# ============================================================================
# PROMPTS - TEAM EXPLANATION
# ============================================================================

TEAM_EXPLANATION_SYSTEM_PROMPT = """You are SkillSync, a team reporting assistant.

HARD RULES:
- Use ONLY the provided team data - do NOT invent facts
- Belbin roles are teamwork styles, NOT job titles
- Be specific about skills and experience
- Keep explanations concise but insightful
"""

TEAM_EXPLANATION_USER_PROMPT = """Generate analysis for this team selection.

PROJECT: {project_summary}
STRATEGY: {strategy_name} - {strategy_rationale}
REQUIRED SKILLS: {required_skills}

TEAM MEMBERS:
{team_details}

Generate a JSON response:
{{
  "analysis": "2-3 sentence analysis of why this team works",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "risks": ["potential risk 1", "potential risk 2"],
  "skill_coverage": {{
    "covered": ["skills this team has"],
    "missing": ["skills that may be gaps"]
  }},
  "dynamics_summary": "1 sentence on team dynamics based on Belbin roles"
}}

Return JSON only:"""


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
    validation_notes: List[str] = None
    enhancements_made: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_search_query(self) -> str:
        """Generate optimized search query."""
        parts = [self.summary]
        parts.extend(self.technical_keywords[:10])
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
    
    def to_markdown(self) -> str:
        """Convert to display-ready markdown."""
        md = f"{self.analysis}\n\n"
        
        if self.strengths:
            md += "**Strengths:**\n"
            for s in self.strengths:
                md += f"- {s}\n"
        
        if self.risks:
            md += "\n**Watch-outs:**\n"
            for r in self.risks:
                md += f"- {r}\n"
        
        if self.dynamics_summary:
            md += f"\n*{self.dynamics_summary}*"
        
        return md


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class ChainedLLMExtractor:
    """
    LLM extractor using chain-of-prompts approach.
    
    Pipeline: Extract → Validate → Enhance
    
    Example:
        extractor = ChainedLLMExtractor(client, config)
        requirements = extractor.extract_requirements(description)
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
        
        # Track extraction stages for evaluation
        self.last_extraction_stages: Dict[str, Any] = {}
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0
    ) -> str:
        """Make LLM API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                extra_headers=self.extra_headers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        raw = raw.strip()
        # Remove markdown code blocks
        raw = raw.replace("```json", "").replace("```", "")
        # Find JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in response")
        return json.loads(raw[start:end+1])
    
    def _stage_extract(self, description: str) -> Dict[str, Any]:
        """Stage 1: Initial extraction."""
        logger.info("🔍 Stage 1: Extracting requirements...")
        
        prompt = EXTRACT_USER_PROMPT.format(description=description)
        raw = self._call_llm(
            EXTRACT_SYSTEM_PROMPT,
            prompt,
            temperature=self.config.temperature_extract
        )
        return self._parse_json(raw)
    
    def _stage_validate(self, description: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Validate extraction."""
        logger.info("✅ Stage 2: Validating extraction...")
        
        prompt = VALIDATE_USER_PROMPT.format(
            description=description,
            extracted_json=json.dumps(extracted, indent=2),
            allowed_roles="\n".join(f"- {r}" for r in ALLOWED_ROLES)
        )
        raw = self._call_llm(
            VALIDATE_SYSTEM_PROMPT,
            prompt,
            temperature=self.config.temperature_validate
        )
        return self._parse_json(raw)
    
    def _stage_enhance(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Enhance extraction."""
        logger.info("✨ Stage 3: Enhancing extraction...")
        
        prompt = ENHANCE_USER_PROMPT.format(
            validated_json=json.dumps(validated, indent=2)
        )
        raw = self._call_llm(
            ENHANCE_SYSTEM_PROMPT,
            prompt,
            temperature=self.config.temperature_enhance
        )
        return self._parse_json(raw)
    
    def extract_requirements(
        self,
        description: str,
        skip_enhance: bool = False
    ) -> ProjectRequirements:
        """
        Extract structured requirements using chain approach.
        
        Args:
            description: Natural language project description
            skip_enhance: Skip enhancement stage (faster)
            
        Returns:
            ProjectRequirements with extracted data
        """
        # Reset tracking
        self.last_extraction_stages = {"description": description}
        
        try:
            # Stage 1: Extract
            extracted = self._stage_extract(description)
            self.last_extraction_stages["stage1_extract"] = extracted
            
            # Stage 2: Validate
            validated = self._stage_validate(description, extracted)
            self.last_extraction_stages["stage2_validate"] = validated
            
            # Stage 3: Enhance (optional)
            if not skip_enhance:
                enhanced = self._stage_enhance(validated)
                self.last_extraction_stages["stage3_enhance"] = enhanced
                final = enhanced
            else:
                final = validated
            
            # Build requirements object
            return ProjectRequirements(
                technical_keywords=final.get("technical_keywords", []),
                tools=final.get("tools", []),
                target_roles=final.get("target_roles", []),
                domain=final.get("domain", "General"),
                seniority_level=final.get("seniority_level", "Mixed"),
                summary=final.get("summary", description),
                min_experience=final.get("min_experience"),
                max_experience=final.get("max_experience"),
                min_availability_hours=final.get("min_availability_hours"),
                validation_notes=validated.get("validation_notes", []),
                enhancements_made=final.get("enhancements_made", []) if not skip_enhance else []
            )
            
        except Exception as e:
            logger.warning(f"Extraction chain failed, using fallback: {e}")
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
        """
        Generate structured team explanation.
        """
        # Format team details
        team_details = []
        for member in team_members:
            skills = member.get("technical", {}).get("skills", [])[:5]
            exp = member.get("metadata", {}).get("work_experience_years", "?")
            role = member.get("role", "Unknown")
            name = member.get("name", "Unknown")
            belbin = member.get("personality", {}).get("Belbin_team_role", "Unknown")
            
            team_details.append(
                f"- {name}: {role}, {exp}y exp, Belbin: {belbin}, Skills: {', '.join(skills)}"
            )
        
        prompt = TEAM_EXPLANATION_USER_PROMPT.format(
            project_summary=project_summary,
            strategy_name=strategy_name,
            strategy_rationale=strategy_rationale,
            required_skills=", ".join(required_skills or []),
            team_details="\n".join(team_details)
        )
        
        try:
            raw = self._call_llm(
                TEAM_EXPLANATION_SYSTEM_PROMPT,
                prompt,
                temperature=0.7
            )
            data = self._parse_json(raw)
            
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
                analysis="Team analysis unavailable.",
                strengths=[],
                risks=[],
                skill_coverage={"covered": [], "missing": []},
                dynamics_summary=""
            )
    
    def get_extraction_stages(self) -> Dict[str, Any]:
        """Get details of last extraction for debugging/evaluation."""
        return self.last_extraction_stages


# ============================================================================
# SIMPLE EXTRACTOR (single prompt, for comparison)
# ============================================================================

class SimpleLLMExtractor:
    """Simple single-prompt extractor for comparison."""
    
    def __init__(self, client: OpenAI, model: str, extra_headers: Dict = None):
        self.client = client
        self.model = model
        self.extra_headers = extra_headers or {}
    
    def extract_requirements(self, description: str) -> ProjectRequirements:
        """Single-prompt extraction."""
        prompt = f"""Extract requirements from this project description.

Return JSON:
{{
  "technical_keywords": [],
  "tools": [],
  "target_roles": [],
  "domain": "",
  "seniority_level": "Mixed",
  "summary": ""
}}

Project: {description}

JSON only:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                extra_headers=self.extra_headers
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "")
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end+1])
            
            return ProjectRequirements(
                technical_keywords=data.get("technical_keywords", []),
                tools=data.get("tools", []),
                target_roles=data.get("target_roles", []),
                domain=data.get("domain", "General"),
                seniority_level=data.get("seniority_level", "Mixed"),
                summary=data.get("summary", description)
            )
        except Exception as e:
            logger.error(f"Simple extraction failed: {e}")
            return ProjectRequirements(
                technical_keywords=[],
                tools=[],
                target_roles=[],
                domain="General",
                seniority_level="Mixed",
                summary=description
            )