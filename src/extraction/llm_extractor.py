"""
SkillSync Extraction Module - Optimized for High-Quality Results

This module implements a streamlined, production-grade extraction system that:
- Uses clear, unambiguous prompts for reliable LLM extraction
- Validates outputs with smart heuristics
- Provides detailed logging and error handling
- Achieves high precision and recall on benchmark tests

Author: SkillSync Team
Version: 3.0 (Production-Ready)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict

from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Standardized role names (from StackOverflow Developer Survey)
ALLOWED_ROLES = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "Developer, embedded applications or devices",
    "Developer, desktop or enterprise applications",
    "Developer, AI",
    "Developer, QA or test",
    "DevOps specialist",
    "Engineer, site reliability",
    "Cloud infrastructure engineer",
    "Data engineer",
    "Data scientist or machine learning specialist",
    "Data or business analyst",
    "Security professional",
    "Product manager",
    "Project manager",
    "System administrator",
    "Research & Development role",
]

# Domain taxonomy
DOMAINS = [
    "Fintech",
    "Healthcare",
    "E-commerce",
    "Education",
    "Manufacturing",
    "Gaming",
    "Cybersecurity",
    "Agriculture",
    "Technology",
    "General"
]

# Seniority levels
SENIORITY_LEVELS = ["Junior", "Mid", "Senior", "Mixed"]


# ============================================================================
# EXTRACTION PROMPT - Optimized for Clarity and Accuracy
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a technical requirements extraction specialist.

Your task: Extract structured information from project descriptions with HIGH ACCURACY.

Key principles:
1. EXTRACT ONLY what is explicitly mentioned or clearly implied
2. Use EXACT role names from the provided list
3. Distinguish between technologies (languages, frameworks, platforms) and tools (development utilities)
4. Be thorough but not creative - don't invent requirements
5. Return clean JSON without markdown formatting"""

EXTRACTION_USER_PROMPT = """Extract structured requirements from this project description.

PROJECT DESCRIPTION:
\"\"\"
{description}
\"\"\"

EXTRACTION RULES:

**1. Technical Keywords** (technologies, frameworks, languages, databases, platforms):
   - Programming languages: Python, JavaScript, TypeScript, Java, C++, Go, Rust, etc.
   - Frontend frameworks: React, Vue, Angular, Next.js, Svelte, etc.
   - Backend frameworks: Django, Flask, FastAPI, Express, NestJS, Spring, etc.
   - Databases: PostgreSQL, MongoDB, MySQL, Redis, Cassandra, etc.
   - Cloud platforms: AWS, GCP, Azure, Vercel, Railway, Heroku, etc.
   - Other tech: Docker, Kubernetes, GraphQL, REST, WebSockets, etc.
   → Include ALL explicitly mentioned technologies

**2. Tools** (development utilities, NOT cloud platforms):
   - Version control: Git, GitHub, GitLab, Bitbucket
   - CI/CD: GitHub Actions, Jenkins, CircleCI, Travis CI
   - Design: Figma, Sketch, Adobe XD
   - Project management: Jira, Trello, Asana, Monday
   - Communication: Slack, Discord, Teams
   - Testing: Selenium, Cypress, Postman
   → Cloud platforms (AWS, GCP, Azure) are NOT tools - they go in technical_keywords

**3. Target Roles** (use EXACT names from this list):
{allowed_roles_formatted}

   Role inference rules:
   - Frontend tech (React, Vue, Angular) → "Developer, front-end"
   - Backend tech (Node.js, Django, Flask, Express) → "Developer, back-end"
   - Both frontend + backend → "Developer, full-stack"
   - Mobile tech (React Native, Flutter, iOS, Android) → "Developer, mobile"
   - AI/ML tech (TensorFlow, PyTorch, ML) → "Data scientist or machine learning specialist"
   - DevOps tech (Docker, Kubernetes, CI/CD) → "DevOps specialist"
   - Cloud deployment mentioned → "Cloud infrastructure engineer"
   - Embedded/IoT → "Developer, embedded applications or devices"

**4. Domain Classification**:
   - Fintech: payment, banking, trading, financial services
   - Healthcare: medical, health, patient care, diagnostics
   - E-commerce: shopping, marketplace, retail, online store
   - Education: learning, teaching, courses, e-learning
   - Manufacturing: production, industrial, supply chain
   - Gaming: game development, entertainment
   - Cybersecurity: security, penetration testing, vulnerability
   - Agriculture: farming, AgTech, IoT agriculture
   - Technology: infrastructure, SRE, general tech (use sparingly)
   - General: when no specific domain fits

**5. Seniority Level**:
   - "Junior": explicitly mentions junior/entry-level
   - "Senior": mentions senior/experienced/lead or requires 5+ years
   - "Mid": mentions mid-level or 2-4 years
   - "Mixed": not specified or needs various levels (DEFAULT)

**6. Experience Requirements**:
   - Extract minimum/maximum years if mentioned (e.g., "5+ years" → min=5)
   - Leave null if not specified

RETURN THIS EXACT JSON SCHEMA (no markdown, no backticks):
{{
  "technical_keywords": ["technology1", "technology2", "..."],
  "tools": ["tool1", "tool2", "..."],
  "target_roles": ["Role from allowed list", "..."],
  "domain": "Domain from list above",
  "seniority_level": "Junior|Mid|Senior|Mixed",
  "summary": "Clear 1-2 sentence summary of project needs",
  "min_experience": null or number,
  "max_experience": null or number,
  "required_availability": null or hours per week
}}"""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProjectRequirements:
    """Structured project requirements extracted from natural language."""
    technical_keywords: List[str]
    tools: List[str]
    target_roles: List[str]
    domain: str
    seniority_level: str
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
        parts.extend(self.technical_keywords[:10])  # Top 10 skills
        parts.extend(self.target_roles[:5])  # Top 5 roles
        return " ".join(parts)
    
    def get_all_skills(self) -> List[str]:
        """Get combined list of technical keywords and tools."""
        return self.technical_keywords + self.tools


# ============================================================================
# LLM EXTRACTOR
# ============================================================================

class LLMExtractor:
    """
    Production-grade LLM-based requirements extractor.
    
    Features:
    - Optimized prompts for high precision/recall
    - Automatic validation and normalization
    - Robust error handling with fallbacks
    - Detailed logging for debugging
    
    Example:
        extractor = LLMExtractor(client, model="gpt-4o-mini")
        requirements = extractor.extract_requirements(description)
        print(f"Found {len(requirements.technical_keywords)} skills")
    """
    
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        extra_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the extractor.
        
        Args:
            client: OpenAI-compatible API client
            model: Model identifier (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens for response
            extra_headers: Optional headers (e.g., for OpenRouter)
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        
        # Statistics tracking
        self.stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "avg_keywords_extracted": 0.0,
            "avg_roles_extracted": 0.0
        }
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make LLM API call with error handling.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/task
            
        Returns:
            Raw LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=self.extra_headers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with robust error handling.
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra whitespace
        - Partial JSON extraction
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown formatting
        cleaned = response.strip()
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try to find JSON object boundaries
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in response")
        
        json_str = cleaned[start_idx:end_idx + 1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Attempted to parse: {json_str[:500]}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    def _validate_and_normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize extracted data.
        
        Validation:
        - Ensures required fields exist
        - Normalizes role names to match ALLOWED_ROLES
        - Validates domain against known domains
        - Deduplicates items
        - Removes empty values
        
        Args:
            data: Raw extracted data
            
        Returns:
            Validated and normalized data
        """
        # Ensure lists
        technical_keywords = data.get("technical_keywords", [])
        tools = data.get("tools", [])
        target_roles = data.get("target_roles", [])
        
        if not isinstance(technical_keywords, list):
            technical_keywords = []
        if not isinstance(tools, list):
            tools = []
        if not isinstance(target_roles, list):
            target_roles = []
        
        # Deduplicate and clean
        technical_keywords = list(set(kw.strip() for kw in technical_keywords if kw and kw.strip()))
        tools = list(set(t.strip() for t in tools if t and t.strip()))
        
        # Normalize role names (fuzzy matching to ALLOWED_ROLES)
        normalized_roles = []
        for role in target_roles:
            if not role or not role.strip():
                continue
            
            role_clean = role.strip()
            
            # Exact match (case-insensitive)
            exact_match = next(
                (r for r in ALLOWED_ROLES if r.lower() == role_clean.lower()),
                None
            )
            if exact_match:
                normalized_roles.append(exact_match)
                continue
            
            # Fuzzy match - find closest role
            role_lower = role_clean.lower()
            for allowed_role in ALLOWED_ROLES:
                allowed_lower = allowed_role.lower()
                # Check if significant overlap
                if role_lower in allowed_lower or allowed_lower in role_lower:
                    normalized_roles.append(allowed_role)
                    break
            else:
                # No match found, log warning but keep original
                logger.warning(f"Role '{role_clean}' not in ALLOWED_ROLES list")
                normalized_roles.append(role_clean)
        
        normalized_roles = list(set(normalized_roles))  # Deduplicate
        
        # Validate domain
        domain = data.get("domain", "General")
        if domain not in DOMAINS:
            # Try case-insensitive match
            domain_match = next(
                (d for d in DOMAINS if d.lower() == domain.lower()),
                "General"
            )
            domain = domain_match
        
        # Validate seniority
        seniority = data.get("seniority_level", "Mixed")
        if seniority not in SENIORITY_LEVELS:
            seniority_match = next(
                (s for s in SENIORITY_LEVELS if s.lower() == seniority.lower()),
                "Mixed"
            )
            seniority = seniority_match
        
        return {
            "technical_keywords": technical_keywords,
            "tools": tools,
            "target_roles": normalized_roles,
            "domain": domain,
            "seniority_level": seniority,
            "summary": data.get("summary", "").strip() or "Project requirements",
            "min_experience": data.get("min_experience"),
            "max_experience": data.get("max_experience"),
            "required_availability": data.get("required_availability")
        }
    
    def extract_requirements(self, description: str) -> ProjectRequirements:
        """
        Extract structured requirements from project description.
        
        This is the main entry point for extraction. It:
        1. Formats the extraction prompt
        2. Calls the LLM
        3. Parses the JSON response
        4. Validates and normalizes the output
        5. Returns a ProjectRequirements object
        
        Args:
            description: Natural language project description
            
        Returns:
            ProjectRequirements object with extracted data
            
        Example:
            desc = "Build a React + Node.js e-commerce site with MongoDB"
            reqs = extractor.extract_requirements(desc)
            print(reqs.technical_keywords)  # ["React", "Node.js", "MongoDB"]
            print(reqs.domain)  # "E-commerce"
        """
        self.stats["total_extractions"] += 1
        logger.info(f"Extracting requirements from description ({len(description)} chars)...")
        
        try:
            # Format prompt with allowed roles
            allowed_roles_formatted = "\n".join(f"   - \"{role}\"" for role in ALLOWED_ROLES)
            
            user_prompt = EXTRACTION_USER_PROMPT.format(
                description=description,
                allowed_roles_formatted=allowed_roles_formatted
            )
            
            # Call LLM
            raw_response = self._call_llm(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            
            # Parse JSON
            data = self._parse_json_response(raw_response)
            
            # Validate and normalize
            validated_data = self._validate_and_normalize(data)
            
            # Update stats
            self.stats["successful_extractions"] += 1
            self.stats["avg_keywords_extracted"] = (
                (self.stats["avg_keywords_extracted"] * (self.stats["successful_extractions"] - 1) +
                 len(validated_data["technical_keywords"])) / self.stats["successful_extractions"]
            )
            self.stats["avg_roles_extracted"] = (
                (self.stats["avg_roles_extracted"] * (self.stats["successful_extractions"] - 1) +
                 len(validated_data["target_roles"])) / self.stats["successful_extractions"]
            )
            
            # Build and return ProjectRequirements
            requirements = ProjectRequirements(
                technical_keywords=validated_data["technical_keywords"],
                tools=validated_data["tools"],
                target_roles=validated_data["target_roles"],
                domain=validated_data["domain"],
                seniority_level=validated_data["seniority_level"],
                summary=validated_data["summary"],
                min_experience=validated_data.get("min_experience"),
                max_experience=validated_data.get("max_experience"),
                required_availability=validated_data.get("required_availability")
            )
            
            logger.info(
                f"✅ Extraction successful: {len(requirements.technical_keywords)} skills, "
                f"{len(requirements.tools)} tools, {len(requirements.target_roles)} roles"
            )
            
            return requirements
            
        except Exception as e:
            self.stats["failed_extractions"] += 1
            logger.error(f"❌ Extraction failed: {e}")
            
            # Return fallback requirements
            logger.warning("Returning fallback requirements (empty extraction)")
            return ProjectRequirements(
                technical_keywords=[],
                tools=[],
                target_roles=[],
                domain="General",
                seniority_level="Mixed",
                summary=description[:200] if len(description) > 200 else description
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self.stats.copy()
