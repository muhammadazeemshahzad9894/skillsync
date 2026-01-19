"""
Enhanced CSV Parser Module

Handles StackOverflow dataset with 28 columns, including:
- Multi-value skill columns (semicolon-separated)
- Personality text extraction
- Role normalization for "Other" values
- Availability parsing
"""

import re
import uuid
import logging
from typing import Dict, List, Optional, Any, IO
from dataclasses import dataclass, field

import pandas as pd

from .normalizer import normalizer

logger = logging.getLogger(__name__)


# StackOverflow column mapping to our schema
STACKOVERFLOW_COLUMN_MAP = {
    # Direct mappings
    "Candidate ID": "id",
    "Employment": "employment",
    "YearsCode": "years_code",
    "DevType": "dev_type",
    "OrgSize": "org_size",
    "WorkExp": "work_experience_years",
    "Industry": "industry",
    "WeeklyAvailabilityHours": "weekly_availability_hours",
    "PersonalityText": "personality_text",
    "ICorPM": "ic_or_pm",
    
    # Skill columns (will be merged)
    "LanguageHaveWorkedWith": "languages",
    "DatabaseHaveWorkedWith": "databases",
    "PlatformHaveWorkedWith": "platforms",
    "WebframeHaveWorkedWith": "webframes",
    "MiscTechHaveWorkedWith": "misc_tech",
    "ToolsTechHaveWorkedWith": "tools_tech",
    "NEWCollabToolsHaveWorkedWith": "collab_tools",
    
    # Additional context columns
    "LanguageAdmired": "languages_admired",
    "OpSysPersonalUse": "os_personal",
    "AIToolCurrentlyUsing": "ai_tools_using",
    "AISearchDevHaveWorkedWith": "ai_search_tools",
    "ProfessionalTech": "professional_tech",
    "ProfessionalCloud": "professional_cloud",
    "Frustration": "frustrations",
}

# Standard roles for mapping "Other" values
STANDARD_ROLES = [
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

# Belbin role keywords for extraction from personality text
BELBIN_KEYWORDS = {
    "Plant": ["creative", "innovative", "ideas", "unconventional", "solving difficult problems"],
    "Resource Investigator": ["curious", "exploring", "opportunities", "networking", "contacts"],
    "Co-ordinator": ["clarifies goals", "delegates", "mature", "confident", "chairperson"],
    "Shaper": ["challenging", "dynamic", "pressure", "drive", "courage"],
    "Monitor Evaluator": ["strategic", "analytical", "judges", "critical thinking", "objective"],
    "Teamworker": ["supportive", "cooperative", "diplomatic", "team working", "harmony"],
    "Implementer": ["practical", "reliable", "efficient", "organizer", "systematic"],
    "Completer Finisher": ["conscientious", "perfectionist", "attention to detail", "quality", "thorough"],
    "Specialist": ["dedicated", "expert", "specialized", "deep knowledge", "technical expertise"],
}


def parse_semicolon_list(value: Any) -> List[str]:
    """Parse semicolon-separated string into list."""
    if pd.isna(value) or not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def parse_availability(value: Any) -> Dict[str, Any]:
    """
    Parse availability string like "10–20" or "40+" into structured format.
    Returns dict with min_hours, max_hours, and original string.
    """
    if pd.isna(value) or not value:
        return {"min_hours": 0, "max_hours": 40, "display": "Not specified"}
    
    value_str = str(value).strip()
    
    # Handle various formats: "10–20", "10-20", "40+", "40"
    # Note: StackOverflow uses en-dash (–) not hyphen (-)
    range_match = re.match(r'(\d+)\s*[–\-]\s*(\d+)', value_str)
    if range_match:
        min_h = int(range_match.group(1))
        max_h = int(range_match.group(2))
        return {"min_hours": min_h, "max_hours": max_h, "display": value_str}
    
    plus_match = re.match(r'(\d+)\+', value_str)
    if plus_match:
        min_h = int(plus_match.group(1))
        return {"min_hours": min_h, "max_hours": 60, "display": value_str}
    
    single_match = re.match(r'(\d+)', value_str)
    if single_match:
        hours = int(single_match.group(1))
        return {"min_hours": hours, "max_hours": hours, "display": value_str}
    
    return {"min_hours": 0, "max_hours": 40, "display": value_str}


def extract_belbin_from_text(personality_text: str) -> str:
    """
    Extract Belbin team role from personality text using keyword matching.
    """
    if not personality_text or pd.isna(personality_text):
        return "Teamworker"  # Default
    
    text_lower = personality_text.lower()
    
    # Check for explicit Belbin mention
    for role in BELBIN_KEYWORDS.keys():
        if role.lower() in text_lower:
            return role
        # Check for hyphenated version
        if role.lower().replace(" ", "-") in text_lower:
            return role
    
    # Keyword scoring
    scores = {}
    for role, keywords in BELBIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[role] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return "Teamworker"


def extract_communication_style(personality_text: str) -> str:
    """Extract communication style from personality text."""
    if not personality_text or pd.isna(personality_text):
        return "Mixed"
    
    text_lower = personality_text.lower()
    
    if "written" in text_lower or "async" in text_lower or "messages" in text_lower:
        return "Indirect"
    if "meetings" in text_lower or "face to face" in text_lower or "verbal" in text_lower:
        return "Direct"
    if "mix" in text_lower or "adapt" in text_lower:
        return "Mixed"
    
    return "Mixed"


def extract_leadership_preference(personality_text: str, ic_or_pm: str = None) -> str:
    """Extract leadership preference from personality text and role."""
    if ic_or_pm and "manager" in str(ic_or_pm).lower():
        return "Lead"
    
    if not personality_text or pd.isna(personality_text):
        return "Neutral"
    
    text_lower = personality_text.lower()
    
    if "take the lead" in text_lower or "organize" in text_lower or "direction" in text_lower:
        return "Lead"
    if "shared ownership" in text_lower or "together" in text_lower:
        return "Co-lead"
    if "follow" in text_lower or "support" in text_lower:
        return "Follow"
    if "autonomous" in text_lower or "independent" in text_lower:
        return "Autonomous"
    
    return "Neutral"


def extract_conflict_style(personality_text: str) -> str:
    """Extract conflict handling style from personality text."""
    if not personality_text or pd.isna(personality_text):
        return "Collaborate"
    
    text_lower = personality_text.lower()
    
    if "avoid" in text_lower or "step back" in text_lower or "don't like" in text_lower:
        return "Avoid"
    if "collaborate" in text_lower or "work together" in text_lower:
        return "Collaborate"
    if "compromise" in text_lower or "middle ground" in text_lower:
        return "Compromise"
    if "constructive" in text_lower or "feedback" in text_lower:
        return "Collaborate"
    
    return "Collaborate"


def extract_deadline_discipline(personality_text: str) -> str:
    """Extract deadline discipline from personality text."""
    if not personality_text or pd.isna(personality_text):
        return "Flexible"
    
    text_lower = personality_text.lower()
    
    if "strict" in text_lower or "hard to stick" in text_lower or "motivate" in text_lower:
        return "Strict"
    if "depends" in text_lower or "impact" in text_lower or "workload" in text_lower:
        return "Flexible"
    if "flexible" in text_lower:
        return "Flexible"
    
    return "Flexible"


class StackOverflowCSVParser:
    """
    Parser for StackOverflow-format CSV files.
    
    Handles all 28 columns and converts to our profile schema.
    """
    
    def __init__(self, llm_client=None, llm_model: str = None):
        """
        Initialize parser.
        
        Args:
            llm_client: Optional OpenAI client for role mapping
            llm_model: Model to use for role mapping
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.normalizer = normalizer
    
    def map_other_role(self, role_text: str, context: Dict[str, Any] = None) -> str:
        """
        Map "Other (please specify):" roles to standard roles using LLM.
        """
        if not role_text or "other" not in role_text.lower():
            return role_text
        
        if not self.llm_client:
            # Fallback: use skills context to guess
            if context:
                skills = context.get("skills", [])
                if any("data" in s.lower() or "ml" in s.lower() or "tensorflow" in s.lower() for s in skills):
                    return "Data scientist or machine learning specialist"
                if any("devops" in s.lower() or "kubernetes" in s.lower() or "docker" in s.lower() for s in skills):
                    return "DevOps specialist"
                if any("mobile" in s.lower() or "flutter" in s.lower() or "react native" in s.lower() for s in skills):
                    return "Developer, mobile"
            return "Developer, full-stack"
        
        # Use LLM for mapping
        prompt = f"""Map this developer role to the closest standard role.

Original role: "{role_text}"
Context skills: {context.get('skills', []) if context else 'None'}

Standard roles (choose ONE):
{chr(10).join(f"- {r}" for r in STANDARD_ROLES)}

Return ONLY the exact role name from the list above, nothing else."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            mapped = response.choices[0].message.content.strip()
            # Validate it's in our list
            if mapped in STANDARD_ROLES:
                return mapped
            # Try partial match
            for role in STANDARD_ROLES:
                if role.lower() in mapped.lower() or mapped.lower() in role.lower():
                    return role
            return "Developer, full-stack"
        except Exception as e:
            logger.warning(f"LLM role mapping failed: {e}")
            return "Developer, full-stack"
    
    def parse_row(self, row: pd.Series, use_llm_for_roles: bool = True) -> Dict[str, Any]:
        """
        Parse a single CSV row into profile schema.
        """
        # Collect all skills from multiple columns
        all_skills = []
        all_skills.extend(parse_semicolon_list(row.get("LanguageHaveWorkedWith", "")))
        all_skills.extend(parse_semicolon_list(row.get("DatabaseHaveWorkedWith", "")))
        all_skills.extend(parse_semicolon_list(row.get("PlatformHaveWorkedWith", "")))
        all_skills.extend(parse_semicolon_list(row.get("WebframeHaveWorkedWith", "")))
        all_skills.extend(parse_semicolon_list(row.get("MiscTechHaveWorkedWith", "")))
        
        # Normalize skills
        all_skills = self.normalizer.normalize_skills(all_skills)
        
        # Tools from dedicated column
        tools = parse_semicolon_list(row.get("ToolsTechHaveWorkedWith", ""))
        tools.extend(parse_semicolon_list(row.get("NEWCollabToolsHaveWorkedWith", "")))
        tools = list(set(tools))  # Dedupe
        
        # Parse availability
        availability = parse_availability(row.get("WeeklyAvailabilityHours", ""))
        
        # Extract personality traits from text
        personality_text = str(row.get("PersonalityText", "")) if pd.notna(row.get("PersonalityText")) else ""
        ic_or_pm = str(row.get("ICorPM", "")) if pd.notna(row.get("ICorPM")) else ""
        
        # Handle role
        raw_role = str(row.get("DevType", "Developer, full-stack")) if pd.notna(row.get("DevType")) else "Developer, full-stack"
        
        context = {"skills": all_skills, "tools": tools}
        
        if "other" in raw_role.lower() and use_llm_for_roles:
            role = self.map_other_role(raw_role, context)
        else:
            role = raw_role
        
        # Normalize role
        role = self.normalizer.normalize_role(role)
        
        # Parse experience
        work_exp = row.get("WorkExp", 0)
        if pd.isna(work_exp):
            work_exp = row.get("YearsCode", 0)
        if pd.isna(work_exp):
            work_exp = 2
        
        # Industry handling
        industry = str(row.get("Industry", "General")) if pd.notna(row.get("Industry")) else "General"
        if "other" in industry.lower():
            industry = "Other"
        
        # Build profile
        profile = {
            "id": str(row.get("Candidate ID", uuid.uuid4())),
            "name": f"Candidate {row.get('Candidate ID', 'Unknown')}",  # No names in SO data
            "role": role,
            "constraints": {
                "weekly_availability_hours": availability["display"],
                "min_hours": availability["min_hours"],
                "max_hours": availability["max_hours"],
            },
            "metadata": {
                "dev_type": role,
                "work_experience_years": str(float(work_exp) if work_exp else 2),
                "years_code": str(row.get("YearsCode", 0)) if pd.notna(row.get("YearsCode")) else "0",
                "employment": str(row.get("Employment", "Employed, full-time")) if pd.notna(row.get("Employment")) else "Employed, full-time",
                "org_size": str(row.get("OrgSize", "Unknown")) if pd.notna(row.get("OrgSize")) else "Unknown",
                "industry": industry,
                "ic_or_pm": ic_or_pm,
                "professional_cloud": str(row.get("ProfessionalCloud", "")) if pd.notna(row.get("ProfessionalCloud")) else "",
            },
            "technical": {
                "skills": all_skills,
                "tools": tools,
                "languages_admired": parse_semicolon_list(row.get("LanguageAdmired", "")),
                "ai_tools": parse_semicolon_list(row.get("AIToolCurrentlyUsing", "")),
                "os_preference": parse_semicolon_list(row.get("OpSysPersonalUse", "")),
            },
            "personality": {
                "Belbin_team_role": extract_belbin_from_text(personality_text),
                "personality_text": personality_text[:500] if personality_text else "",  # Truncate for storage
            },
            "collaboration": {
                "communication_style": extract_communication_style(personality_text),
                "conflict_style": extract_conflict_style(personality_text),
                "leadership_preference": extract_leadership_preference(personality_text, ic_or_pm),
                "deadline_discipline": extract_deadline_discipline(personality_text),
            },
            "learning_behavior": {
                "learning_orientation": "High" if "learning" in personality_text.lower() else "Medium",
                "knowledge_sharing": "High" if "feedback" in personality_text.lower() or "share" in personality_text.lower() else "Medium",
            },
            "context": {
                "frustrations": str(row.get("Frustration", "")) if pd.notna(row.get("Frustration")) else "",
                "professional_tech": str(row.get("ProfessionalTech", "")) if pd.notna(row.get("ProfessionalTech")) else "",
                "ai_ethics_concerns": str(row.get("AIEthics", "")) if pd.notna(row.get("AIEthics")) else "",
            }
        }
        
        return profile
    
    def parse_csv(
        self,
        file_or_path: Any,
        use_llm_for_roles: bool = True,
        min_availability_hours: int = None
    ) -> List[Dict[str, Any]]:
        """
        Parse entire CSV file.
        
        Args:
            file_or_path: CSV file path or file-like object
            use_llm_for_roles: Whether to use LLM for "Other" role mapping
            min_availability_hours: Filter out candidates below this availability
            
        Returns:
            List of profile dictionaries
        """
        try:
            df = pd.read_csv(file_or_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise ValueError(f"Could not read CSV file: {e}")
        
        logger.info(f"Parsing {len(df)} rows from CSV...")
        
        profiles = []
        skipped_availability = 0
        
        for idx, row in df.iterrows():
            try:
                profile = self.parse_row(row, use_llm_for_roles)
                
                # Filter by availability if specified
                if min_availability_hours:
                    max_hours = profile["constraints"].get("max_hours", 40)
                    if max_hours < min_availability_hours:
                        skipped_availability += 1
                        continue
                
                profiles.append(profile)
                
            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                continue
        
        if skipped_availability > 0:
            logger.info(f"Filtered out {skipped_availability} candidates below {min_availability_hours}h availability")
        
        logger.info(f"Successfully parsed {len(profiles)} profiles from CSV")
        return profiles
    
    def get_column_stats(self, file_or_path: Any) -> Dict[str, Any]:
        """
        Get statistics about the CSV for display.
        """
        df = pd.read_csv(file_or_path)
        
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_roles": df["DevType"].nunique() if "DevType" in df.columns else 0,
            "other_roles_count": len(df[df["DevType"].str.contains("Other", na=False)]) if "DevType" in df.columns else 0,
            "industries": df["Industry"].value_counts().to_dict() if "Industry" in df.columns else {},
            "availability_distribution": df["WeeklyAvailabilityHours"].value_counts().to_dict() if "WeeklyAvailabilityHours" in df.columns else {},
        }
        
        return stats


# Legacy parser for backward compatibility with simple CSVs
class SimpleCSVParser:
    """Simple CSV parser for basic format (Name, Role, Experience, Skills)."""
    
    def parse_csv(self, file_or_path: Any) -> List[Dict[str, Any]]:
        """Parse simple CSV format."""
        df = pd.read_csv(file_or_path)
        profiles = []
        
        for idx, row in df.iterrows():
            # Try to detect column names
            name = row.get("Name") or row.get("name") or f"Candidate {idx+1}"
            role = row.get("Role") or row.get("role") or "Developer, full-stack"
            exp = row.get("Experience") or row.get("experience") or row.get("WorkExp") or 2
            skills_str = row.get("Skills") or row.get("skills") or ""
            
            skills = [s.strip() for s in str(skills_str).split(",") if s.strip()]
            
            profile = {
                "id": str(uuid.uuid4()),
                "name": str(name),
                "role": normalizer.normalize_role(str(role)),
                "constraints": {"weekly_availability_hours": "40", "min_hours": 40, "max_hours": 40},
                "metadata": {
                    "dev_type": normalizer.normalize_role(str(role)),
                    "work_experience_years": str(float(exp) if exp else 2),
                    "industry": "General",
                },
                "technical": {
                    "skills": normalizer.normalize_skills(skills),
                    "tools": [],
                },
                "personality": {"Belbin_team_role": "Teamworker"},
                "collaboration": {
                    "communication_style": "Mixed",
                    "conflict_style": "Collaborate",
                    "leadership_preference": "Neutral",
                    "deadline_discipline": "Flexible",
                },
                "learning_behavior": {"learning_orientation": "Medium", "knowledge_sharing": "Medium"},
            }
            profiles.append(profile)
        
        return profiles


def detect_csv_format(file_or_path: Any) -> str:
    """
    Detect CSV format based on columns.
    
    Returns: "stackoverflow" or "simple"
    """
    df = pd.read_csv(file_or_path, nrows=1)
    columns = set(df.columns)
    
    stackoverflow_indicators = {"Candidate ID", "DevType", "LanguageHaveWorkedWith", "PersonalityText"}
    
    if stackoverflow_indicators & columns:
        return "stackoverflow"
    return "simple"


# Convenience function
def parse_csv_auto(file_or_path: Any, llm_client=None, llm_model: str = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Auto-detect CSV format and parse accordingly.
    """
    # Need to read file twice, so handle file objects
    if hasattr(file_or_path, 'seek'):
        file_or_path.seek(0)
    
    format_type = detect_csv_format(file_or_path)
    
    if hasattr(file_or_path, 'seek'):
        file_or_path.seek(0)
    
    if format_type == "stackoverflow":
        parser = StackOverflowCSVParser(llm_client=llm_client, llm_model=llm_model)
        return parser.parse_csv(file_or_path, **kwargs)
    else:
        parser = SimpleCSVParser()
        return parser.parse_csv(file_or_path)
