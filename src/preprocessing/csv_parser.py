"""
CSV Parser Module

Handles CSV file ingestion with flexible column mapping,
validation, and profile generation.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any, IO
from dataclasses import dataclass, field

import pandas as pd

from .normalizer import normalizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CSVColumnMapping:
    """Configuration for mapping CSV columns to profile fields."""
    name: str = "Name"
    role: str = "Role"
    experience: str = "Experience"
    skills: str = "Skills"
    tools: str = "Tools"
    industry: str = "Industry"
    availability: str = "Availability"
    belbin_role: str = "Belbin"
    communication_style: str = "CommunicationStyle"
    
    # Alternative column names to check
    alternatives: Dict[str, List[str]] = field(default_factory=lambda: {
        "name": ["Name", "Full Name", "FullName", "Candidate Name", "Employee Name"],
        "role": ["Role", "Title", "Job Title", "Position", "Job Role"],
        "experience": ["Experience", "Years Experience", "YearsExp", "Exp", "Work Experience"],
        "skills": ["Skills", "Technical Skills", "Programming Languages", "Technologies"],
        "tools": ["Tools", "Platforms", "Software", "Tech Stack"],
        "industry": ["Industry", "Domain", "Sector"],
        "availability": ["Availability", "Hours", "Weekly Hours"],
        "belbin_role": ["Belbin", "Team Role", "Belbin Role"],
    })


class CSVParser:
    """
    Parses CSV files into structured employee profiles.
    
    Supports flexible column mapping and handles various CSV formats.
    
    Example:
        parser = CSVParser()
        profiles = parser.parse_csv(uploaded_file)
    """
    
    def __init__(self, column_mapping: CSVColumnMapping = None):
        """
        Initialize CSV parser.
        
        Args:
            column_mapping: Custom column mapping configuration
        """
        self.mapping = column_mapping or CSVColumnMapping()
        self.normalizer = normalizer
    
    def _find_column(self, df: pd.DataFrame, field_name: str) -> Optional[str]:
        """
        Find a column in DataFrame, checking alternatives.
        
        Args:
            df: DataFrame to search
            field_name: Field name to find
            
        Returns:
            Actual column name found, or None
        """
        alternatives = self.mapping.alternatives.get(field_name, [])
        
        # Check primary name first
        primary = getattr(self.mapping, field_name, None)
        if primary and primary in df.columns:
            return primary
        
        # Check alternatives
        for alt in alternatives:
            if alt in df.columns:
                return alt
            # Case-insensitive check
            for col in df.columns:
                if col.lower() == alt.lower():
                    return col
        
        return None
    
    def _parse_skills_string(self, skills_str: Any) -> List[str]:
        """Parse skills from various string formats."""
        if pd.isna(skills_str) or not skills_str:
            return []
        
        skills_str = str(skills_str)
        
        # Try different delimiters
        for delimiter in [",", ";", "|", "\n"]:
            if delimiter in skills_str:
                skills = [s.strip() for s in skills_str.split(delimiter)]
                return self.normalizer.normalize_skills([s for s in skills if s])
        
        # Single skill
        return self.normalizer.normalize_skills([skills_str.strip()])
    
    def _parse_experience(self, exp_value: Any) -> str:
        """Parse experience value to string format."""
        if pd.isna(exp_value):
            return "2.0"
        
        try:
            # Handle numeric values
            if isinstance(exp_value, (int, float)):
                return str(float(exp_value))
            
            # Handle string values like "5 years" or "5+"
            exp_str = str(exp_value).lower()
            import re
            numbers = re.findall(r'\d+\.?\d*', exp_str)
            if numbers:
                return str(float(numbers[0]))
            
            return "2.0"
        except (ValueError, TypeError):
            return "2.0"
    
    def validate_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate CSV structure and content.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result with errors and warnings
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "row_count": len(df),
            "columns_found": list(df.columns)
        }
        
        # Check for required columns
        name_col = self._find_column(df, "name")
        if not name_col:
            result["errors"].append("Missing required column: Name")
            result["valid"] = False
        
        # Check for recommended columns
        recommended = ["role", "experience", "skills"]
        for field in recommended:
            if not self._find_column(df, field):
                result["warnings"].append(f"Missing recommended column: {field}")
        
        # Check for empty rows
        if df.empty:
            result["errors"].append("CSV file is empty")
            result["valid"] = False
        
        return result
    
    def parse_csv(
        self,
        file_or_path: Any,
        validate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV file into list of profile dictionaries.
        
        Args:
            file_or_path: File path string, file-like object, or bytes
            validate: Whether to validate before parsing
            
        Returns:
            List of profile dictionaries
        """
        # Read CSV
        try:
            df = pd.read_csv(file_or_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise ValueError(f"Could not read CSV file: {e}")
        
        # Validate if requested
        if validate:
            validation = self.validate_csv(df)
            if not validation["valid"]:
                errors = "; ".join(validation["errors"])
                raise ValueError(f"CSV validation failed: {errors}")
            
            for warning in validation.get("warnings", []):
                logger.warning(warning)
        
        # Find columns
        name_col = self._find_column(df, "name")
        role_col = self._find_column(df, "role")
        exp_col = self._find_column(df, "experience")
        skills_col = self._find_column(df, "skills")
        tools_col = self._find_column(df, "tools")
        industry_col = self._find_column(df, "industry")
        availability_col = self._find_column(df, "availability")
        belbin_col = self._find_column(df, "belbin_role")
        
        # Parse rows
        profiles = []
        
        for idx, row in df.iterrows():
            try:
                # Extract values with defaults
                name = row.get(name_col, f"Candidate_{idx}") if name_col else f"Candidate_{idx}"
                role = row.get(role_col, "Developer, full-stack") if role_col else "Developer, full-stack"
                experience = self._parse_experience(row.get(exp_col)) if exp_col else "2.0"
                skills = self._parse_skills_string(row.get(skills_col)) if skills_col else []
                tools = self._parse_skills_string(row.get(tools_col)) if tools_col else []
                industry = row.get(industry_col, "General") if industry_col else "General"
                availability = row.get(availability_col, "40") if availability_col else "40"
                belbin_role = row.get(belbin_col, "Teamworker") if belbin_col else "Teamworker"
                
                # Normalize role
                role = self.normalizer.normalize_role(str(role))
                
                # Create profile
                profile = {
                    "id": str(uuid.uuid4()),
                    "name": str(name).strip() if pd.notna(name) else f"Candidate_{idx}",
                    "role": role,
                    "constraints": {
                        "weekly_availability_hours": str(availability)
                    },
                    "metadata": {
                        "dev_type": role,
                        "work_experience_years": experience,
                        "years_code": experience,
                        "employment": "Employed, full-time",
                        "org_size": "Unknown",
                        "industry": str(industry) if pd.notna(industry) else "General"
                    },
                    "technical": {
                        "skills": skills,
                        "tools": tools
                    },
                    "personality": {
                        "Belbin_team_role": str(belbin_role) if pd.notna(belbin_role) else "Teamworker"
                    },
                    "collaboration": {
                        "communication_style": "Mixed",
                        "conflict_style": "Collaborate",
                        "leadership_preference": "Neutral",
                        "deadline_discipline": "Flexible"
                    },
                    "learning_behavior": {
                        "learning_orientation": "Medium",
                        "knowledge_sharing": "Medium"
                    }
                }
                
                profiles.append(profile)
                
            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(profiles)} profiles from CSV")
        return profiles
    
    def generate_sample_csv(self) -> str:
        """
        Generate a sample CSV template.
        
        Returns:
            CSV content as string
        """
        sample_data = """Name,Role,Experience,Skills,Tools,Industry,Belbin
John Smith,Full-stack Developer,5,Python; JavaScript; React; Node.js,AWS; Docker; Git,Fintech,Implementer
Sarah Johnson,Data Scientist,3,Python; TensorFlow; SQL; Pandas,Jupyter; AWS; Git,Healthcare,Specialist
Mike Chen,DevOps Engineer,7,Python; Bash; Kubernetes; Terraform,AWS; GCP; Jenkins,E-commerce,Monitor Evaluator
Emily Brown,Frontend Developer,2,JavaScript; TypeScript; React; Vue.js,Figma; Git; VSCode,Education,Plant
"""
        return sample_data


# Singleton instance
csv_parser = CSVParser()
