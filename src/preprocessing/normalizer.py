"""
Skill Normalization Module

Handles standardization of skills, tools, and role names to ensure
consistent matching across different naming conventions.
"""

import re
from typing import Dict, List, Set


# Comprehensive skill synonym mapping
SKILL_SYNONYMS: Dict[str, str] = {
    # JavaScript variants
    "js": "JavaScript",
    "javascript": "JavaScript",
    "es6": "JavaScript",
    "es2015": "JavaScript",
    "ecmascript": "JavaScript",
    
    # TypeScript
    "ts": "TypeScript",
    "typescript": "TypeScript",
    
    # Python variants
    "python3": "Python",
    "python2": "Python",
    "py": "Python",
    
    # React variants
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    
    # Node variants
    "nodejs": "Node.js",
    "node": "Node.js",
    "node.js": "Node.js",
    
    # Vue variants
    "vuejs": "Vue.js",
    "vue": "Vue.js",
    "vue.js": "Vue.js",
    
    # Angular variants
    "angularjs": "Angular",
    "angular.js": "Angular",
    "angular2": "Angular",
    
    # Database variants
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "mssql": "SQL Server",
    "sqlserver": "SQL Server",
    "sql server": "SQL Server",
    
    # Cloud platforms
    "amazon web services": "AWS",
    "aws": "AWS",
    "google cloud platform": "Google Cloud",
    "gcp": "Google Cloud",
    "azure": "Azure",
    "microsoft azure": "Azure",
    
    # DevOps tools
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "terraform": "Terraform",
    "tf": "Terraform",
    
    # ML/AI
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "artificial intelligence": "AI",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    
    # Languages
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",
    "c++": "C++",
    "cpp": "C++",
    "golang": "Go",
    "go": "Go",
    
    # Frameworks
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    ".net": ".NET",
    "dotnet": ".NET",
    "asp.net": "ASP.NET",
    "aspnet": "ASP.NET",
    
    # Version Control
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "bitbucket": "Bitbucket",
    
    # Methodologies
    "agile": "Agile",
    "scrum": "Scrum",
    "kanban": "Kanban",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    
    # Web technologies
    "html": "HTML/CSS",
    "css": "HTML/CSS",
    "html/css": "HTML/CSS",
    "html5": "HTML/CSS",
    "css3": "HTML/CSS",
    "sass": "SASS/SCSS",
    "scss": "SASS/SCSS",
    
    # Mobile
    "react native": "React Native",
    "reactnative": "React Native",
    "flutter": "Flutter",
    "ios": "iOS",
    "android": "Android",
    "swift": "Swift",
    "kotlin": "Kotlin",
}

# Role synonym mapping
ROLE_SYNONYMS: Dict[str, str] = {
    # Frontend
    "frontend developer": "Developer, front-end",
    "front-end developer": "Developer, front-end",
    "front end developer": "Developer, front-end",
    "ui developer": "Developer, front-end",
    "web developer": "Developer, front-end",
    
    # Backend
    "backend developer": "Developer, back-end",
    "back-end developer": "Developer, back-end",
    "back end developer": "Developer, back-end",
    "server developer": "Developer, back-end",
    "api developer": "Developer, back-end",
    
    # Full-stack
    "fullstack developer": "Developer, full-stack",
    "full-stack developer": "Developer, full-stack",
    "full stack developer": "Developer, full-stack",
    "software developer": "Developer, full-stack",
    "software engineer": "Developer, full-stack",
    "swe": "Developer, full-stack",
    
    # Mobile
    "mobile developer": "Developer, mobile",
    "ios developer": "Developer, mobile",
    "android developer": "Developer, mobile",
    "app developer": "Developer, mobile",
    
    # DevOps
    "devops engineer": "DevOps specialist",
    "devops": "DevOps specialist",
    "site reliability engineer": "DevOps specialist",
    "sre": "DevOps specialist",
    "platform engineer": "DevOps specialist",
    
    # Data
    "data scientist": "Data scientist or machine learning specialist",
    "ml engineer": "Data scientist or machine learning specialist",
    "machine learning engineer": "Data scientist or machine learning specialist",
    "ai engineer": "Data scientist or machine learning specialist",
    "data analyst": "Data scientist or machine learning specialist",
    
    # Cloud
    "cloud engineer": "Cloud infrastructure engineer",
    "cloud architect": "Cloud infrastructure engineer",
    "aws engineer": "Cloud infrastructure engineer",
    
    # Sysadmin
    "system administrator": "System administrator",
    "sysadmin": "System administrator",
    "linux administrator": "System administrator",
}


class SkillNormalizer:
    """
    Normalizes skills, tools, and roles to canonical forms.
    
    Example:
        normalizer = SkillNormalizer()
        normalized = normalizer.normalize_skill("ReactJS")  # Returns "React"
    """
    
    def __init__(
        self,
        skill_synonyms: Dict[str, str] = None,
        role_synonyms: Dict[str, str] = None
    ):
        """
        Initialize normalizer with optional custom synonym mappings.
        
        Args:
            skill_synonyms: Custom skill synonym dictionary
            role_synonyms: Custom role synonym dictionary
        """
        self.skill_synonyms = skill_synonyms or SKILL_SYNONYMS
        self.role_synonyms = role_synonyms or ROLE_SYNONYMS
        
        # Create lowercase lookup for case-insensitive matching
        self._skill_lookup = {k.lower(): v for k, v in self.skill_synonyms.items()}
        self._role_lookup = {k.lower(): v for k, v in self.role_synonyms.items()}
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a single skill to its canonical form.
        
        Args:
            skill: Raw skill string
            
        Returns:
            Normalized skill string
        """
        if not skill:
            return ""
        
        cleaned = skill.strip().lower()
        return self._skill_lookup.get(cleaned, skill.strip())
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skills, removing duplicates.
        
        Args:
            skills: List of raw skill strings
            
        Returns:
            List of unique normalized skills
        """
        normalized = []
        seen = set()
        
        for skill in skills:
            norm = self.normalize_skill(skill)
            if norm and norm.lower() not in seen:
                normalized.append(norm)
                seen.add(norm.lower())
        
        return normalized
    
    def normalize_role(self, role: str) -> str:
        """
        Normalize a job role to its canonical form.
        
        Args:
            role: Raw role string
            
        Returns:
            Normalized role string
        """
        if not role:
            return "Developer, full-stack"
        
        cleaned = role.strip().lower()
        return self._role_lookup.get(cleaned, role.strip())
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract and normalize skills mentioned in free text.
        
        Args:
            text: Free-form text potentially containing skill mentions
            
        Returns:
            List of normalized skills found in text
        """
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = set()
        
        # Check for each known skill synonym
        for synonym, canonical in self._skill_lookup.items():
            # Word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(canonical)
        
        return list(found_skills)
    
    def calculate_skill_overlap(
        self,
        skills1: List[str],
        skills2: List[str]
    ) -> float:
        """
        Calculate Jaccard similarity between two skill sets.
        
        Args:
            skills1: First list of skills
            skills2: Second list of skills
            
        Returns:
            Jaccard similarity score (0-1)
        """
        norm1 = set(s.lower() for s in self.normalize_skills(skills1))
        norm2 = set(s.lower() for s in self.normalize_skills(skills2))
        
        if not norm1 or not norm2:
            return 0.0
        
        intersection = norm1 & norm2
        union = norm1 | norm2
        
        return len(intersection) / len(union)


# Singleton instance for convenience
normalizer = SkillNormalizer()
