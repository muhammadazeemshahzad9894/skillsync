"""
Unit Tests for SkillSync Preprocessing Module

Run with: pytest tests/test_normalizer.py -v
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.normalizer import SkillNormalizer, normalizer


class TestSkillNormalizer:
    """Test cases for SkillNormalizer class."""
    
    def test_normalize_skill_basic(self):
        """Test basic skill normalization."""
        assert normalizer.normalize_skill("js") == "JavaScript"
        assert normalizer.normalize_skill("JS") == "JavaScript"
        assert normalizer.normalize_skill("javascript") == "JavaScript"
    
    def test_normalize_skill_python(self):
        """Test Python variants."""
        assert normalizer.normalize_skill("python") == "Python"
        assert normalizer.normalize_skill("python3") == "Python"
        assert normalizer.normalize_skill("py") == "Python"
    
    def test_normalize_skill_react(self):
        """Test React variants."""
        assert normalizer.normalize_skill("reactjs") == "React"
        assert normalizer.normalize_skill("react.js") == "React"
        assert normalizer.normalize_skill("React") == "React"
    
    def test_normalize_skill_cloud(self):
        """Test cloud platform normalization."""
        assert normalizer.normalize_skill("aws") == "AWS"
        assert normalizer.normalize_skill("amazon web services") == "AWS"
        assert normalizer.normalize_skill("gcp") == "Google Cloud"
    
    def test_normalize_skill_unknown(self):
        """Test unknown skill returns original."""
        assert normalizer.normalize_skill("UnknownSkill") == "UnknownSkill"
        assert normalizer.normalize_skill("CustomTool123") == "CustomTool123"
    
    def test_normalize_skill_empty(self):
        """Test empty input handling."""
        assert normalizer.normalize_skill("") == ""
        assert normalizer.normalize_skill(None) == ""
    
    def test_normalize_skills_list(self):
        """Test normalizing a list of skills."""
        skills = ["js", "python", "react", "aws"]
        normalized = normalizer.normalize_skills(skills)
        
        assert "JavaScript" in normalized
        assert "Python" in normalized
        assert "React" in normalized
        assert "AWS" in normalized
    
    def test_normalize_skills_deduplication(self):
        """Test that duplicates are removed."""
        skills = ["js", "javascript", "JS", "Python"]
        normalized = normalizer.normalize_skills(skills)
        
        # Should only have one JavaScript entry
        js_count = sum(1 for s in normalized if s == "JavaScript")
        assert js_count == 1
    
    def test_normalize_role_basic(self):
        """Test basic role normalization."""
        assert normalizer.normalize_role("frontend developer") == "Developer, front-end"
        assert normalizer.normalize_role("backend developer") == "Developer, back-end"
        assert normalizer.normalize_role("fullstack developer") == "Developer, full-stack"
    
    def test_normalize_role_devops(self):
        """Test DevOps role variants."""
        assert normalizer.normalize_role("devops engineer") == "DevOps specialist"
        assert normalizer.normalize_role("sre") == "DevOps specialist"
        assert normalizer.normalize_role("site reliability engineer") == "DevOps specialist"
    
    def test_normalize_role_unknown(self):
        """Test unknown role returns original."""
        assert normalizer.normalize_role("Chief Innovation Officer") == "Chief Innovation Officer"
    
    def test_extract_skills_from_text(self):
        """Test skill extraction from free text."""
        text = "I have experience with Python and JavaScript, and I use AWS for deployment."
        skills = normalizer.extract_skills_from_text(text)
        
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "AWS" in skills
    
    def test_extract_skills_from_text_empty(self):
        """Test empty text handling."""
        assert normalizer.extract_skills_from_text("") == []
        assert normalizer.extract_skills_from_text(None) == []
    
    def test_calculate_skill_overlap_identical(self):
        """Test overlap calculation with identical skills."""
        skills = ["Python", "JavaScript", "React"]
        overlap = normalizer.calculate_skill_overlap(skills, skills)
        assert overlap == 1.0
    
    def test_calculate_skill_overlap_partial(self):
        """Test overlap calculation with partial match."""
        skills1 = ["Python", "JavaScript", "React"]
        skills2 = ["Python", "Java", "Angular"]
        overlap = normalizer.calculate_skill_overlap(skills1, skills2)
        
        # 1 common skill (Python) out of 5 unique skills
        assert 0 < overlap < 1
    
    def test_calculate_skill_overlap_none(self):
        """Test overlap calculation with no common skills."""
        skills1 = ["Python", "Django"]
        skills2 = ["Java", "Spring Boot"]
        overlap = normalizer.calculate_skill_overlap(skills1, skills2)
        assert overlap == 0.0
    
    def test_calculate_skill_overlap_empty(self):
        """Test overlap calculation with empty lists."""
        assert normalizer.calculate_skill_overlap([], ["Python"]) == 0.0
        assert normalizer.calculate_skill_overlap(["Python"], []) == 0.0


class TestSkillNormalizerCustom:
    """Test custom normalizer configuration."""
    
    def test_custom_synonyms(self):
        """Test normalizer with custom synonyms."""
        custom_synonyms = {
            "ml": "Machine Learning",
            "ai": "Artificial Intelligence",
        }
        custom_normalizer = SkillNormalizer(skill_synonyms=custom_synonyms)
        
        assert custom_normalizer.normalize_skill("ml") == "Machine Learning"
        assert custom_normalizer.normalize_skill("ai") == "Artificial Intelligence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
