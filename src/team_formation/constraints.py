"""
Team Constraints Module

Validates team compositions against project requirements and
reports constraint violations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from ..preprocessing.normalizer import normalizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConstraintViolation:
    """Represents a single constraint violation."""
    constraint_type: str
    severity: str  # "error", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of team validation against constraints."""
    is_valid: bool
    violations: List[ConstraintViolation]
    coverage_score: float
    warnings: List[str]
    
    @property
    def has_errors(self) -> bool:
        return any(v.severity == "error" for v in self.violations)
    
    @property
    def has_warnings(self) -> bool:
        return any(v.severity == "warning" for v in self.violations)


class TeamConstraintValidator:
    """
    Validates team compositions against project constraints.
    
    Checks:
    - Required skill coverage
    - Role diversity
    - Experience requirements
    - Team size constraints
    - Availability requirements
    
    Example:
        validator = TeamConstraintValidator()
        result = validator.validate(team, requirements)
        if not result.is_valid:
            print(result.violations)
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        min_skill_coverage: float = 0.5
    ):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
            min_skill_coverage: Minimum required skill coverage ratio
        """
        self.strict_mode = strict_mode
        self.min_skill_coverage = min_skill_coverage
    
    def validate(
        self,
        team_members: List[Dict[str, Any]],
        required_skills: List[str] = None,
        target_roles: List[str] = None,
        min_experience: float = None,
        max_experience: float = None,
        required_team_size: int = None,
        min_availability_hours: int = None
    ) -> ValidationResult:
        """
        Validate a team against all constraints.
        
        Args:
            team_members: List of team member profiles
            required_skills: Skills that should be covered
            target_roles: Roles that should be represented
            min_experience: Minimum average experience
            max_experience: Maximum average experience
            required_team_size: Expected team size
            min_availability_hours: Minimum weekly hours per member
            
        Returns:
            ValidationResult with all violations and coverage score
        """
        violations = []
        warnings = []
        
        # Check team size
        if required_team_size and len(team_members) != required_team_size:
            violations.append(ConstraintViolation(
                constraint_type="team_size",
                severity="error",
                message=f"Team size mismatch: expected {required_team_size}, got {len(team_members)}",
                details={"expected": required_team_size, "actual": len(team_members)}
            ))
        
        # Check skill coverage
        coverage_score = 1.0
        if required_skills:
            coverage_score, covered, missing = self._check_skill_coverage(
                team_members, required_skills
            )
            
            if coverage_score < 1.0:
                severity = "error" if coverage_score < self.min_skill_coverage else "warning"
                violations.append(ConstraintViolation(
                    constraint_type="skill_coverage",
                    severity=severity,
                    message=f"Skill coverage: {coverage_score:.0%} ({len(covered)}/{len(required_skills)} skills)",
                    details={"covered": covered, "missing": missing, "coverage": coverage_score}
                ))
                
                if missing:
                    warnings.append(f"Missing skills: {', '.join(missing)}")
        
        # Check role coverage
        if target_roles:
            role_coverage, matched, unmatched = self._check_role_coverage(
                team_members, target_roles
            )
            
            if role_coverage < 1.0:
                violations.append(ConstraintViolation(
                    constraint_type="role_coverage",
                    severity="warning",
                    message=f"Role coverage: {role_coverage:.0%}",
                    details={"matched": matched, "unmatched": unmatched}
                ))
                
                if unmatched:
                    warnings.append(f"Missing roles: {', '.join(unmatched)}")
        
        # Check experience requirements
        if min_experience is not None or max_experience is not None:
            exp_valid, avg_exp, issues = self._check_experience(
                team_members, min_experience, max_experience
            )
            
            if not exp_valid:
                violations.append(ConstraintViolation(
                    constraint_type="experience",
                    severity="warning",
                    message=f"Experience constraint: avg={avg_exp:.1f} years",
                    details={"average": avg_exp, "issues": issues}
                ))
                warnings.extend(issues)
        
        # Check availability
        if min_availability_hours:
            avail_issues = self._check_availability(team_members, min_availability_hours)
            if avail_issues:
                violations.append(ConstraintViolation(
                    constraint_type="availability",
                    severity="warning",
                    message=f"{len(avail_issues)} members below minimum availability",
                    details={"issues": avail_issues}
                ))
                warnings.append(f"Low availability: {', '.join(avail_issues)}")
        
        # Check for duplicate members
        duplicates = self._check_duplicates(team_members)
        if duplicates:
            violations.append(ConstraintViolation(
                constraint_type="duplicates",
                severity="error",
                message=f"Duplicate team members found",
                details={"duplicates": duplicates}
            ))
        
        # Determine overall validity
        is_valid = not any(v.severity == "error" for v in violations)
        if self.strict_mode:
            is_valid = is_valid and not any(v.severity == "warning" for v in violations)
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            coverage_score=coverage_score,
            warnings=warnings
        )
    
    def _check_skill_coverage(
        self,
        team_members: List[Dict[str, Any]],
        required_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Check how well team covers required skills."""
        # Collect all team skills (normalized)
        team_skills = set()
        for member in team_members:
            skills = member.get("technical", {}).get("skills", [])
            tools = member.get("technical", {}).get("tools", [])
            all_skills = skills + tools
            
            for skill in all_skills:
                normalized = normalizer.normalize_skill(skill).lower()
                team_skills.add(normalized)
        
        # Normalize required skills
        required_normalized = {}
        for skill in required_skills:
            normalized = normalizer.normalize_skill(skill).lower()
            required_normalized[normalized] = skill
        
        # Check coverage
        covered = []
        missing = []
        
        for norm_skill, original_skill in required_normalized.items():
            if norm_skill in team_skills:
                covered.append(original_skill)
            else:
                # Check for partial matches
                partial_match = any(
                    norm_skill in ts or ts in norm_skill 
                    for ts in team_skills
                )
                if partial_match:
                    covered.append(original_skill)
                else:
                    missing.append(original_skill)
        
        coverage = len(covered) / len(required_skills) if required_skills else 1.0
        
        return coverage, covered, missing
    
    def _check_role_coverage(
        self,
        team_members: List[Dict[str, Any]],
        target_roles: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Check if team covers required roles."""
        # Collect team roles
        team_roles = set()
        for member in team_members:
            role = member.get("role", "")
            dev_type = member.get("metadata", {}).get("dev_type", "")
            
            team_roles.add(normalizer.normalize_role(role).lower())
            if dev_type:
                team_roles.add(normalizer.normalize_role(dev_type).lower())
        
        # Check coverage
        matched = []
        unmatched = []
        
        for target in target_roles:
            target_norm = normalizer.normalize_role(target).lower()
            
            if target_norm in team_roles:
                matched.append(target)
            else:
                # Partial match
                found = any(target_norm in tr or tr in target_norm for tr in team_roles)
                if found:
                    matched.append(target)
                else:
                    unmatched.append(target)
        
        coverage = len(matched) / len(target_roles) if target_roles else 1.0
        
        return coverage, matched, unmatched
    
    def _check_experience(
        self,
        team_members: List[Dict[str, Any]],
        min_exp: Optional[float],
        max_exp: Optional[float]
    ) -> Tuple[bool, float, List[str]]:
        """Check experience constraints."""
        experiences = []
        issues = []
        
        for member in team_members:
            try:
                exp = float(member.get("metadata", {}).get("work_experience_years", 0))
                experiences.append(exp)
                
                name = member.get("name", "Unknown")
                if min_exp and exp < min_exp:
                    issues.append(f"{name} has {exp}y exp (min: {min_exp})")
                if max_exp and exp > max_exp:
                    issues.append(f"{name} has {exp}y exp (max: {max_exp})")
                    
            except (ValueError, TypeError):
                continue
        
        avg_exp = sum(experiences) / len(experiences) if experiences else 0
        is_valid = len(issues) == 0
        
        return is_valid, avg_exp, issues
    
    def _check_availability(
        self,
        team_members: List[Dict[str, Any]],
        min_hours: int
    ) -> List[str]:
        """Check availability constraints."""
        issues = []
        
        for member in team_members:
            availability = member.get("constraints", {}).get("weekly_availability_hours", "40")
            
            try:
                # Handle range format like "20-40"
                if "-" in str(availability):
                    hours = int(str(availability).split("-")[0])
                else:
                    hours = int(availability)
                
                if hours < min_hours:
                    name = member.get("name", "Unknown")
                    issues.append(f"{name} ({hours}h/week)")
                    
            except (ValueError, TypeError):
                continue
        
        return issues
    
    def _check_duplicates(
        self,
        team_members: List[Dict[str, Any]]
    ) -> List[str]:
        """Check for duplicate team members."""
        seen_ids = set()
        duplicates = []
        
        for member in team_members:
            member_id = member.get("id")
            if member_id in seen_ids:
                duplicates.append(member.get("name", member_id))
            seen_ids.add(member_id)
        
        return duplicates
    
    def suggest_improvements(
        self,
        validation_result: ValidationResult,
        available_candidates: List[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest improvements based on validation result.
        
        Args:
            validation_result: Result from validate()
            available_candidates: Optional pool of additional candidates
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for violation in validation_result.violations:
            if violation.constraint_type == "skill_coverage":
                missing = violation.details.get("missing", [])
                if missing:
                    suggestions.append(
                        f"Consider adding members with skills: {', '.join(missing[:3])}"
                    )
            
            elif violation.constraint_type == "role_coverage":
                unmatched = violation.details.get("unmatched", [])
                if unmatched:
                    suggestions.append(
                        f"Team may need: {', '.join(unmatched)}"
                    )
            
            elif violation.constraint_type == "experience":
                suggestions.append(
                    "Consider adjusting team seniority balance"
                )
        
        return suggestions


# Singleton instance
constraint_validator = TeamConstraintValidator()
