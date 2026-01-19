"""
Team Formation Module

Handles team assembly using multiple strategies with constraint validation.
"""

from .strategies import (
    TeamFormationEngine,
    TeamFormationResult,
    BaseStrategy,
    ExpertStrategy,
    BalancedStrategy,
    DiverseStrategy,
    SkillCoverageStrategy,
    team_formation_engine
)
from .constraints import (
    TeamConstraintValidator,
    ConstraintViolation,
    ValidationResult,
    constraint_validator
)

__all__ = [
    "TeamFormationEngine",
    "TeamFormationResult",
    "BaseStrategy",
    "ExpertStrategy",
    "BalancedStrategy",
    "DiverseStrategy",
    "SkillCoverageStrategy",
    "team_formation_engine",
    "TeamConstraintValidator",
    "ConstraintViolation",
    "ValidationResult",
    "constraint_validator",
]
