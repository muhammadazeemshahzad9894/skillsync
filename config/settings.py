"""
SkillSync Configuration Management

Centralized configuration with environment variable loading and validation.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM (OpenRouter/OpenAI) connection."""
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # OpenRouter specific headers
    http_referer: str = "https://skillsync.app"
    app_title: str = "SkillSync"


@dataclass  
class EmbeddingConfig:
    """Configuration for sentence transformer embeddings."""
    model_name: str = "all-MiniLM-L6-v2"
    show_progress: bool = True


@dataclass
class TeamFormationConfig:
    """Configuration for team formation parameters."""
    min_team_size: int = 2
    max_team_size: int = 10
    default_team_size: int = 4
    candidate_pool_multiplier: int = 5  # Pool = team_size * multiplier


@dataclass
class PathConfig:
    """File paths configuration."""
    data_dir: str = "data"
    employees_file: str = "employees.json"
    
    @property
    def employees_path(self) -> str:
        return os.path.join(self.data_dir, self.employees_file)


class Settings:
    """
    Main settings class that aggregates all configurations.
    
    Usage:
        from config.settings import settings
        print(settings.llm.model)
    """
    
    def __init__(self):
        self._validate_environment()
        
        self.llm = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        )
        
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        
        self.team = TeamFormationConfig(
            min_team_size=int(os.getenv("MIN_TEAM_SIZE", "2")),
            max_team_size=int(os.getenv("MAX_TEAM_SIZE", "10")),
            default_team_size=int(os.getenv("DEFAULT_TEAM_SIZE", "4")),
        )
        
        self.paths = PathConfig(
            data_dir=os.getenv("DATA_DIR", "data"),
            employees_file=os.getenv("EMPLOYEES_FILE", "employees.json"),
        )
    
    def _validate_environment(self) -> None:
        """Validate required environment variables exist."""
        required_vars = ["OPENAI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )
    
    @property
    def llm_headers(self) -> dict:
        """Return headers for OpenRouter API calls."""
        return {
            "HTTP-Referer": self.llm.http_referer,
            "X-Title": self.llm.app_title,
        }


# Singleton instance
settings = Settings()
