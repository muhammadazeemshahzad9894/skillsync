"""
SkillSync Configuration Management

Centralized configuration with support for:
- Local development (.env file)
- Streamlit Cloud (st.secrets)
"""

import os
from dataclasses import dataclass
from typing import Optional

# Try Streamlit secrets first (for cloud deployment)
try:
    import streamlit as st
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        os.environ['OPENAI_API_KEY'] = st.secrets.get('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
        os.environ['OPENAI_BASE_URL'] = st.secrets.get('OPENAI_BASE_URL', os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1'))
        os.environ['OPENAI_MODEL'] = st.secrets.get('OPENAI_MODEL', os.environ.get('OPENAI_MODEL', 'openai/gpt-4o-mini'))
except:
    pass

# Load .env for local development
from dotenv import load_dotenv
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration."""
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1500
    http_referer: str = "https://skillsync.app"
    app_title: str = "SkillSync"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    show_progress: bool = True


@dataclass
class TeamConfig:
    """Team formation configuration."""
    min_team_size: int = 2
    max_team_size: int = 10
    default_team_size: int = 4
    candidate_pool_multiplier: int = 5


@dataclass
class PathConfig:
    """File paths configuration."""
    data_dir: str = "data"
    employees_file: str = "employees.json"
    
    @property
    def employees_path(self) -> str:
        return os.path.join(self.data_dir, self.employees_file)


class Settings:
    """Main settings class."""
    
    def __init__(self):
        self._validate_environment()
        
        self.llm = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1500")),
        )
        
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        
        self.team = TeamConfig(
            min_team_size=int(os.getenv("MIN_TEAM_SIZE", "2")),
            max_team_size=int(os.getenv("MAX_TEAM_SIZE", "10")),
            default_team_size=int(os.getenv("DEFAULT_TEAM_SIZE", "4")),
        )
        
        self.paths = PathConfig(
            data_dir=os.getenv("DATA_DIR", "data"),
            employees_file=os.getenv("EMPLOYEES_FILE", "employees.json"),
        )
    
    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "Missing OPENAI_API_KEY. Please set it in .env file or Streamlit secrets."
            )
    
    @property
    def llm_headers(self) -> dict:
        """Headers for OpenRouter API calls."""
        return {
            "HTTP-Referer": self.llm.http_referer,
            "X-Title": self.llm.app_title,
        }


# Singleton
settings = Settings()
