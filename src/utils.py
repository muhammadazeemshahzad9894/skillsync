"""
Utility Functions

Shared helper functions for data loading, saving, and common operations.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of data dictionaries, empty list if file doesn't exist
    """
    if not os.path.exists(filepath):
        logger.warning(f"Data file not found: {filepath}")
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {filepath}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []


def save_data(filepath: str, data: List[Dict[str, Any]]) -> bool:
    """
    Save data to JSON file.
    
    Args:
        filepath: Path to save to
        data: Data to save
        
    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} records to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")
        return False


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default: Default if conversion fails
        
    Returns:
        Integer value or default
    """
    if value is None:
        return default
    
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text or ""
    
    return text[:max_length - len(suffix)] + suffix


def format_experience(years: Any) -> str:
    """
    Format experience years for display.
    
    Args:
        years: Years of experience (string or number)
        
    Returns:
        Formatted string like "5 years" or "1.5 years"
    """
    value = safe_float(years)
    
    if value == 0:
        return "No experience"
    elif value == 1:
        return "1 year"
    elif value == int(value):
        return f"{int(value)} years"
    else:
        return f"{value:.1f} years"


def format_skills_list(skills: List[str], max_display: int = 5) -> str:
    """
    Format skills list for display.
    
    Args:
        skills: List of skill names
        max_display: Maximum skills to show
        
    Returns:
        Formatted string
    """
    if not skills:
        return "No skills listed"
    
    if len(skills) <= max_display:
        return ", ".join(skills)
    
    displayed = skills[:max_display]
    remaining = len(skills) - max_display
    
    return f"{', '.join(displayed)}, +{remaining} more"


def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate percentage safely.
    
    Args:
        part: Numerator
        whole: Denominator
        
    Returns:
        Percentage (0-100)
    """
    if whole == 0:
        return 0.0
    
    return (part / whole) * 100


def deduplicate_list(items: List[Any]) -> List[Any]:
    """
    Remove duplicates while preserving order.
    
    Args:
        items: List with potential duplicates
        
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    
    for item in items:
        # For dicts, use string representation
        key = str(item) if isinstance(item, dict) else item
        
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result
