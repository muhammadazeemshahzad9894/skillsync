"""
Synthetic Data Generator

Generates realistic employee profiles for testing and demonstration.
Run with: python -m src.data_generator
"""

import json
import random
import os
import sys
from typing import List, Dict, Any

# Handle imports whether run as module or script
try:
    from faker import Faker
except ImportError:
    print("Installing faker...")
    os.system(f"{sys.executable} -m pip install faker")
    from faker import Faker

# Initialize Faker
fake = Faker()

# --- CONSTANTS ---
DATA_DIR = "data"
DATA_FILE = "employees.json"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Developer Types
DEV_TYPES = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "DevOps specialist",
    "Data scientist or machine learning specialist",
    "Cloud infrastructure engineer",
    "System administrator",
]

# Employment Types
EMPLOYMENT_TYPES = [
    "Employed, full-time",
    "Independent contractor, freelancer, or self-employed",
    "Employed, part-time",
]

# Organization Sizes
ORG_SIZES = [
    "2 to 9 employees",
    "10 to 19 employees",
    "20 to 99 employees",
    "100 to 499 employees",
    "500 to 999 employees",
    "1,000 to 4,999 employees",
    "5,000 to 9,999 employees",
    "10,000 or more employees",
]

# Industries
INDUSTRIES = [
    "Healthcare",
    "Fintech",
    "E-commerce",
    "Education",
    "Consulting",
    "Cybersecurity",
    "Manufacturing",
    "Media",
    "Telecommunications",
    "Gaming",
    "Travel",
    "Real Estate",
]

# Technical Skills Pool
LANGUAGES_AND_FRAMEWORKS = [
    "Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "Go", "Rust",
    "PHP", "Ruby", "Swift", "Kotlin", "Dart", "Scala", "R",
    "React", "Angular", "Vue.js", "Next.js", "Node.js", "Django", "Flask",
    "Spring Boot", "ASP.NET Core", "FastAPI", "Express", "NestJS", "Flutter",
    "React Native", "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
    "Laravel", "Rails", "GraphQL", "REST API", "gRPC",
]

# Tools and Platforms
TOOLS_AND_PLATFORMS = [
    "Docker", "Kubernetes", "AWS", "Google Cloud", "Azure", "Git", "GitHub",
    "GitLab", "Jira", "Trello", "Slack", "VS Code", "IntelliJ IDEA",
    "PyCharm", "Android Studio", "Xcode", "Postman", "Terraform", "Ansible",
    "Jenkins", "CircleCI", "GitHub Actions", "Figma", "Webpack", "Vite",
    "MongoDB", "PostgreSQL", "MySQL", "Redis", "Elasticsearch", "Kafka",
    "RabbitMQ", "Nginx", "Linux", "Datadog", "Grafana", "Prometheus",
]

# Belbin Team Roles
BELBIN_ROLES = [
    "Plant",                    # Creative innovator
    "Resource Investigator",    # Explores opportunities
    "Co-ordinator",            # Clarifies goals, delegates
    "Shaper",                  # Drives team forward
    "Monitor Evaluator",       # Analyzes options
    "Teamworker",              # Helps team gel
    "Implementer",             # Turns ideas into action
    "Completer Finisher",      # Ensures thorough completion
    "Specialist",              # Deep knowledge in specific area
]

# Collaboration Attributes
COMMUNICATION_STYLES = ["Direct", "Indirect", "Mixed", "Formal", "Informal"]
CONFLICT_STYLES = ["Avoid", "Collaborate", "Compete", "Accommodate", "Compromise"]
LEADERSHIP_PREFERENCES = ["Lead", "Follow", "Co-lead", "Autonomous"]
DEADLINE_DISCIPLINES = ["Strict", "Flexible", "Pressure-dependent"]
LEARNING_ORIENTATIONS = ["Low", "Medium", "High", "Very High"]
KNOWLEDGE_SHARING_LEVELS = ["Low", "Medium", "High"]


def generate_profile(profile_id: int) -> Dict[str, Any]:
    """
    Generate a single synthetic employee profile.
    
    Args:
        profile_id: Unique identifier for the profile
        
    Returns:
        Complete profile dictionary
    """
    # Generate consistent experience values
    work_exp = round(random.uniform(0.5, 20), 1)
    years_code = min(work_exp + random.randint(0, 5), 25)
    
    # Select appropriate role based on experience
    if work_exp < 2:
        role_weights = [0.3, 0.2, 0.2, 0.15, 0.05, 0.05, 0.03, 0.02]
    elif work_exp < 5:
        role_weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]
    else:
        role_weights = [0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1]
    
    dev_type = random.choices(DEV_TYPES, weights=role_weights)[0]
    
    # Generate skills based on role
    num_skills = random.randint(4, 12)
    num_tools = random.randint(3, 8)
    
    # Role-specific skill biases
    role_skill_bias = {
        "Developer, front-end": ["JavaScript", "TypeScript", "React", "Vue.js", "Angular", "Next.js"],
        "Developer, back-end": ["Python", "Java", "Node.js", "Go", "Django", "Spring Boot"],
        "Developer, full-stack": ["JavaScript", "Python", "React", "Node.js", "TypeScript"],
        "Developer, mobile": ["Swift", "Kotlin", "React Native", "Flutter", "Dart"],
        "DevOps specialist": ["Docker", "Kubernetes", "Terraform", "AWS", "Jenkins"],
        "Data scientist or machine learning specialist": ["Python", "TensorFlow", "PyTorch", "Pandas", "R"],
        "Cloud infrastructure engineer": ["AWS", "Azure", "Google Cloud", "Terraform", "Kubernetes"],
        "System administrator": ["Linux", "Docker", "Ansible", "Nginx", "Python"],
    }
    
    # Get biased skills + random skills
    biased_skills = role_skill_bias.get(dev_type, [])
    selected_bias = random.sample(biased_skills, min(3, len(biased_skills)))
    remaining_skills = [s for s in LANGUAGES_AND_FRAMEWORKS if s not in selected_bias]
    additional_skills = random.sample(remaining_skills, min(num_skills - len(selected_bias), len(remaining_skills)))
    skills = selected_bias + additional_skills
    
    # Tools
    tools = random.sample(TOOLS_AND_PLATFORMS, min(num_tools, len(TOOLS_AND_PLATFORMS)))
    
    # Availability based on employment type
    employment = random.choice(EMPLOYMENT_TYPES)
    if "full-time" in employment:
        availability = random.choice(["35-40", "40", "40-50"])
    elif "part-time" in employment:
        availability = random.choice(["10-20", "20-30"])
    else:
        availability = random.choice(["20-40", "30-50", "40"])
    
    return {
        "id": str(profile_id),
        "name": fake.name(),
        "role": dev_type,
        "constraints": {
            "weekly_availability_hours": availability,
        },
        "metadata": {
            "dev_type": dev_type,
            "work_experience_years": str(work_exp),
            "years_code": str(years_code),
            "employment": employment,
            "org_size": random.choice(ORG_SIZES),
            "industry": random.choice(INDUSTRIES),
        },
        "technical": {
            "skills": skills,
            "tools": tools,
        },
        "personality": {
            "Belbin_team_role": random.choice(BELBIN_ROLES),
        },
        "collaboration": {
            "communication_style": random.choice(COMMUNICATION_STYLES),
            "conflict_style": random.choice(CONFLICT_STYLES),
            "leadership_preference": random.choice(LEADERSHIP_PREFERENCES),
            "deadline_discipline": random.choice(DEADLINE_DISCIPLINES),
        },
        "learning_behavior": {
            "learning_orientation": random.choice(LEARNING_ORIENTATIONS),
            "knowledge_sharing": random.choice(KNOWLEDGE_SHARING_LEVELS),
        },
    }


def generate_dataset(count: int = 200) -> List[Dict[str, Any]]:
    """
    Generate a complete dataset of synthetic profiles.
    
    Args:
        count: Number of profiles to generate
        
    Returns:
        List of profile dictionaries
    """
    print(f"Generating {count} synthetic profiles...")
    profiles = [generate_profile(i + 1) for i in range(count)]
    print(f"✅ Generated {len(profiles)} profiles")
    return profiles


def save_dataset(profiles: List[Dict[str, Any]], filepath: str = DATA_PATH) -> None:
    """
    Save dataset to JSON file.
    
    Args:
        profiles: List of profile dictionaries
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(profiles)} profiles to {filepath}")


def main():
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic employee data")
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=200,
        help="Number of profiles to generate (default: 200)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DATA_PATH,
        help=f"Output file path (default: {DATA_PATH})"
    )
    
    args = parser.parse_args()
    
    # Generate and save
    profiles = generate_dataset(args.count)
    save_dataset(profiles, args.output)
    
    # Print sample
    print("\n📋 Sample profile:")
    sample = profiles[0]
    print(f"  Name: {sample['name']}")
    print(f"  Role: {sample['role']}")
    print(f"  Experience: {sample['metadata']['work_experience_years']} years")
    print(f"  Skills: {', '.join(sample['technical']['skills'][:5])}...")
    print(f"  Belbin Role: {sample['personality']['Belbin_team_role']}")


if __name__ == "__main__":
    main()
