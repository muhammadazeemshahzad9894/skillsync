# 🧩 SkillSync AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

**AI-Powered Team Formation System using LLM + RAG Pipeline**

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Architecture](#architecture) • [Evaluation](#evaluation)

</div>

---

## 📋 Overview

SkillSync AI automatically forms **balanced, diverse, and complementary teams** by analyzing skills, experience, and preferences using Generative AI. Instead of relying on chance or incomplete information, the system uses **Large Language Models** and **semantic search** to create groups that are more effective, fair, and aligned with project requirements.

### 🎯 Problem Statement

Modern team formation suffers from:
- **Fragmented information** scattered across documents and platforms
- **Incomplete soft-skill and availability data**
- **Subjective, bias-prone decisions** based on familiarity
- **No scalable mechanism** to infer patterns and complementarity

### 💡 Solution

SkillSync addresses these challenges by:
1. **Extracting structured requirements** from natural language project descriptions
2. **Matching candidates** using semantic similarity with sentence transformers
3. **Forming teams** using multiple optimization strategies
4. **Validating constraints** (skills, roles, experience, availability)
5. **Generating explanations** for transparent, trustworthy recommendations

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Requirement Extraction** | Natural language → structured project requirements |
| 🔍 **Semantic Candidate Search** | Embedding-based similarity matching |
| 📊 **Multiple Formation Strategies** | Expert, Balanced, Diverse, Skill-Coverage teams |
| ✅ **Constraint Validation** | Skills, roles, experience, availability checks |
| 📄 **PDF Resume Parsing** | Upload CVs and auto-extract profiles |
| 📁 **CSV Bulk Import** | Mass import employee data |
| 📈 **Quality Evaluation** | Metrics + random baseline comparison |
| 🤖 **AI Explanations** | Human-readable team selection rationale |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│                    (Streamlit Web Application)                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       SKILLSYNC ENGINE                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   LLM       │  │  Embedding  │  │    Team     │  │ Evaluation  │ │
│  │ Extractor   │  │   Manager   │  │  Formation  │  │   Module    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
└─────────┼────────────────┼────────────────┼────────────────┼────────┘
          │                │                │                │
┌─────────▼────────────────▼────────────────▼────────────────▼────────┐
│                        DATA LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    PDF      │  │    CSV      │  │   Skill     │  │  Candidate  │ │
│  │   Parser    │  │   Parser    │  │ Normalizer  │  │   Store     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
User Input (Project Description)
        │
        ▼
┌───────────────────┐
│  LLM Extraction   │ ──► Structured Requirements (skills, roles, domain)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Semantic Search   │ ──► Candidate Pool (top-K by similarity)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Team Formation    │ ──► Multiple Strategy Teams
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Constraint Check  │ ──► Validation Results
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ LLM Explanation   │ ──► Human-Readable Analysis
└───────────────────┘
        │
        ▼
    Final Output
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- OpenRouter API key (or OpenAI API key)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/skillsync-ai.git
cd skillsync-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key

# Generate sample data
python -m src.data_generator --count 200

# Run the application
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "your-api-key"
   OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
   OPENAI_MODEL = "openai/gpt-4o-mini"
   ```
5. Deploy!

---

## 📖 Usage

### 1. Team Formation

```python
from src import SkillSyncEngine

# Initialize engine
engine = SkillSyncEngine()

# Form teams from project description
strategies, requirements = engine.form_teams(
    project_description="Build a fintech mobile app with React Native and Python backend",
    team_size=4
)

# Access results
for name, team in strategies.items():
    print(f"{team.strategy_name}: {[m['name'] for m in team.members]}")
    print(f"Analysis: {team.llm_analysis}")
```

### 2. Add Candidates from CSV

```python
# Add candidates from CSV
count = engine.add_candidates_from_csv("path/to/employees.csv")
print(f"Added {count} candidates")
```

### 3. Parse Resume (PDF)

```python
# Parse and add candidate from PDF resume
with open("resume.pdf", "rb") as f:
    profile = engine.add_candidates_from_pdf(f.read())
    print(f"Added: {profile['name']}")
```

### 4. Evaluate Team Quality

```python
# Get evaluation metrics
evaluation = engine.get_team_evaluation(
    team=strategies["Option A: The Expert Team"].members,
    required_skills=requirements.technical_keywords,
    compare_to_random=True
)

print(f"Overall Score: {evaluation['metrics']['overall_score']:.1%}")
print(f"Improvement over random: {evaluation['benchmark']['improvement_percentage']:.1f}%")
```

---

## 📊 Evaluation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Skill Coverage** | % of required skills covered by team | > 80% |
| **Role Diversity** | Uniqueness of team roles and Belbin types | > 60% |
| **Experience Balance** | Mix of senior and junior members | > 50% |
| **Match Score** | Average semantic similarity to requirements | > 70% |
| **Constraint Satisfaction** | All constraints met | 100% |

### Benchmark Results

Tested against random team assignment (50 trials):

| Strategy | Overall Score | vs Random |
|----------|---------------|-----------|
| Expert Team | 78.3% | +42.1% |
| Balanced Team | 75.6% | +37.2% |
| Diverse Team | 73.4% | +33.1% |

---

## 📁 Project Structure

```
SkillSync/
├── config/
│   └── settings.py           # Centralized configuration
├── data/
│   └── employees.json        # Employee database
├── src/
│   ├── __init__.py
│   ├── engine.py             # Main orchestration engine
│   ├── preprocessing/
│   │   ├── normalizer.py     # Skill normalization
│   │   ├── pdf_parser.py     # Resume/CV parsing
│   │   └── csv_parser.py     # CSV import
│   ├── extraction/
│   │   └── llm_extractor.py  # LLM-based extraction
│   ├── matching/
│   │   ├── embeddings.py     # Sentence embeddings
│   │   └── retrieval.py      # Candidate search
│   ├── team_formation/
│   │   ├── strategies.py     # Formation strategies
│   │   └── constraints.py    # Validation logic
│   ├── evaluation/
│   │   └── metrics.py        # Quality metrics
│   ├── utils.py              # Helper functions
│   └── data_generator.py     # Synthetic data
├── tests/
│   └── test_normalizer.py    # Unit tests
├── app.py                    # Streamlit application
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.9+ |
| **LLM** | GPT-4o-mini via OpenRouter |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Similarity** | Cosine Similarity (scikit-learn) |
| **PDF Parsing** | PyMuPDF |
| **Data** | Pandas, JSON |

---

## 👥 Team

**Group 45 - TU Wien Generative AI Course (194.207) 2025W**

- Shahzad Muhammad Azeem (12346021)
- Lasheen Nooreldin (12302427)
- Baranga Roxana Mary (12502784)
- Kormaku Ana (12534172)
- Şaban Akay (12045645)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TU Wien for the Generative AI course
- Anthropic for Claude AI assistance
- OpenRouter for affordable API access
- Sentence-Transformers team for embedding models

---

<div align="center">

**Built with ❤️ for the Generative AI Course at TU Wien**

[⬆ Back to Top](#-skillsync-ai)

</div>
