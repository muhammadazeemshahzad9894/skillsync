# 🧩 SkillSync AI v2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)

**AI-Powered Team Formation System with Chain-of-Prompts Extraction**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Evaluation](#-evaluation)

</div>

---

## 🆕 What's New in v2

| Feature | Description |
|---------|-------------|
| 📊 **Dashboard** | Quick stats, role distribution, one-click actions |
| 🔗 **Chain Extraction** | Extract → Validate → Enhance pipeline |
| 📁 **StackOverflow CSV** | Full 28-column support with auto-detection |
| ⏰ **Availability Filter** | Exclude candidates below minimum hours |
| 📈 **Inline Evaluation** | Icons (✅/⚠️/❌) + detailed metrics |
| 🧪 **Test Set** | 10 built-in test cases for extraction evaluation |
| 👥 **Teams First** | See team members before explanations |

---

## ✨ Features

### 🤖 Chain-of-Prompts Extraction
Instead of a single LLM call, we use a 3-stage pipeline:

1. **Extract** - Initial extraction from project description
2. **Validate** - Remove hallucinations, enforce role whitelist
3. **Enhance** - Normalize terminology, add implicit skills

### 📊 Dashboard
- Quick stats (candidates, roles, experience, industries)
- Role and industry distribution charts
- One-click navigation to Team Builder

### 👥 Team Builder (Improved)
- Teams displayed **FIRST**, explanations below
- Inline quality metrics with status icons
- Availability filtering option

### 📁 Multi-Format CSV Support
- **Simple format**: Name, Role, Experience, Skills
- **StackOverflow format**: All 28 columns including:
  - Multiple skill columns (languages, databases, platforms, etc.)
  - PersonalityText → Belbin role extraction
  - WeeklyAvailabilityHours parsing

### 📈 Evaluation Framework
- **Extraction metrics**: Precision, Recall, F1, Domain accuracy
- **Team metrics**: Skill coverage, Role diversity, Experience balance
- **Benchmark**: Comparison against random baseline (50 trials)
- **Latency tracking**: Per-stage timing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                               │
│     Dashboard │ Team Builder │ Talent Pool                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    SKILLSYNC ENGINE v2                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  Chained    │ │  Embedding  │ │    Team     │ │Evaluation │ │
│  │  Extractor  │ │   Manager   │ │  Formation  │ │  Module   │ │
│  │ (3 stages)  │ │             │ │             │ │           │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬─────┘ │
└─────────┼───────────────┼───────────────┼──────────────┼───────┘
          │               │               │              │
┌─────────▼───────────────▼───────────────▼──────────────▼───────┐
│                      DATA LAYER                                 │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ StackOverflow    │  │    Skill       │  │   Candidate    │  │
│  │  CSV Parser      │  │  Normalizer    │  │    Store       │  │
│  │ (28 columns)     │  │  (70+ maps)    │  │   (JSON)       │  │
│  └──────────────────┘  └────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Extraction Pipeline

```
Project Description
        │
        ▼
┌───────────────────┐
│   STAGE 1: EXTRACT │ → Initial extraction (temp=0.0)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  STAGE 2: VALIDATE │ → Remove hallucinations, enforce role whitelist
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  STAGE 3: ENHANCE  │ → Normalize terms, add implicit skills
└───────────────────┘
        │
        ▼
  Structured Requirements
```

---

## 🚀 Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/skillsync-ai.git
cd skillsync-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key

# Run application
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add secrets:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "openai/gpt-4o-mini"
```

---

## 📊 Evaluation

### Built-in Test Set (10 cases)

| Test | Domain | Key Technologies |
|------|--------|------------------|
| 001 | Fintech | React Native, Python, AWS |
| 002 | Healthcare | TensorFlow, PyTorch, Jupyter |
| 003 | E-commerce | React, Node.js, Kubernetes |
| 004 | Agriculture | C++, Rust, MQTT, IoT |
| 005 | Education | TypeScript, Next.js, MongoDB |
| 006 | Cybersecurity | Python, OWASP, GitHub Actions |
| 007 | Manufacturing | Kafka, Spark, Grafana |
| 008 | Gaming | Unity, C#, Photon |
| 009 | AI/ML | GPT-4, LangChain, React |
| 010 | Cloud | Terraform, Prometheus, AWS/Azure |

### Quality Metrics

| Metric | Description | Icon Thresholds |
|--------|-------------|-----------------|
| Skill Coverage | % required skills covered | ✅ ≥80%, ⚠️ ≥50%, ❌ <50% |
| Role Diversity | Unique roles / team size | ✅ ≥80%, ⚠️ ≥50%, ❌ <50% |
| Experience Balance | Mix of senior/junior | ✅ ≥80%, ⚠️ ≥50%, ❌ <50% |
| Match Score | Semantic similarity avg | ✅ ≥80%, ⚠️ ≥50%, ❌ <50% |

---

## 📁 Project Structure

```
skillsync_v2/
├── config/
│   └── settings.py           # Centralized configuration
├── data/
│   ├── employees.json        # Candidate database
│   └── employees.csv         # StackOverflow source
├── src/
│   ├── __init__.py
│   ├── engine.py             # Main orchestration
│   ├── preprocessing/
│   │   ├── normalizer.py     # Skill normalization (70+ mappings)
│   │   ├── csv_parser.py     # StackOverflow + simple CSV
│   │   └── pdf_parser.py     # Resume parsing
│   ├── extraction/
│   │   └── llm_extractor.py  # Chain-of-prompts extraction
│   ├── matching/
│   │   ├── embeddings.py     # Sentence embeddings
│   │   └── retrieval.py      # Semantic search
│   ├── team_formation/
│   │   ├── strategies.py     # 4 formation strategies
│   │   └── constraints.py    # Validation
│   ├── evaluation/
│   │   └── metrics.py        # Evaluation + test set
│   └── utils.py
├── app.py                    # Streamlit application
├── requirements.txt
├── README.md
└── .env.example
```

---

## 👥 Team

**Group 45 - TU Wien Generative AI Course (194.207) 2025W**

- Shahzad Muhammad Azeem
- Lasheen Nooreldin
- Baranga Roxana Mary
- Kormaku Ana
- Şaban Akay

---

## 📄 License

MIT License

---

<div align="center">

**Built with ❤️ for the Generative AI Course at TU Wien**

</div>
