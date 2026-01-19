"""
SkillSync AI - Streamlit Application

Main entry point for the SkillSync team formation web application.
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any

# Must be first Streamlit command
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import our modules
from src import SkillSyncEngine
from src.preprocessing import csv_parser
from src.utils import format_experience, format_skills_list, safe_float


# --- CACHED RESOURCES ---
@st.cache_resource
def load_engine():
    """Load and cache the SkillSync engine."""
    try:
        return SkillSyncEngine()
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        return None


# --- HELPER FUNCTIONS ---
def display_metric_card(col, label: str, value: float, format_str: str = "{:.0%}"):
    """Display a metric in a styled card."""
    formatted = format_str.format(value) if isinstance(value, (int, float)) else str(value)
    col.metric(label, formatted)


def display_team_member(member: Dict[str, Any], show_details: bool = True):
    """Display a team member card."""
    name = member.get("name", "Unknown")
    role = member.get("role", "Unknown Role")
    match_score = member.get("match_score", 0)
    exp = member.get("metadata", {}).get("work_experience_years", 0)
    skills = member.get("technical", {}).get("skills", [])
    belbin = member.get("personality", {}).get("Belbin_team_role", "Unknown")
    
    with st.expander(f"**{name}** - {role} (Match: {match_score:.0%})"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Experience:** {format_experience(exp)}")
            st.write(f"**Team Role:** {belbin}")
        
        with col2:
            industry = member.get("metadata", {}).get("industry", "General")
            st.write(f"**Industry:** {industry}")
            comm_style = member.get("collaboration", {}).get("communication_style", "Mixed")
            st.write(f"**Communication:** {comm_style}")
        
        st.write(f"**Skills:** {format_skills_list(skills, 8)}")
        
        tools = member.get("technical", {}).get("tools", [])
        if tools:
            st.write(f"**Tools:** {format_skills_list(tools, 6)}")


def display_evaluation_metrics(metrics: Dict[str, Any]):
    """Display evaluation metrics in a grid."""
    cols = st.columns(4)
    
    display_metric_card(cols[0], "Skill Coverage", metrics.get("skill_coverage", 0))
    display_metric_card(cols[1], "Role Diversity", metrics.get("role_diversity", 0))
    display_metric_card(cols[2], "Match Score", metrics.get("avg_match_score", 0))
    display_metric_card(cols[3], "Overall", metrics.get("overall_score", 0))


# --- MAIN APP ---
def main():
    # Load engine
    engine = load_engine()
    
    if engine is None:
        st.error("🚨 Failed to initialize SkillSync Engine. Please check your configuration.")
        st.info("Make sure you have:")
        st.code("""
1. Created a .env file with:
   OPENAI_API_KEY=your_api_key
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=openai/gpt-4o-mini

2. Generated sample data:
   python -m src.data_generator
        """)
        st.stop()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🧩 SkillSync")
        st.caption("AI-Powered Team Formation")
        st.markdown("---")
        
        st.metric("Candidates in Pool", engine.candidate_count)
        
        if st.button("🔄 Reload Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        SkillSync uses AI to form optimal teams by:
        - Extracting requirements from descriptions
        - Matching candidates using semantic search
        - Balancing skills, experience & roles
        - Validating against constraints
        """)
        
        st.markdown("---")
        st.caption("TU Wien - Generative AI Project")
        st.caption("Group 45 | 2025W")
    
    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs([
        "🚀 Team Builder", 
        "👥 Talent Pool",
        "📊 Evaluation"
    ])
    
    # ==========================================
    # TAB 1: TEAM BUILDER
    # ==========================================
    with tab1:
        st.header("Project Team Assembly")
        st.markdown("Describe your project needs, and AI will form optimized teams using multiple strategies.")
        
        # Input Section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            project_desc = st.text_area(
                "Project Requirements",
                height=150,
                placeholder="Example: We need a team to build a Fintech mobile app. Requires React Native, Python backend, AWS deployment, and strong security knowledge. Need mix of senior and mid-level developers."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            team_size = st.number_input(
                "Team Size", 
                min_value=2, 
                max_value=10, 
                value=4
            )
            
            include_eval = st.checkbox("Include Evaluation", value=True)
            generate_btn = st.button("✨ Generate Teams", type="primary", use_container_width=True)
        
        # Generate Teams
        if generate_btn and project_desc:
            with st.spinner("🤖 AI is analyzing requirements and forming teams..."):
                results = engine.form_teams(
                    project_desc, 
                    team_size,
                    include_evaluation=include_eval
                )
                
                # Check for error
                if isinstance(results, tuple) and len(results) == 2:
                    strategies, requirements = results
                    
                    if isinstance(strategies, dict) and "error" in strategies:
                        st.error(strategies["error"])
                    else:
                        # Save to session state
                        st.session_state['team_results'] = strategies
                        st.session_state['requirements'] = requirements
                        st.session_state['project_desc'] = project_desc
                else:
                    st.error("Unexpected result format")
        
        # Display Results
        if 'team_results' in st.session_state:
            strategies = st.session_state['team_results']
            requirements = st.session_state['requirements']
            
            # Show extracted requirements
            with st.expander("🧠 AI Understanding (Extracted Requirements)", expanded=True):
                req_cols = st.columns(4)
                req_cols[0].write(f"**Domain:** {requirements.domain}")
                req_cols[1].write(f"**Seniority:** {requirements.seniority_level}")
                req_cols[2].write(f"**Roles:** {', '.join(requirements.target_roles[:3]) if requirements.target_roles else 'General'}")
                req_cols[3].write(f"**Skills:** {len(requirements.technical_keywords)} identified")
                
                st.info(f"**Summary:** {requirements.summary}")
                
                if requirements.technical_keywords:
                    st.write(f"**Technical Keywords:** {', '.join(requirements.technical_keywords[:10])}")
            
            st.success("✅ Analysis Complete! Here are your team options:")
            st.markdown("---")
            
            # Display teams in columns
            team_cols = st.columns(3)
            icons = ["🏆", "⚖️", "🎨"]
            
            for idx, (name, result) in enumerate(strategies.items()):
                with team_cols[idx]:
                    st.subheader(f"{icons[idx]} {result.strategy_name}")
                    
                    # AI Analysis
                    with st.container(border=True):
                        st.markdown("### 🤖 AI Analysis")
                        st.write(result.llm_analysis or "Analysis generating...")
                    
                    # Validation Status
                    validation = result.metadata.get("validation", {})
                    if validation.get("is_valid", True):
                        st.success(f"✓ Constraints satisfied ({validation.get('coverage_score', 1):.0%} skill coverage)")
                    else:
                        st.warning("⚠️ Some constraints not fully met")
                        for warning in validation.get("warnings", [])[:2]:
                            st.caption(f"  • {warning}")
                    
                    # Evaluation Metrics
                    if include_eval and "evaluation" in result.metadata:
                        with st.expander("📊 Quality Metrics"):
                            metrics = result.metadata["evaluation"]
                            st.write(f"**Overall Score:** {metrics.get('overall_score', 0):.0%}")
                            st.write(f"**Skill Coverage:** {metrics.get('skill_coverage', 0):.0%}")
                            st.write(f"**Diversity:** {metrics.get('role_diversity', 0):.0%}")
                    
                    # Team Members
                    st.markdown("### Team Members")
                    for member in result.members:
                        display_team_member(member)
    
    # ==========================================
    # TAB 2: TALENT POOL
    # ==========================================
    with tab2:
        st.header("Talent Pool Management")
        
        # Upload Section
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            with st.expander("📤 Upload CSV", expanded=True):
                st.markdown("Upload a CSV with columns: `Name`, `Role`, `Experience`, `Skills`")
                
                uploaded_csv = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
                
                if uploaded_csv is not None:
                    # Preview
                    try:
                        preview_df = pd.read_csv(uploaded_csv)
                        st.write(f"Preview ({len(preview_df)} rows):")
                        st.dataframe(preview_df.head(5), use_container_width=True)
                        
                        # Reset file position for actual upload
                        uploaded_csv.seek(0)
                        
                        if st.button("➕ Add to Database", key="add_csv"):
                            try:
                                count = engine.add_candidates_from_csv(uploaded_csv)
                                st.success(f"✅ Added {count} candidates!")
                                st.cache_resource.clear()
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
        
        with upload_col2:
            with st.expander("📄 Upload Resume (PDF)", expanded=True):
                st.markdown("Upload a PDF resume to extract candidate profile")
                
                uploaded_pdf = st.file_uploader("Choose PDF file", type=["pdf"], key="pdf_upload")
                
                if uploaded_pdf is not None:
                    if st.button("🔍 Parse Resume", key="parse_pdf"):
                        with st.spinner("Extracting profile from resume..."):
                            try:
                                profile = engine.add_candidates_from_pdf(uploaded_pdf.read())
                                st.success(f"✅ Added: {profile.get('name', 'Unknown')}")
                                st.json(profile)
                                st.cache_resource.clear()
                            except Exception as e:
                                st.error(f"Error parsing PDF: {e}")
        
        st.markdown("---")
        
        # Search and Display
        st.subheader("Current Database")
        
        search_query = st.text_input("🔍 Search candidates (by name, skill, or role)")
        
        # Build display dataframe
        if engine.raw_data:
            flat_data = []
            for p in engine.raw_data:
                flat_data.append({
                    "Name": p.get("name", "N/A"),
                    "Role": p.get("role", "N/A"),
                    "Experience": format_experience(p.get("metadata", {}).get("work_experience_years", 0)),
                    "Industry": p.get("metadata", {}).get("industry", "General"),
                    "Skills": format_skills_list(p.get("technical", {}).get("skills", []), 4),
                    "Belbin Role": p.get("personality", {}).get("Belbin_team_role", "Unknown")
                })
            
            df = pd.DataFrame(flat_data)
            
            # Filter if search query
            if search_query:
                mask = df.apply(
                    lambda x: x.astype(str).str.contains(search_query, case=False).any(), 
                    axis=1
                )
                df = df[mask]
            
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"Showing {len(df)} of {len(engine.raw_data)} candidates")
        else:
            st.warning("No candidates in database. Upload CSV or PDF to add candidates.")
            
            # Sample CSV download
            st.download_button(
                "📥 Download Sample CSV Template",
                data=csv_parser.generate_sample_csv(),
                file_name="sample_candidates.csv",
                mime="text/csv"
            )
    
    # ==========================================
    # TAB 3: EVALUATION
    # ==========================================
    with tab3:
        st.header("System Evaluation")
        st.markdown("Evaluate team formation quality against random baseline.")
        
        if 'team_results' not in st.session_state:
            st.info("👆 Generate teams in the Team Builder tab first to see evaluation results.")
        else:
            strategies = st.session_state['team_results']
            requirements = st.session_state['requirements']
            
            st.subheader("Team Quality Comparison")
            
            # Create comparison table
            comparison_data = []
            for name, result in strategies.items():
                metrics = result.metadata.get("evaluation", {})
                comparison_data.append({
                    "Strategy": result.strategy_name,
                    "Overall Score": f"{metrics.get('overall_score', 0):.1%}",
                    "Skill Coverage": f"{metrics.get('skill_coverage', 0):.1%}",
                    "Role Diversity": f"{metrics.get('role_diversity', 0):.1%}",
                    "Match Score": f"{metrics.get('avg_match_score', 0):.1%}",
                    "Exp Balance": f"{metrics.get('experience_balance', 0):.1%}",
                })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Benchmark against random
            st.subheader("Benchmark Against Random")
            
            if st.button("🎲 Run Random Baseline Comparison"):
                with st.spinner("Running 50 random trials..."):
                    # Get first team for comparison
                    first_team = list(strategies.values())[0]
                    
                    eval_result = engine.get_team_evaluation(
                        team=first_team.members,
                        required_skills=requirements.technical_keywords,
                        compare_to_random=True
                    )
                    
                    if "benchmark" in eval_result:
                        benchmark = eval_result["benchmark"]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(
                            "System Score",
                            f"{benchmark['system']['overall_score']:.1%}"
                        )
                        col2.metric(
                            "Random Baseline",
                            f"{benchmark['random_baseline']['overall_score']:.1%}"
                        )
                        col3.metric(
                            "Improvement",
                            f"{benchmark['improvement_percentage']:.1f}%",
                            delta=f"+{benchmark['improvement_percentage']:.1f}%"
                        )
                        
                        st.success(f"✅ System outperforms random by {benchmark['improvement_percentage']:.1f}% (50 trials)")
                    else:
                        st.warning("Could not run benchmark - not enough candidates")
            
            # Latency metrics
            st.subheader("Pipeline Performance")
            latency_report = engine.latency_tracker.get_report()
            
            latency_df = pd.DataFrame([{
                "Stage": "Requirement Extraction",
                "Time (ms)": latency_report.extraction_ms
            }, {
                "Stage": "Candidate Retrieval",
                "Time (ms)": latency_report.retrieval_ms
            }, {
                "Stage": "Team Formation",
                "Time (ms)": latency_report.team_formation_ms
            }, {
                "Stage": "Explanation Generation",
                "Time (ms)": latency_report.explanation_ms
            }, {
                "Stage": "Total",
                "Time (ms)": latency_report.total_ms
            }])
            
            st.dataframe(latency_df, use_container_width=True)


if __name__ == "__main__":
    main()
