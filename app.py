"""
SkillSync AI - Enhanced Streamlit Application

Features:
- Dashboard with quick stats
- Team Builder with teams displayed first
- Inline evaluation with icons
- Enhanced Talent Pool for 28-column CSV
- Non-tech friendly explanations
"""

import streamlit as st
import pandas as pd
import time
import json
from typing import Dict, Any, List
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .team-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .score-excellent { color: #28a745; }
    .score-good { color: #ffc107; }
    .score-poor { color: #dc3545; }
    .status-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    .badge-danger { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# IMPORTS (after st.set_page_config)
# ============================================================================

from src import SkillSyncEngine
from src.preprocessing.csv_parser import (
    StackOverflowCSVParser, 
    detect_csv_format, 
    parse_csv_auto
)
from src.evaluation.metrics import (
    TeamEvaluator, 
    TeamQualityMetrics,
    format_score_with_icon,
    get_overall_status,
    LatencyTracker,
    ExtractionEvaluator,
    EXTRACTION_TEST_SET,
    ExtractionMetrics
)
from src.utils import format_experience, format_skills_list, safe_float


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def load_engine():
    """Load and cache the SkillSync engine."""
    try:
        return SkillSyncEngine()
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_metric_card(label: str, value: Any, icon: str = "📊"):
    """Display a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def display_score_badge(score: float, label: str = None):
    """Display score with appropriate badge styling."""
    if score >= 0.8:
        badge_class = "badge-success"
        icon = "✅"
    elif score >= 0.5:
        badge_class = "badge-warning"
        icon = "⚠️"
    else:
        badge_class = "badge-danger"
        icon = "❌"
    
    text = f"{icon} {score:.0%}"
    if label:
        text = f"{label}: {text}"
    
    st.markdown(f'<span class="status-badge {badge_class}">{text}</span>', unsafe_allow_html=True)


def display_team_member_compact(member: Dict[str, Any]):
    """Display team member in compact format."""
    name = member.get("name", "Unknown")
    role = member.get("role", "Unknown")
    exp = member.get("metadata", {}).get("work_experience_years", 0)
    skills = member.get("technical", {}).get("skills", [])[:4]
    belbin = member.get("personality", {}).get("Belbin_team_role", "Unknown")
    match = member.get("match_score", 0)
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{name}** - {role}")
            st.caption(f"📅 {format_experience(exp)} | 🎭 {belbin}")
            st.caption(f"🛠️ {', '.join(skills)}")
        with col2:
            display_score_badge(match, "Match")


def display_quality_metrics_inline(metrics: TeamQualityMetrics):
    """Display quality metrics inline with icons."""
    icons = metrics.get_status_icons()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Skills", f"{metrics.skill_coverage:.0%}", delta=None)
        st.caption(icons["skill_coverage"])
    with col2:
        st.metric("Diversity", f"{metrics.role_diversity:.0%}", delta=None)
        st.caption(icons["role_diversity"])
    with col3:
        st.metric("Experience", f"{metrics.experience_balance:.0%}", delta=None)
        st.caption(icons["experience_balance"])
    with col4:
        st.metric("Overall", f"{metrics.overall_score:.0%}", delta=None)
        st.caption(icons["overall"])


# ============================================================================
# DASHBOARD TAB
# ============================================================================

def render_dashboard(engine):
    """Render the dashboard tab."""
    st.header("📊 Dashboard")
    st.markdown("Welcome to SkillSync AI - Your intelligent team formation assistant")
    
    # Quick Stats Row
    st.subheader("Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Candidates",
            value=engine.candidate_count,
            delta=None
        )
    
    with col2:
        # Calculate role distribution
        roles = {}
        for p in engine.raw_data:
            role = p.get("role", "Unknown")
            roles[role] = roles.get(role, 0) + 1
        st.metric(
            label="🎭 Unique Roles",
            value=len(roles),
            delta=None
        )
    
    with col3:
        # Calculate average experience
        experiences = []
        for p in engine.raw_data:
            try:
                exp = float(p.get("metadata", {}).get("work_experience_years", 0))
                experiences.append(exp)
            except:
                pass
        avg_exp = sum(experiences) / len(experiences) if experiences else 0
        st.metric(
            label="📈 Avg Experience",
            value=f"{avg_exp:.1f} yrs",
            delta=None
        )
    
    with col4:
        # Count industries
        industries = set()
        for p in engine.raw_data:
            ind = p.get("metadata", {}).get("industry", "")
            if ind and ind != "Other":
                industries.add(ind)
        st.metric(
            label="🏢 Industries",
            value=len(industries),
            delta=None
        )
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔨 Build a Team", use_container_width=True, type="primary"):
            st.session_state["active_tab"] = 1
            st.rerun()
    
    with col2:
        if st.button("📤 Upload Candidates", use_container_width=True):
            st.session_state["active_tab"] = 2
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Role Distribution Chart
    st.subheader("👔 Role Distribution")
    if roles:
        role_df = pd.DataFrame([
            {"Role": k, "Count": v} 
            for k, v in sorted(roles.items(), key=lambda x: -x[1])[:10]
        ])
        st.bar_chart(role_df.set_index("Role"))
    
    # Industry Distribution
    st.subheader("🏢 Industry Distribution")
    industries_count = {}
    for p in engine.raw_data:
        ind = p.get("metadata", {}).get("industry", "Other")
        industries_count[ind] = industries_count.get(ind, 0) + 1
    
    if industries_count:
        ind_df = pd.DataFrame([
            {"Industry": k, "Count": v}
            for k, v in sorted(industries_count.items(), key=lambda x: -x[1])[:8]
        ])
        st.bar_chart(ind_df.set_index("Industry"))
    
    # How It Works Section
    st.markdown("---")
    st.subheader("🧠 How SkillSync Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **1️⃣ Describe Project**
        
        Tell us what you're building in plain English.
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ AI Extracts Needs**
        
        Our AI identifies required skills, roles, and expertise.
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Smart Matching**
        
        Candidates are ranked by semantic similarity to your needs.
        """)
    
    with col4:
        st.markdown("""
        **4️⃣ Team Formation**
        
        Get 3 optimized team options with explanations.
        """)


# ============================================================================
# TEAM BUILDER TAB
# ============================================================================

def render_team_builder(engine):
    """Render the team builder tab."""
    st.header("🔨 Team Builder")
    st.markdown("Describe your project and we'll form optimized teams")
    
    # Input Section
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            project_desc = st.text_area(
                "📝 Project Description",
                height=120,
                placeholder="Example: Build a Fintech mobile app with React Native, Python backend on AWS. Need payment integration and security expertise.",
                help="Describe your project in plain language. Mention technologies, domain, and any specific requirements."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            team_size = st.number_input(
                "👥 Team Size",
                min_value=2,
                max_value=10,
                value=4
            )
            
            min_availability = st.selectbox(
                "⏰ Min Availability",
                options=[None, 10, 20, 30, 40],
                format_func=lambda x: "Any" if x is None else f"{x}+ hrs/week"
            )
            
            generate_btn = st.button(
                "✨ Generate Teams",
                type="primary",
                use_container_width=True
            )
    
    # Process and Display Results
    if generate_btn and project_desc:
        with st.spinner("🤖 AI is analyzing requirements and forming teams..."):
            start_time = time.time()
            
            results = engine.form_teams(
                project_desc,
                team_size,
                include_evaluation=True
            )
            
            total_time = (time.time() - start_time) * 1000
            
            if isinstance(results, tuple) and len(results) == 2:
                strategies, requirements = results
                
                if isinstance(strategies, dict) and "error" in strategies:
                    st.error(strategies["error"])
                else:
                    st.session_state['team_results'] = strategies
                    st.session_state['requirements'] = requirements
                    st.session_state['generation_time'] = total_time
    
    # Display Results (if available)
    if 'team_results' in st.session_state:
        strategies = st.session_state['team_results']
        requirements = st.session_state['requirements']
        gen_time = st.session_state.get('generation_time', 0)
        
        # Requirements Summary (collapsible)
        with st.expander("🧠 Extracted Requirements", expanded=False):
            req_cols = st.columns(4)
            req_cols[0].write(f"**Domain:** {requirements.domain}")
            req_cols[1].write(f"**Seniority:** {requirements.seniority_level}")
            req_cols[2].write(f"**Skills:** {len(requirements.technical_keywords)}")
            req_cols[3].write(f"**Roles:** {len(requirements.target_roles)}")
            
            st.info(f"**Summary:** {requirements.summary}")
            
            if requirements.technical_keywords:
                st.write(f"**Technologies:** {', '.join(requirements.technical_keywords[:12])}")
            if requirements.target_roles:
                st.write(f"**Target Roles:** {', '.join(requirements.target_roles)}")
        
        st.success(f"✅ Generated 3 team options in {gen_time:.0f}ms")
        st.markdown("---")
        
        # TEAMS FIRST - Display all teams in columns
        st.subheader("👥 Your Team Options")
        
        team_cols = st.columns(3)
        strategy_icons = ["🏆", "⚖️", "🎨"]
        strategy_names = list(strategies.keys())
        
        for idx, (name, result) in enumerate(strategies.items()):
            with team_cols[idx]:
                st.markdown(f"### {strategy_icons[idx]} {result.strategy_name}")
                
                # Quick Quality Score
                metrics = result.metadata.get("evaluation", {})
                overall = metrics.get("overall_score", 0)
                icon, label, _ = get_overall_status(
                    TeamQualityMetrics(
                        skill_coverage=metrics.get("skill_coverage", 0),
                        role_diversity=metrics.get("role_diversity", 0),
                        experience_balance=metrics.get("experience_balance", 0),
                        avg_match_score=metrics.get("avg_match_score", 0),
                        availability_fit=1.0
                    )
                )
                
                st.markdown(f"**{icon} {label}** - {overall:.0%} overall")
                
                # Compact Quality Metrics
                col1, col2 = st.columns(2)
                col1.caption(f"Skills: {metrics.get('skill_coverage', 0):.0%}")
                col2.caption(f"Diversity: {metrics.get('role_diversity', 0):.0%}")
                
                st.markdown("---")
                
                # Team Members
                for member in result.members:
                    with st.container():
                        name_str = member.get("name", "Unknown")
                        role_str = member.get("role", "Unknown")
                        exp = member.get("metadata", {}).get("work_experience_years", 0)
                        match = member.get("match_score", 0)
                        
                        st.markdown(f"**{name_str}**")
                        st.caption(f"{role_str} | {format_experience(exp)} | Match: {match:.0%}")
                        
                        skills = member.get("technical", {}).get("skills", [])[:3]
                        st.caption(f"🛠️ {', '.join(skills)}")
                        st.markdown("")
        
        st.markdown("---")
        
        # EXPLANATIONS BELOW
        st.subheader("💡 AI Analysis")
        
        for idx, (name, result) in enumerate(strategies.items()):
            with st.expander(f"{strategy_icons[idx]} {result.strategy_name} - Analysis", expanded=(idx == 0)):
                st.markdown(result.llm_analysis)
                
                # Detailed metrics
                if "evaluation" in result.metadata:
                    st.markdown("**Quality Breakdown:**")
                    m = result.metadata["evaluation"]
                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Skills", f"{m.get('skill_coverage', 0):.0%}")
                    metric_cols[1].metric("Diversity", f"{m.get('role_diversity', 0):.0%}")
                    metric_cols[2].metric("Experience", f"{m.get('experience_balance', 0):.0%}")
                    metric_cols[3].metric("Match", f"{m.get('avg_match_score', 0):.0%}")
                    metric_cols[4].metric("Overall", f"{m.get('overall_score', 0):.0%}")
                
                # Validation warnings
                validation = result.metadata.get("validation", {})
                if validation.get("warnings"):
                    st.warning("**Potential Gaps:**")
                    for w in validation["warnings"][:3]:
                        st.caption(f"⚠️ {w}")


# ============================================================================
# TALENT POOL TAB
# ============================================================================

def render_talent_pool(engine):
    """Render the talent pool tab."""
    st.header("👥 Talent Pool")
    
    # Upload Section
    upload_col1, upload_col2 = st.columns(2)
    
    with upload_col1:
        st.subheader("📤 Upload CSV")
        st.markdown("Supports both simple and StackOverflow formats (auto-detected)")
        
        uploaded_csv = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="csv_upload",
            help="Upload employee data. We auto-detect column format."
        )
        
        if uploaded_csv is not None:
            # Detect format and show preview
            uploaded_csv.seek(0)
            format_type = detect_csv_format(uploaded_csv)
            uploaded_csv.seek(0)
            
            st.info(f"📋 Detected format: **{format_type.upper()}**")
            
            # Preview
            preview_df = pd.read_csv(uploaded_csv)
            st.write(f"Found **{len(preview_df)} rows** and **{len(preview_df.columns)} columns**")
            
            with st.expander("Preview Data"):
                st.dataframe(preview_df.head(5))
            
            uploaded_csv.seek(0)
            
            # Options
            col1, col2 = st.columns(2)
            with col1:
                use_llm_roles = st.checkbox(
                    "Use AI to map 'Other' roles",
                    value=True,
                    help="Use LLM to map ambiguous role names to standard roles"
                )
            with col2:
                min_avail_filter = st.selectbox(
                    "Min availability filter",
                    options=[None, 10, 20, 30],
                    format_func=lambda x: "No filter" if x is None else f"{x}+ hrs/week"
                )
            
            if st.button("➕ Add to Database", type="primary"):
                with st.spinner("Processing CSV..."):
                    try:
                        uploaded_csv.seek(0)
                        profiles = parse_csv_auto(
                            uploaded_csv,
                            llm_client=engine.llm_client,
                            llm_model=engine.extractor.config.model if hasattr(engine.extractor, 'config') else "openai/gpt-4o-mini",
                            use_llm_for_roles=use_llm_roles,
                            min_availability_hours=min_avail_filter
                        )
                        
                        # Add to engine
                        engine.raw_data.extend(profiles)
                        from src.utils import save_data
                        save_data(engine.data_path, engine.raw_data)
                        
                        st.success(f"✅ Added {len(profiles)} candidates!")
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with upload_col2:
        st.subheader("📄 Upload Resume (PDF)")
        st.markdown("Extract candidate profile from PDF resume")
        
        uploaded_pdf = st.file_uploader(
            "Choose PDF file",
            type=["pdf"],
            key="pdf_upload"
        )
        
        if uploaded_pdf is not None:
            if st.button("🔍 Parse Resume"):
                with st.spinner("Extracting profile from resume..."):
                    try:
                        profile = engine.add_candidates_from_pdf(uploaded_pdf.read())
                        st.success(f"✅ Added: {profile.get('name', 'Unknown')}")
                        
                        with st.expander("View Extracted Profile"):
                            st.json(profile)
                        
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Error parsing PDF: {e}")
    
    st.markdown("---")
    
    # Current Database
    st.subheader("📊 Current Database")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search", placeholder="Search by name, skill, or role...")
    with col2:
        role_filter = st.selectbox(
            "Filter by Role",
            options=["All"] + list(set(p.get("role", "") for p in engine.raw_data if p.get("role")))
        )
    with col3:
        industry_filter = st.selectbox(
            "Filter by Industry",
            options=["All"] + list(set(p.get("metadata", {}).get("industry", "") for p in engine.raw_data if p.get("metadata", {}).get("industry")))
        )
    
    # Build display dataframe with all relevant columns
    if engine.raw_data:
        flat_data = []
        for p in engine.raw_data:
            flat_data.append({
                "ID": p.get("id", "N/A")[:8],
                "Name": p.get("name", "N/A"),
                "Role": p.get("role", "N/A"),
                "Experience": format_experience(p.get("metadata", {}).get("work_experience_years", 0)),
                "Industry": p.get("metadata", {}).get("industry", "N/A"),
                "Skills": format_skills_list(p.get("technical", {}).get("skills", []), 4),
                "Tools": format_skills_list(p.get("technical", {}).get("tools", []), 3),
                "Belbin": p.get("personality", {}).get("Belbin_team_role", "N/A"),
                "Availability": p.get("constraints", {}).get("weekly_availability_hours", "N/A"),
                "Communication": p.get("collaboration", {}).get("communication_style", "N/A"),
            })
        
        df = pd.DataFrame(flat_data)
        
        # Apply filters
        if search_query:
            mask = df.apply(
                lambda x: x.astype(str).str.contains(search_query, case=False).any(),
                axis=1
            )
            df = df[mask]
        
        if role_filter != "All":
            df = df[df["Role"] == role_filter]
        
        if industry_filter != "All":
            df = df[df["Industry"] == industry_filter]
        
        st.dataframe(df, use_container_width=True, height=400)
        st.caption(f"Showing {len(df)} of {len(engine.raw_data)} candidates")
    else:
        st.warning("No candidates in database. Upload CSV or PDF to add candidates.")


# ============================================================================
# EVALUATION TAB
# ============================================================================

def render_evaluation(engine):
    """Render extraction evaluation tab."""
    st.header("📊 Extraction Evaluation")
    st.markdown("""
    Test the AI's ability to extract structured requirements from project descriptions.
    Compares extraction results against **10 ground-truth test cases** covering different domains.
    """)
    
    # Info about test set
    with st.expander("ℹ️ About the Test Set", expanded=False):
        st.markdown(f"""
        **Test Coverage:**
        - {len(EXTRACTION_TEST_SET)} diverse project descriptions
        - Domains: Fintech, Healthcare, E-commerce, IoT, Education, Cybersecurity, Manufacturing, Gaming, AI, Cloud Infrastructure
        - Measures: Technical skills, Tools, Roles, Domain classification
        
        **Metrics:**
        - **Precision**: % of extracted items that are correct
        - **Recall**: % of expected items that were found
        - **F1 Score**: Harmonic mean of precision and recall
        - **Domain/Role Accuracy**: Correct classification rate
        """)
    
    st.markdown("---")
    
    # Run Evaluation Button
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        run_full_eval = st.button(
            "🧪 Run Full Evaluation (10 tests)",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        run_sample_eval = st.button(
            "⚡ Run Quick Sample (3 tests)",
            use_container_width=True
        )
    
    # Run evaluation
    if run_full_eval or run_sample_eval or 'eval_results' in st.session_state:
        
        # Perform evaluation if button clicked
        if run_full_eval or run_sample_eval:
            test_set = EXTRACTION_TEST_SET if run_full_eval else EXTRACTION_TEST_SET[:3]
            
            with st.spinner(f"🔬 Running {len(test_set)} extraction tests..."):
                evaluator = ExtractionEvaluator(test_set=test_set)
                
                start_time = time.time()
                results = evaluator.evaluate_extractor(engine.extractor)
                elapsed_time = (time.time() - start_time) * 1000
                
                # Store in session state
                st.session_state['eval_results'] = results
                st.session_state['eval_time'] = elapsed_time
                st.session_state['eval_num_tests'] = len(test_set)
        
        # Display results
        if 'eval_results' in st.session_state:
            results = st.session_state['eval_results']
            elapsed_time = st.session_state.get('eval_time', 0)
            num_tests = st.session_state.get('eval_num_tests', len(EXTRACTION_TEST_SET))
            
            st.success(f"✅ Evaluation complete! Tested {num_tests} cases in {elapsed_time:.0f}ms")
            st.markdown("---")
            
            # === AGGREGATE METRICS ===
            st.subheader("📈 Overall Performance")
            
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                icon = "✅" if results.precision >= 0.7 else ("⚠️" if results.precision >= 0.5 else "❌")
                st.metric(
                    "Precision",
                    f"{results.precision:.1%}",
                    help="% of extracted items that are correct"
                )
                st.caption(f"{icon} Technical Skills")
            
            with metric_cols[1]:
                icon = "✅" if results.recall >= 0.7 else ("⚠️" if results.recall >= 0.5 else "❌")
                st.metric(
                    "Recall",
                    f"{results.recall:.1%}",
                    help="% of expected items that were found"
                )
                st.caption(f"{icon} Coverage")
            
            with metric_cols[2]:
                icon = "✅" if results.f1_score >= 0.7 else ("⚠️" if results.f1_score >= 0.5 else "❌")
                st.metric(
                    "F1 Score",
                    f"{results.f1_score:.3f}",
                    help="Balanced metric (harmonic mean)"
                )
                st.caption(f"{icon} Overall Quality")
            
            with metric_cols[3]:
                icon = "✅" if results.domain_accuracy >= 0.8 else ("⚠️" if results.domain_accuracy >= 0.6 else "❌")
                st.metric(
                    "Domain Accuracy",
                    f"{results.domain_accuracy:.1%}",
                    help="Correct domain classification"
                )
                st.caption(f"{icon} Classification")
            
            with metric_cols[4]:
                icon = "✅" if results.role_accuracy >= 0.7 else ("⚠️" if results.role_accuracy >= 0.5 else "❌")
                st.metric(
                    "Role F1",
                    f"{results.role_accuracy:.3f}",
                    help="Role extraction quality"
                )
                st.caption(f"{icon} Role Matching")
            
            # === INTERPRETATION ===
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            overall_score = (results.f1_score + results.domain_accuracy + results.role_accuracy) / 3
            
            if overall_score >= 0.75:
                st.success(f"""
                **🎉 Excellent Performance** ({overall_score:.1%})
                
                The extraction system is performing very well! It accurately identifies technical requirements,
                classifies domains correctly, and matches roles effectively.
                """)
            elif overall_score >= 0.60:
                st.warning(f"""
                **✅ Good Performance** ({overall_score:.1%})
                
                The extraction system works well overall but could be improved. Check the detailed results
                below to see specific areas for optimization.
                """)
            else:
                st.error(f"""
                **⚠️ Needs Improvement** ({overall_score:.1%})
                
                The extraction system is struggling with some test cases. Review the detailed per-test
                results below to identify patterns in errors.
                """)
            
            # === DETAILED RESULTS ===
            st.markdown("---")
            st.subheader("🔬 Per-Test Results")
            
            if results.details and "per_test" in results.details:
                test_results = results.details["per_test"]
                
                # Create summary table
                table_data = []
                for i, test_result in enumerate(test_results):
                    test_case = EXTRACTION_TEST_SET[i] if i < len(EXTRACTION_TEST_SET) else {}
                    
                    table_data.append({
                        "Test ID": test_result.get("test_id", f"test_{i+1}"),
                        "Domain": test_case.get("expected", {}).get("domain", "N/A"),
                        "Skills F1": f"{test_result.get('skills', {}).get('f1', 0):.3f}",
                        "Tools F1": f"{test_result.get('tools', {}).get('f1', 0):.3f}",
                        "Roles F1": f"{test_result.get('roles', {}).get('f1', 0):.3f}",
                        "Domain Match": "✅" if test_result.get("domain_match", 0) == 1.0 else "❌"
                    })
                
                df_results = pd.DataFrame(table_data)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Detailed view
                with st.expander("📋 View Detailed Test Cases"):
                    for i, test_result in enumerate(test_results):
                        if i >= len(EXTRACTION_TEST_SET):
                            continue
                        
                        test_case = EXTRACTION_TEST_SET[i]
                        test_id = test_result.get("test_id")
                        
                        st.markdown(f"#### {test_id}")
                        st.caption(f"**Description:** {test_case['description'][:200]}...")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Expected:**")
                            st.json({
                                "skills": test_case["expected"].get("technical_keywords", [])[:5],
                                "roles": test_case["expected"].get("target_roles", []),
                                "domain": test_case["expected"].get("domain")
                            })
                        
                        with col2:
                            st.markdown("**Metrics:**")
                            skills_metrics = test_result.get("skills", {})
                            st.write(f"Skills - P: {skills_metrics.get('precision', 0):.2f}, R: {skills_metrics.get('recall', 0):.2f}, F1: {skills_metrics.get('f1', 0):.2f}")
                            roles_metrics = test_result.get("roles", {})
                            st.write(f"Roles - F1: {roles_metrics.get('f1', 0):.2f}")
                            st.write(f"Domain: {'✅ Match' if test_result.get('domain_match') == 1.0 else '❌ Mismatch'}")
                        
                        st.markdown("---")
            
            # === RECOMMENDATIONS ===
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            if results.precision < 0.7:
                recommendations.append("**Reduce Hallucinations**: The system is extracting items not in the text. Strengthen validation prompts.")
            
            if results.recall < 0.7:
                recommendations.append("**Improve Coverage**: The system is missing items. Enhance extraction prompts to be more comprehensive.")
            
            if results.domain_accuracy < 0.8:
                recommendations.append("**Better Domain Classification**: Improve domain detection by providing more examples or constraints.")
            
            if results.role_accuracy < 0.7:
                recommendations.append("**Role Matching**: Refine role inference rules or expand the ALLOWED_ROLES list.")
            
            if not recommendations:
                recommendations.append("**✅ System is performing well!** Continue monitoring on new test cases.")
            
            for rec in recommendations:
                st.info(rec)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load engine
    engine = load_engine()
    
    if engine is None:
        st.error("🚨 Failed to initialize SkillSync Engine")
        st.info("""
        **Setup Required:**
        1. Create `.env` file with your API key
        2. Run `python -m src.data_generator` to generate sample data
        """)
        st.code("""
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("🧩 SkillSync AI")
        st.caption("Intelligent Team Formation")
        st.markdown("---")
        
        # Stats
        st.metric("📊 Candidates", engine.candidate_count)
        
        # Quick actions
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **SkillSync AI** uses:
            - 🤖 LLM for requirement extraction
            - 🔍 Semantic search for matching
            - 📊 Multiple team strategies
            - ✅ Quality evaluation
            
            Built for TU Wien GenAI Course
            """)
        
        st.markdown("---")
        st.caption("Group 45 | 2025W")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔨 Team Builder",
        "👥 Talent Pool",
        "🧪 Evaluation"
    ])
    
    with tab1:
        render_dashboard(engine)
    
    with tab2:
        render_team_builder(engine)
    
    with tab3:
        render_talent_pool(engine)
    
    with tab4:
        render_evaluation(engine)


if __name__ == "__main__":
    main()