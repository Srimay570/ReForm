import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from models import predict_trends, generate_recommendations
from utils import calculate_health_score, analyze_study_patterns
from database import get_db, User, WaitlistEntry, Habit, HealthMetric, StudyLog
from sqlalchemy.orm import Session
from health_analysis import calculate_detailed_health_score, analyze_metric_trend, HEALTH_GUIDELINES

# Initialize session state for user authentication
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Page configuration
st.set_page_config(page_title="ReForm", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 3rem 1rem;
    }
    .stTitle {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Get database session
db = next(get_db())

def login_user(email: str, password: str, db: Session) -> bool:
    """Authenticate user and set session state"""
    user = db.query(User).filter(User.email == email).first()
    if user and user.check_password(password):
        st.session_state.user_id = user.id
        st.session_state.authenticated = True
        return True
    return False

def signup_user(email: str, password: str, name: str, db: Session) -> bool:
    """Create new user account"""
    try:
        user = User(email=email, name=name)
        user.set_password(password)
        db.add(user)
        db.commit()
        st.session_state.user_id = user.id
        st.session_state.authenticated = True
        return True
    except Exception as e:
        db.rollback()
        return False

def show_auth_page():
    """Display login/signup page"""
    st.title("Welcome to ReForm")
    st.markdown("### Transform Your Life Through Data-Driven Insights")

    # Login/Signup tabs
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if login_user(email, password, db):
                    st.success("Successfully logged in! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    with tab2:
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if password != password_confirm:
                    st.error("Passwords do not match")
                elif signup_user(email, password, name, db):
                    st.success("Account created successfully! 🎉")
                    st.rerun()
                else:
                    st.error("Error creating account. Email might already be registered.")

def main():
    # Check authentication
    if not st.session_state.authenticated:
        show_auth_page()
        return

    # Sidebar navigation with icons
    st.sidebar.markdown("# 🎯 ReForm")
    st.sidebar.markdown("---")

    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.authenticated = False
        st.rerun()

    page = st.sidebar.selectbox(
        "📍 Navigate",
        ["Dashboard", "Health Tracking", "Study Progress", "Analysis"],
        format_func=lambda x: {
            "Dashboard": "📊 " + x,
            "Health Tracking": "💪 " + x,
            "Study Progress": "📚 " + x,
            "Analysis": "🧠 " + x
        }[x]
    )

    if page == "Dashboard":
        show_dashboard(db)
    elif page == "Health Tracking":
        show_health_tracking(db)
    elif page == "Study Progress":
        show_study_progress(db)
    elif page == "Analysis":
        show_analysis(db)

def show_landing_page(db: Session):
    st.title("Welcome to ReForm")
    st.markdown("### Transform Your Life Through Data-Driven Insights")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>🎯 Track Your Progress</h3>
            <p>Monitor your habits, health metrics, and study patterns in one place.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
            <h3>🧠 AI-Powered Insights</h3>
            <p>Get personalized recommendations based on your data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
        <h3>Join our Waitlist</h3>
        """, unsafe_allow_html=True)
        with st.form("waitlist_form"):
            email = st.text_input("Email Address")
            interests = st.multiselect(
                "Areas of Interest",
                ["Habit Tracking", "Health Monitoring", "Study Progress", "AI Analysis"]
            )
            submitted = st.form_submit_button("Join Waitlist")

            if submitted and email:
                entry = WaitlistEntry(
                    email=email,
                    interests=",".join(interests)
                )
                try:
                    db.add(entry)
                    db.commit()
                    st.success("Thank you for joining our waitlist! 🎉")
                except Exception as e:
                    st.error("An error occurred. Please try again.")
                    db.rollback()

def show_dashboard(db: Session):
    st.title("Your Wellness Dashboard")

    # Summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>🎯 Active Habits</h3>
            <p>Track your daily progress</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>💪 Health Score</h3>
            <p>Monitor your wellness</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>📚 Study Time</h3>
            <p>Track your learning</p>
        </div>
        """, unsafe_allow_html=True)

    # Habit tracking section
    st.markdown("### 📝 Habit Tracker")
    with st.form("habit_form"):
        habit = st.text_input("Add a new habit")
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", ["Health", "Study", "Lifestyle"])
        with col2:
            frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
        submitted = st.form_submit_button("Add Habit")

        if submitted and habit:
            new_habit = Habit(
                habit=habit,
                category=category,
                frequency=frequency,
                user_id = st.session_state.user_id
            )
            try:
                db.add(new_habit)
                db.commit()
                st.success("✅ Habit added successfully!")
            except Exception as e:
                st.error("An error occurred. Please try again.")
                db.rollback()

    # Display habits in a modern table
    habits = db.query(Habit).filter(Habit.user_id == st.session_state.user_id).all()
    if habits:
        df_habits = pd.DataFrame([{
            'habit': h.habit,
            'category': h.category,
            'frequency': h.frequency,
            'created_at': h.created_at.strftime("%Y-%m-%d")
        } for h in habits])

        st.markdown("""
        <div class="metric-container">
            <h4>Your Habits</h4>
        """, unsafe_allow_html=True)
        st.dataframe(
            df_habits,
            column_config={
                "habit": "Habit",
                "category": "Category",
                "frequency": "Frequency",
                "created_at": "Created"
            },
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

def show_health_tracking(db: Session):
    st.title("Health Metrics Tracking")
    st.markdown("### Monitor Your Daily Wellness")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
        <div class="metric-container">
        <h4>Log Today's Metrics</h4>
        """, unsafe_allow_html=True)

        # Add tooltips with medical guidelines
        with st.form("health_metrics_form"):
            date = st.date_input("Date")

            sleep_guidelines = HEALTH_GUIDELINES['sleep']
            sleep_help = f"Recommended: {sleep_guidelines['recommended_range'][0]}-{sleep_guidelines['recommended_range'][1]} hours"
            sleep = st.number_input("Sleep (hours)", 0.0, 24.0, step=0.5, help=sleep_help)

            exercise_guidelines = HEALTH_GUIDELINES['exercise']
            exercise_help = f"Recommended: {exercise_guidelines['recommended_range'][0]}-{exercise_guidelines['recommended_range'][1]} minutes per week"
            exercise = st.number_input("Exercise (minutes)", 0, 500, step=5, help=exercise_help)

            water_guidelines = HEALTH_GUIDELINES['water']
            water_help = f"Recommended: {water_guidelines['recommended_range'][0]}-{water_guidelines['recommended_range'][1]} glasses"
            water = st.number_input("Water intake (glasses)", 0, 20, help=water_help)

            stress_guidelines = HEALTH_GUIDELINES['stress']
            stress_help = "Lower is better. 1-4: Good, 5-7: Moderate, 8-10: High stress"
            stress = st.slider("Stress level", 1, 10, help=stress_help)

            submitted = st.form_submit_button("Log Health Metrics")

            if submitted:
                metric = HealthMetric(
                    date=date,
                    sleep=sleep,
                    exercise=exercise,
                    water=water,
                    stress=stress,
                    user_id = st.session_state.user_id
                )
                try:
                    db.add(metric)
                    db.commit()
                    st.success("✅ Health metrics logged successfully!")
                except Exception as e:
                    st.error("An error occurred. Please try again.")
                    db.rollback()

    with col2:
        metrics = db.query(HealthMetric).filter(HealthMetric.user_id == st.session_state.user_id).all()
        if metrics:
            df_health = pd.DataFrame([{
                'date': m.date,
                'sleep': m.sleep,
                'exercise': m.exercise,
                'water': m.water,
                'stress': m.stress
            } for m in metrics])

            # Calculate detailed health analysis
            metrics_data = {
                'sleep': df_health['sleep'].tolist(),
                'exercise': df_health['exercise'].tolist(),
                'water': df_health['water'].tolist(),
                'stress': df_health['stress'].tolist()
            }

            health_analysis = calculate_detailed_health_score(metrics_data)

            # Display health score components
            st.markdown("""
            <div class="metric-container">
            <h4>Health Score Breakdown</h4>
            """, unsafe_allow_html=True)

            cols = st.columns(4)
            for idx, (metric, score) in enumerate(health_analysis['component_scores'].items()):
                with cols[idx]:
                    risk_level = health_analysis['risk_levels'][metric]
                    color = {
                        'low': 'green',
                        'moderate': 'orange',
                        'high': 'red'
                    }.get(risk_level, 'blue')
                    st.metric(
                        metric.title(),
                        f"{score}/10",
                        delta=risk_level,
                        delta_color='normal' if risk_level == 'low' else 'inverse'
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # Display trends
            st.markdown("""
            <div class="metric-container">
            <h4>Your Health Trends</h4>
            """, unsafe_allow_html=True)

            fig = px.line(df_health, x="date", y=["sleep", "exercise", "water", "stress"],
                         title="Health Metrics Over Time")
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(t=40, l=40, r=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display medical recommendations
            st.markdown("""
            <div class="metric-container">
            <h4>💡 Medical Recommendations</h4>
            """, unsafe_allow_html=True)

            for recommendation in health_analysis['recommendations']:
                if recommendation.startswith('📊'):
                    st.markdown(f"**{recommendation}**")
                else:
                    st.markdown(recommendation)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("📝 Start tracking your health metrics to see detailed analysis and recommendations.")

def show_study_progress(db: Session):
    st.title("Study Progress Tracking")
    st.markdown("### Track Your Learning Journey")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
        <div class="metric-container">
        <h4>Log Study Session</h4>
        """, unsafe_allow_html=True)

        with st.form("study_log_form"):
            subject = st.text_input("Subject")
            duration = st.number_input("Study Duration (minutes)", 0, 1440, step=15)
            productivity = st.slider("Productivity Rating", 1, 10)
            notes = st.text_area("Study Notes")

            submitted = st.form_submit_button("Log Study Session")

            if submitted and subject:
                log = StudyLog(
                    subject=subject,
                    duration=duration,
                    productivity=productivity,
                    notes=notes,
                    user_id = st.session_state.user_id
                )
                try:
                    db.add(log)
                    db.commit()
                    st.success("✅ Study session logged successfully!")
                except Exception as e:
                    st.error("An error occurred. Please try again.")
                    db.rollback()

    with col2:
        logs = db.query(StudyLog).filter(StudyLog.user_id == st.session_state.user_id).all()
        if logs:
            df_study = pd.DataFrame([{
                'date': l.created_at,
                'subject': l.subject,
                'duration': l.duration,
                'productivity': l.productivity
            } for l in logs])

            st.markdown("""
            <div class="metric-container">
            <h4>Study Analysis</h4>
            """, unsafe_allow_html=True)

            fig = px.bar(df_study, x="subject", y="duration",
                        title="Study Time by Subject",
                        color="productivity",
                        color_continuous_scale="Viridis")
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(t=40, l=40, r=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

def show_analysis(db: Session):
    st.title("AI Analysis & Recommendations")
    st.markdown("### Your Personalized Insights")

    health_metrics = db.query(HealthMetric).filter(HealthMetric.user_id == st.session_state.user_id).all()
    study_logs = db.query(StudyLog).filter(StudyLog.user_id == st.session_state.user_id).all()

    if health_metrics and study_logs:
        # Convert to DataFrames
        df_health = pd.DataFrame([{
            'date': m.date,
            'sleep': m.sleep,
            'exercise': m.exercise,
            'water': m.water,
            'stress': m.stress
        } for m in health_metrics])

        df_study = pd.DataFrame([{
            'date': l.created_at,
            'subject': l.subject,
            'duration': l.duration,
            'productivity': l.productivity
        } for l in study_logs])

        # Generate insights
        health_score = calculate_health_score(df_health)
        study_patterns = analyze_study_patterns(df_study)
        trends = predict_trends(df_health, df_study)
        recommendations = generate_recommendations(health_score, study_patterns)

        # Display insights in a modern layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="metric-container">
            <h4>Health Score</h4>
            """, unsafe_allow_html=True)
            st.metric("Overall Health Score", f"{health_score:.1f}/10.0")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-container">
            <h4>Study Patterns</h4>
            """, unsafe_allow_html=True)
            for key, value in study_patterns.items():
                st.metric(key.replace("_", " ").title(), value)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-container">
            <h4>Predicted Trends</h4>
            """, unsafe_allow_html=True)
            for key, value in trends.items():
                st.metric(
                    key.replace("_", " ").title(),
                    value,
                    delta="positive" if "improving" in value else "negative" if "declining" in value else None
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Display recommendations in cards
        st.markdown("### 💡 Personalized Recommendations")
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-card">
                {rec}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📝 Please log some health metrics and study sessions to see AI analysis.")

if __name__ == "__main__":
    main()