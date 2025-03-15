import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import os

# Constants for model paths
MODEL_DIR = "ml_models"
HEALTH_MODEL_PATH = os.path.join(MODEL_DIR, "health_model.joblib")
STUDY_MODEL_PATH = os.path.join(MODEL_DIR, "study_model.joblib")

def ensure_model_directory():
    """Ensure the model directory exists"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def extract_features(df, target_col):
    """
    Extract features from time series data
    """
    features = pd.DataFrame()

    # Time-based features
    features['dayofweek'] = df.index.dayofweek
    features['month'] = df.index.month
    features['day'] = df.index.day

    # Rolling statistics
    for col in df.columns:
        if col != target_col:
            features[f'{col}_rolling_mean_7d'] = df[col].rolling(window=7, min_periods=1).mean()
            features[f'{col}_rolling_std_7d'] = df[col].rolling(window=7, min_periods=1).std()

    return features

def train_health_model(health_df):
    """
    Train advanced health prediction model
    """
    if len(health_df) < 7:  # Need minimum data points
        return None

    # Prepare features
    health_df['date'] = pd.to_datetime(health_df['date'])
    health_df.set_index('date', inplace=True)
    health_df.sort_index(inplace=True)

    # Feature engineering
    features = extract_features(health_df, 'stress')

    # Prepare target variables
    targets = {
        'stress': health_df['stress'],
        'sleep': health_df['sleep']
    }

    models = {}
    for target_name, target in targets.items():
        # Initialize model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )

        # Train model
        model.fit(features, target)

        # Evaluate model
        scores = cross_val_score(model, features, target, cv=3)
        print(f"{target_name} model CV scores: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

        models[target_name] = model

    # Save models
    ensure_model_directory()
    joblib.dump(models, HEALTH_MODEL_PATH)

    return models

def train_study_model(study_df):
    """
    Train advanced study prediction model
    """
    if len(study_df) < 7:  # Need minimum data points
        return None

    # Prepare features
    study_df['date'] = pd.to_datetime(study_df['date'])
    study_df.set_index('date', inplace=True)
    study_df.sort_index(inplace=True)

    # Feature engineering
    features = extract_features(study_df, 'productivity')

    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(features, study_df['productivity'])

    # Evaluate model
    scores = cross_val_score(model, features, study_df['productivity'], cv=3)
    print(f"Study model CV scores: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # Save model
    ensure_model_directory()
    joblib.dump(model, STUDY_MODEL_PATH)

    return model

def predict_trends(health_df, study_df):
    """
    Predict trends using advanced ML models
    """
    trends = {}

    if len(health_df) >= 7:
        # Train or load health models
        if os.path.exists(HEALTH_MODEL_PATH):
            health_models = joblib.load(HEALTH_MODEL_PATH)
        else:
            health_models = train_health_model(health_df)

        if health_models:
            # Prepare features for prediction
            health_df['date'] = pd.to_datetime(health_df['date'])
            health_df.set_index('date', inplace=True)
            features = extract_features(health_df, None)

            # Make predictions
            latest_features = features.iloc[-1:]

            for metric, model in health_models.items():
                prediction = model.predict(latest_features)[0]
                current = health_df[metric].iloc[-1]
                trend = "improving" if prediction > current else "declining"
                trends[f"{metric}_trend"] = trend
                trends[f"{metric}_prediction"] = round(prediction, 2)

    if len(study_df) >= 7:
        # Train or load study model
        if os.path.exists(STUDY_MODEL_PATH):
            study_model = joblib.load(STUDY_MODEL_PATH)
        else:
            study_model = train_study_model(study_df)

        if study_model:
            # Prepare features for prediction
            study_df['date'] = pd.to_datetime(study_df['date'])
            study_df.set_index('date', inplace=True)
            features = extract_features(study_df, 'productivity')

            # Make predictions
            latest_features = features.iloc[-1:]
            prediction = study_model.predict(latest_features)[0]
            current = study_df['productivity'].iloc[-1]

            trends["study_productivity_trend"] = "improving" if prediction > current else "declining"
            trends["productivity_prediction"] = round(prediction, 2)

    return trends

def generate_recommendations(health_score, study_patterns):
    """
    Generate personalized recommendations using ML insights
    """
    recommendations = []

    # Health-based recommendations
    if health_score < 5:
        recommendations.extend([
            "Critical: Your health metrics need immediate attention. Consider:",
            "- Establishing a consistent sleep schedule",
            "- Starting with 10-minute exercise sessions",
            "- Setting hourly reminders for water intake"
        ])
    elif health_score < 7:
        recommendations.extend([
            "Your health metrics could use some improvement:",
            "- Aim for 7-8 hours of sleep consistently",
            "- Gradually increase exercise duration",
            "- Use stress management techniques"
        ])

    # Study-based recommendations
    avg_duration = study_patterns.get("avg_duration", 0)
    productivity_score = study_patterns.get("productivity_score", 0)

    if avg_duration < 30:
        recommendations.extend([
            "Study session recommendations:",
            "- Start with 25-minute Pomodoro sessions",
            "- Take 5-minute breaks between sessions",
            "- Gradually increase session duration"
        ])

    if productivity_score < 6:
        recommendations.extend([
            "Productivity enhancement suggestions:",
            "- Study during your peak energy hours",
            "- Use active recall techniques",
            "- Create a distraction-free environment"
        ])

    if not recommendations:
        recommendations.extend([
            "You're doing great! To maintain your progress:",
            "- Continue your current routine",
            "- Consider setting more challenging goals",
            "- Share your success strategies with others"
        ])

    return recommendations