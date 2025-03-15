import pandas as pd
import numpy as np

def calculate_health_score(df):
    """
    Calculate overall health score based on metrics
    """
    if len(df) == 0:
        return 0.0
    
    # Calculate individual scores
    sleep_score = df['sleep'].apply(lambda x: min(x/8 * 10, 10)).mean()
    exercise_score = df['exercise'].apply(lambda x: min(x/30 * 10, 10)).mean()
    water_score = df['water'].apply(lambda x: min(x/8 * 10, 10)).mean()
    stress_score = (10 - df['stress']).mean()  # Inverse of stress level
    
    # Calculate weighted average
    weights = {
        'sleep': 0.3,
        'exercise': 0.3,
        'water': 0.2,
        'stress': 0.2
    }
    
    health_score = (
        sleep_score * weights['sleep'] +
        exercise_score * weights['exercise'] +
        water_score * weights['water'] +
        stress_score * weights['stress']
    )
    
    return round(health_score, 1)

def analyze_study_patterns(df):
    """
    Analyze study patterns from logged data
    """
    if len(df) == 0:
        return {
            "total_sessions": 0,
            "avg_duration": 0,
            "productivity_score": 0
        }
    
    patterns = {
        "total_sessions": len(df),
        "avg_duration": df['duration'].mean(),
        "productivity_score": df['productivity'].mean(),
        "most_studied_subject": df['subject'].mode().iloc[0] if not df['subject'].empty else None,
        "total_study_time": df['duration'].sum()
    }
    
    return patterns
