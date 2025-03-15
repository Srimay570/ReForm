import numpy as np
from datetime import datetime, timedelta

# Medical guidelines and reference ranges
HEALTH_GUIDELINES = {
    'sleep': {
        'recommended_range': (7, 9),  # hours
        'risk_levels': {
            'low': (6, 10),
            'moderate': (5, 11),
            'high': (0, 5)  # or > 11
        },
        'recommendations': {
            'low': [
                "Minor sleep pattern adjustment needed",
                "Try to maintain consistent sleep schedule",
                "Limit screen time before bed"
            ],
            'moderate': [
                "Sleep habits need attention",
                "Establish a bedtime routine",
                "Consider sleep environment improvements",
                "Avoid caffeine after 2 PM"
            ],
            'high': [
                "Immediate sleep pattern intervention recommended",
                "Consult a healthcare provider",
                "Track sleep quality and duration",
                "Evaluate lifestyle factors affecting sleep"
            ]
        }
    },
    'exercise': {
        'recommended_range': (150, 300),  # minutes per week
        'risk_levels': {
            'low': (120, 330),
            'moderate': (60, 120),
            'high': (0, 60)
        },
        'recommendations': {
            'low': [
                "Current exercise level is good",
                "Mix different types of activities",
                "Include strength training"
            ],
            'moderate': [
                "Increase activity gradually",
                "Find activities you enjoy",
                "Set realistic exercise goals",
                "Consider working with a fitness professional"
            ],
            'high': [
                "Start with walking 10 minutes daily",
                "Consult healthcare provider before starting intensive exercise",
                "Set small, achievable goals",
                "Focus on consistency over intensity"
            ]
        }
    },
    'water': {
        'recommended_range': (8, 10),  # glasses per day
        'risk_levels': {
            'low': (6, 12),
            'moderate': (4, 6),
            'high': (0, 4)
        },
        'recommendations': {
            'low': [
                "Maintain current hydration habits",
                "Adjust intake based on activity level",
                "Monitor urine color for hydration"
            ],
            'moderate': [
                "Set hydration reminders",
                "Keep water bottle visible",
                "Track daily intake",
                "Include water-rich foods"
            ],
            'high': [
                "Immediate hydration improvement needed",
                "Set hourly water intake goals",
                "Monitor signs of dehydration",
                "Consult healthcare provider if persistent"
            ]
        }
    },
    'stress': {
        'recommended_range': (1, 4),  # scale 1-10
        'risk_levels': {
            'low': (1, 5),
            'moderate': (6, 7),
            'high': (8, 10)
        },
        'recommendations': {
            'low': [
                "Continue current stress management practices",
                "Practice preventive self-care",
                "Maintain work-life balance"
            ],
            'moderate': [
                "Implement daily stress-reduction techniques",
                "Consider mindfulness or meditation",
                "Evaluate stress triggers",
                "Ensure adequate rest and recovery"
            ],
            'high': [
                "Seek professional support",
                "Practice stress-reduction techniques daily",
                "Evaluate work and life commitments",
                "Prioritize mental health support"
            ]
        }
    }
}

def analyze_metric_trend(values, dates, metric_name):
    """
    Analyze trend for a specific health metric
    """
    if len(values) < 3:
        return {
            'trend': 'insufficient_data',
            'recommendation': 'Continue tracking to establish trends'
        }

    # Calculate trend
    values_arr = np.array(values)
    recent_avg = np.mean(values_arr[-3:])
    older_avg = np.mean(values_arr[:-3])
    
    # Determine trend direction
    trend = 'improving' if recent_avg > older_avg else 'declining'
    if abs(recent_avg - older_avg) < 0.1:
        trend = 'stable'
    
    # Get relevant guidelines
    guidelines = HEALTH_GUIDELINES.get(metric_name, {})
    recommended_range = guidelines.get('recommended_range', (0, 0))
    
    # Determine current status
    current_value = values[-1]
    status = 'optimal' if recommended_range[0] <= current_value <= recommended_range[1] else 'suboptimal'
    
    return {
        'trend': trend,
        'current_value': current_value,
        'recommended_range': recommended_range,
        'status': status
    }

def get_health_risk_level(value, metric_name):
    """
    Determine risk level based on medical guidelines
    """
    guidelines = HEALTH_GUIDELINES.get(metric_name, {})
    risk_levels = guidelines.get('risk_levels', {})
    
    for level, (min_val, max_val) in risk_levels.items():
        if min_val <= value <= max_val:
            return level
    return 'high'  # Default to high risk if outside all ranges

def get_health_recommendations(metrics_data):
    """
    Generate detailed health recommendations based on medical guidelines
    """
    recommendations = []
    risk_summary = {}
    
    for metric_name, values in metrics_data.items():
        if not values:
            continue
            
        current_value = values[-1]
        risk_level = get_health_risk_level(current_value, metric_name)
        risk_summary[metric_name] = risk_level
        
        # Get metric-specific recommendations
        metric_guidelines = HEALTH_GUIDELINES.get(metric_name, {})
        metric_recommendations = metric_guidelines.get('recommendations', {}).get(risk_level, [])
        
        if metric_recommendations:
            recommendations.extend([
                f"📊 {metric_name.title()} ({risk_level} risk):",
                *[f"  • {rec}" for rec in metric_recommendations]
            ])
    
    return {
        'recommendations': recommendations,
        'risk_summary': risk_summary
    }

def calculate_detailed_health_score(metrics_data):
    """
    Calculate comprehensive health score with detailed breakdown
    """
    if not metrics_data or not any(metrics_data.values()):
        return {
            'overall_score': 0,
            'component_scores': {},
            'risk_levels': {},
            'recommendations': []
        }

    component_scores = {}
    risk_levels = {}
    
    # Calculate individual component scores
    for metric, values in metrics_data.items():
        if not values:
            continue
            
        current_value = values[-1]
        guidelines = HEALTH_GUIDELINES.get(metric, {})
        recommended_range = guidelines.get('recommended_range', (0, 0))
        
        # Calculate score based on distance from recommended range
        if metric == 'stress':  # Inverse scoring for stress
            score = 10 - (current_value / 10) * 10
        else:
            mid_range = sum(recommended_range) / 2
            max_deviation = max(recommended_range) - mid_range
            deviation = abs(current_value - mid_range)
            score = max(0, 10 - (deviation / max_deviation) * 5)
        
        component_scores[metric] = round(score, 1)
        risk_levels[metric] = get_health_risk_level(current_value, metric)
    
    # Calculate weighted overall score
    weights = {
        'sleep': 0.3,
        'exercise': 0.3,
        'water': 0.2,
        'stress': 0.2
    }
    
    overall_score = sum(
        component_scores.get(metric, 0) * weight
        for metric, weight in weights.items()
    )
    
    # Get recommendations
    health_recommendations = get_health_recommendations(metrics_data)
    
    return {
        'overall_score': round(overall_score, 1),
        'component_scores': component_scores,
        'risk_levels': risk_levels,
        'recommendations': health_recommendations['recommendations']
    }
