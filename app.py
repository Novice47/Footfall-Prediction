from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
import time

app = Flask(__name__)
CORS(app)

# Load Models & Data
SITES_INFO = [
    {"id": 1, "name": "Taj Mahal", "country": "India", "lat": 27.1751, "lon": 78.0421},
    {"id": 2, "name": "Eiffel Tower", "country": "France", "lat": 48.8584, "lon": 2.2945},
    {"id": 3, "name": "Great Wall of China", "country": "China", "lat": 40.4319, "lon": 116.5704},
    {"id": 4, "name": "Colosseum", "country": "Italy", "lat": 41.8902, "lon": 12.4922},
    {"id": 5, "name": "Machu Picchu", "country": "Peru", "lat": -13.1631, "lon": -72.5450},
    {"id": 6, "name": "Christ the Redeemer", "country": "Brazil", "lat": -22.9519, "lon": -43.2105},
    {"id": 7, "name": "Chichen Itza", "country": "Mexico", "lat": 20.6843, "lon": -88.5678},
    {"id": 8, "name": "Petra", "country": "Jordan", "lat": 30.3285, "lon": 35.4444},
    {"id": 9, "name": "Angkor Wat", "country": "Cambodia", "lat": 13.4125, "lon": 103.8670},
    {"id": 10, "name": "Stonehenge", "country": "UK", "lat": 51.1789, "lon": -1.8262},
]

MODELS = {}
SCALER = None
METRICS = {}
DATASET = None

def load_resources():
    global MODELS, SCALER, METRICS, DATASET
    try:
        with open("ml_models/random_forest.pkl", "rb") as f: MODELS["random_forest"] = pickle.load(f)
        with open("ml_models/gradient_boosting.pkl", "rb") as f: MODELS["gradient_boosting"] = pickle.load(f)
        with open("ml_models/linear_regression.pkl", "rb") as f: MODELS["linear_regression"] = pickle.load(f)
        with open("ml_models/scaler.pkl", "rb") as f: SCALER = pickle.load(f)
        with open("ml_models/metrics.json", "r") as f: METRICS = json.load(f)
        DATASET = pd.read_csv("data/heritage_footfall_dataset.csv")
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_type = data.get("model_type", "random_forest")
    
    # Features required for scaler
    # ['place_id', 'hour', 'day_of_week', 'month', 'season', 'is_holiday', 'is_weekend', 'temperature', 'humidity', 'mobile_signals', 'wifi_connections', 'event_flag', 'visibility_km', 'prev_hour_footfall']
    
    input_data = [
        data.get("place_id"),
        data.get("hour"),
        data.get("day_of_week", 0),
        data.get("month", 1),
        data.get("season", 1),
        data.get("is_holiday", 0),
        data.get("is_weekend", 0),
        data.get("temperature", 20.0),
        data.get("humidity", 50.0),
        data.get("mobile_signals", 10000),
        data.get("wifi_connections", 3500),
        data.get("event_flag", 0),
        data.get("visibility_km", 25.0),
        data.get("prev_hour_footfall", 5000)
    ]
    
    X_scaled = SCALER.transform([input_data])
    prediction = MODELS[model_type].predict(X_scaled)[0]
    
    # Simulated Confidence Interval
    # Based on MAE from metrics
    confidence_interval = METRICS.get(model_type, {}).get("mae", 500) * 1.5
    
    # Simulated Hourly Trend
    hourly_trend = []
    for h in range(6, 22):
        trend_input = input_data.copy()
        trend_input[1] = h
        trend_scaled = SCALER.transform([trend_input])
        trend_pred = MODELS[model_type].predict(trend_scaled)[0]
        hourly_trend.append({"hour": h, "footfall": int(trend_pred)})
    
    return jsonify({
        "prediction": int(prediction),
        "confidence": round(confidence_interval, 2),
        "hourly_trend": hourly_trend,
        "model": model_type
    })

@app.route("/analytics")
def analytics():
    # Site comparison
    site_comp = DATASET.groupby('place_name')['footfall'].mean().to_dict()
    
    # Monthly trend (overall)
    monthly_trend = DATASET.groupby('month')['footfall'].mean().to_dict()
    
    # Hourly trend (overall)
    hourly_trend = DATASET.groupby('hour')['footfall'].mean().to_dict()
    
    return jsonify({
        "site_comparison": site_comp,
        "monthly_trend": monthly_trend,
        "hourly_trend": hourly_trend,
        "total_predictions": len(DATASET) * 2 # Simulated
    })

@app.route("/map-data")
def map_data():
    features = []
    for site in SITES_INFO:
        # Get mean footfall for color coding
        mean_f = int(DATASET[DATASET['place_id'] == site['id']]['footfall'].mean())
        density = "high" if mean_f > 30000 else ("moderate" if mean_f > 15000 else "low")
        
        features.append({
            "type": "Feature",
            "properties": {
                "name": site["name"],
                "country": site["country"],
                "id": site["id"],
                "mean_footfall": mean_f,
                "density": density
            },
            "geometry": {
                "type": "Point",
                "coordinates": [site["lon"], site["lat"]]
            }
        })
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route("/dataset")
def dataset_paginated():
    page = int(request.args.get("page", 1))
    per_page = 50
    start = (page - 1) * per_page
    end = start + per_page
    
    subset = DATASET.iloc[start:end].to_dict(orient="records")
    return jsonify({
        "data": subset,
        "total": len(DATASET),
        "page": page,
        "per_page": per_page
    })

@app.route("/model-metrics")
def model_metrics():
    with open("ml_models/feature_importance.json", "r") as f:
        importance = json.load(f)
    return jsonify({
        "metrics": METRICS,
        "feature_importance": importance
    })

@app.route("/live-feed")
def live_feed():
    # Simulate a real-time signal update
    import random
    site = random.choice(SITES_INFO)
    return jsonify({
        "site": site["name"],
        "signals": random.randint(5000, 45000),
        "wifi": random.randint(1500, 15000),
        "timestamp": time.strftime("%H:%M:%S")
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
