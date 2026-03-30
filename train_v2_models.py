import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def train_models():
    print("Loading dataset...")
    df = pd.read_csv("data/heritage_footfall_dataset.csv")
    
    # Handle missing values
    df['temperature'] = df['temperature'].fillna(df.groupby('place_id')['temperature'].transform('mean'))
    df['humidity'] = df['humidity'].fillna(df.groupby('place_id')['humidity'].transform('mean'))
    
    # Features
    features = [
        'place_id', 'hour', 'day_of_week', 'month', 'season', 
        'is_holiday', 'is_weekend', 'temperature', 'humidity', 
        'mobile_signals', 'wifi_connections', 'event_flag', 
        'visibility_km', 'prev_hour_footfall'
    ]
    
    X = df[features]
    y = df['footfall']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "linear_regression": LinearRegression()
    }
    
    metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics[name] = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "predictions": y_pred[:100].tolist(), # For charts
            "actuals": y_test[:100].tolist()
        }
        
        # Save model
        with open(f"ml_models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    # Save scaler
    with open("ml_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # Save metrics
    with open("ml_models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Feature importance for Random Forest
    rf = models["random_forest"]
    importances = list(rf.feature_importances_)
    feature_importance = [{"feature": f, "importance": float(i)} for f, i in zip(features, importances)]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    with open("ml_models/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=4)
        
    print("Training complete. Metrics saved.")
    return metrics

if __name__ == "__main__":
    metrics = train_models()
    for name, m in metrics.items():
        print(f"{name}: R2={m['r2']:.4f}, MAE={m['mae']:.2f}")
