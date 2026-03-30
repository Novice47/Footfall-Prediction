import pandas as pd
import numpy as np
import datetime
import calendar
from scipy.stats import norm

# Sites Data
SITES = [
    {"id": 1, "name": "Taj Mahal", "country": "India", "lat": 27.1751, "lon": 78.0421, "base": 35000},
    {"id": 2, "name": "Eiffel Tower", "country": "France", "lat": 48.8584, "lon": 2.2945, "base": 50000},
    {"id": 3, "name": "Great Wall of China", "country": "China", "lat": 40.4319, "lon": 116.5704, "base": 45000},
    {"id": 4, "name": "Colosseum", "country": "Italy", "lat": 41.8902, "lon": 12.4922, "base": 42000},
    {"id": 5, "name": "Machu Picchu", "country": "Peru", "lat": -13.1631, "lon": -72.5450, "base": 25000},
    {"id": 6, "name": "Christ the Redeemer", "country": "Brazil", "lat": -22.9519, "lon": -43.2105, "base": 28000},
    {"id": 7, "name": "Chichen Itza", "country": "Mexico", "lat": 20.6843, "lon": -88.5678, "base": 22000},
    {"id": 8, "name": "Petra", "country": "Jordan", "lat": 30.3285, "lon": 35.4444, "base": 20000},
    {"id": 9, "name": "Angkor Wat", "country": "Cambodia", "lat": 13.4125, "lon": 103.8670, "base": 30000},
    {"id": 10, "name": "Stonehenge", "country": "UK", "lat": 51.1789, "lon": -1.8262, "base": 12000},
]

def get_season(month):
    if month in [12, 1, 2]: return 1 # Winter
    if month in [3, 4, 5]: return 2  # Spring
    if month in [6, 7, 8]: return 3  # Summer
    if month in [9, 10, 11]: return 4 # Autumn
    return 1

def generate_data(n_rows=50000):
    np.random.seed(42)
    data = []
    
    # Pre-calculated site info
    site_map = {s['id']: s for s in SITES}
    
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", periods=n_rows)
    
    for i in range(n_rows):
        site = np.random.choice(SITES)
        date = dates[i]
        hour = np.random.randint(6, 22) # Sites closed at night
        day_of_week = date.dayofweek
        month = date.month
        year = date.year
        
        season = get_season(month)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Simulated Holiday (Simplified: same for all countries but includes major dates)
        is_holiday = 1 if (month == 12 and date.day in [25, 31]) or (month == 1 and date.day == 1) or np.random.rand() < 0.05 else 0
        
        # Weather
        # Taj Mahal (India): hot
        if site['name'] == "Taj Mahal":
            base_temp = [15, 20, 30, 35, 40, 42, 38, 35, 33, 30, 25, 18][month-1]
            hum_val = [40, 35, 30, 25, 20, 40, 70, 75, 70, 50, 45, 45][month-1]
        elif site['name'] == "Stonehenge":
            base_temp = [5, 6, 9, 12, 15, 18, 20, 19, 16, 13, 9, 6][month-1]
            hum_val = [80, 78, 75, 70, 70, 72, 75, 78, 80, 82, 85, 85][month-1]
        else:
            base_temp = 20 + 10 * np.sin(2 * np.pi * (month/12))
            hum_val = 60 + 20 * np.cos(2 * np.pi * (month/12))
            
        temp = base_temp + np.random.normal(0, 3)
        humidity = np.clip(hum_val + np.random.normal(0, 5), 0, 100)
        
        # Missing values (1%)
        if np.random.rand() < 0.01: temp = np.nan
        if np.random.rand() < 0.01: humidity = np.nan
        
        # Factors
        h_f = np.exp(-0.5 * ((hour - 13) / 3)**2) # Bell curve peaking at 13:00
        w_f = 1.35 if is_weekend else 1.0
        hol_f = 1.6 if is_holiday else 1.0
        if is_holiday and month == 12: hol_f = 1.9 # Christmas/New Year
        
        # seasonal factor
        if site['name'] == "Taj Mahal":
            s_f = 1.3 if season == 1 else (1.1 if season in [2, 4] else 0.75)
        else:
            s_f = 1.4 if season == 3 else (1.1 if season in [2, 4] else 0.75)
            
        # weather factor
        weath_f = 1.0
        current_temp = temp if not np.isnan(temp) else base_temp
        current_hum = humidity if not np.isnan(humidity) else hum_val
        if current_hum > 75: weath_f -= (current_hum - 75) * 0.01
        if current_temp > 38: weath_f -= 0.25
        
        # Year growth/COVID factor
        y_g_f = {2020: 1.0, 2021: 0.5, 2022: 0.85, 2023: 1.1, 2024: 1.2}[year]
        # COVID dip (Mar 2020 - Jun 2021)
        if (year == 2020 and month >= 3) or (year == 2021 and month <= 6):
            y_g_f *= np.random.uniform(0.15, 0.40)
            
        # Base footfall logic
        footfall = site['base'] * h_f * w_f * hol_f * s_f * weath_f * y_g_f
        
        # Noise
        footfall *= (1 + np.random.normal(0, 0.08))
        
        # Outliers (2%)
        event_flag = 0
        if np.random.rand() < 0.02:
            footfall *= np.random.uniform(2.0, 3.0)
            event_flag = 1
            
        footfall = int(max(0, footfall))
        
        # Signals
        mobile = int(footfall * 0.72 + np.random.normal(0, footfall * 0.05))
        wifi = int(footfall * 0.28 + np.random.normal(0, footfall * 0.03))
        
        visibility = np.clip(25 + np.random.normal(0, 10), 5, 50)
        footfall *= (1 + (visibility - 25) / 25 * 0.05)
        
        data.append({
            "place_id": site['id'],
            "place_name": site['name'],
            "country": site['country'],
            "latitude": site['lat'],
            "longitude": site['lon'],
            "date": date.strftime("%Y-%m-%d"),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "season": season,
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "temperature": temp,
            "humidity": humidity,
            "mobile_signals": mobile,
            "wifi_connections": wifi,
            "footfall": int(footfall),
            "event_flag": event_flag,
            "visibility_km": visibility
        })
        
    df = pd.DataFrame(data)
    
    # Add prev_hour_footfall (sort by place and date/hour first)
    df['dt'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df.sort_values(['place_id', 'dt'])
    df['prev_hour_footfall'] = df.groupby('place_id')['footfall'].shift(1).fillna(0).astype(int)
    
    df.drop(columns=['dt'], inplace=True)
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_data(50000)
    df.to_csv("data/heritage_footfall_dataset.csv", index=False)
    print("Dataset saved to data/heritage_footfall_dataset.csv")
    
    # Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Total rows: {len(df)}")
    print("\nFootfall range per site:")
    print(df.groupby('place_name')['footfall'].agg(['min', 'max', 'mean']))
    
    print("\nCorrelation matrix (selected features):")
    cols = ['mobile_signals', 'wifi_connections', 'temperature', 'humidity', 'footfall']
    print(df[cols].corr()['footfall'])
    
    print("\nMean footfall by season:")
    print(df.groupby('season')['footfall'].mean())
    
    print("\nMean footfall by hour:")
    print(df.groupby('hour')['footfall'].mean())
