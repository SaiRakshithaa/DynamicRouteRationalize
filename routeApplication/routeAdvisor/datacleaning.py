import requests
import math
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os
from xgboost import XGBRegressor



class Api_Data:
    def __init__(self, coordinate):
        self.co = coordinate  # (lat, lon)

    def get_weather_data(self):
        """Fetch current weather data."""
        ENDPOINT = 'https://api.openweathermap.org/data/2.5/weather'
        parameters = {
            'lat': self.co[0],
            'lon': self.co[1],
            'appid': '0794673555559a129540662e3029b866',
            'units': 'metric',
        }

        try:
            resp = requests.get(ENDPOINT, params=parameters, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            main = data.get('main', {})
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            weather = data.get('weather', [{}])[0]
            rain = data.get('rain', {}).get('1h', 0)
            snow = data.get('snow', {}).get('1h', 0)

            humidity = main.get('humidity', 0)
            wind_speed = wind.get('speed', 0)
            wind_direction = wind.get('deg', 0)
            visibility_miles = round(data.get('visibility', 0) / 1609.34, 2)
            temperature_c = main.get('temp', 0)
            temperature_f = round((temperature_c * 9 / 5) + 32, 2)

            # Dew Point (Magnus formula)
            a, b = 17.27, 237.7
            alpha = ((a * temperature_c) / (b + temperature_c)) + math.log(max(humidity, 1) / 100.0)
            dew_point = round((b * alpha) / (a - alpha), 2)

            clouds_all = clouds.get('all', 0)
            weather_type = weather.get('main', "unknown")
            weather_description = weather.get('description', "none")

            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            is_holiday = self.is_holiday()
            is_weekend = self.is_weekend()

            return {
                "date_time": date_time,
                "is_holiday": is_holiday,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "visibility_in_miles": visibility_miles,
                "dew_point": dew_point,
                "temperature": temperature_f,
                "rain_p_h": rain,
                "snow_p_h": snow,
                "clouds_all": clouds_all,
                "weather_type": weather_type,
                "weather_description": weather_description,
                "is_weekend": is_weekend,
            }

        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

    def get_forecast_data(self, target_datetime):
        """Fetch forecast for a given future date & time (within 5 days)."""
        ENDPOINT = 'https://api.openweathermap.org/data/2.5/forecast'
        parameters = {
            'lat': self.co[0],
            'lon': self.co[1],
            'appid': '0794673555559a129540662e3029b866',
            'units': 'metric',
        }

        try:
            resp = requests.get(ENDPOINT, params=parameters, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            forecasts = data.get('list', [])
            if not forecasts:
                raise ValueError("No forecast data available")

            target = datetime.datetime.strptime(target_datetime, "%Y-%m-%d %H:%M:%S")
            closest = min(forecasts, key=lambda x: abs(datetime.datetime.fromtimestamp(x['dt']) - target))

            main = closest.get('main', {})
            wind = closest.get('wind', {})
            clouds = closest.get('clouds', {})
            weather = closest.get('weather', [{}])[0]
            rain = closest.get('rain', {}).get('3h', 0)
            snow = closest.get('snow', {}).get('3h', 0)

            humidity = main.get('humidity', 0)
            wind_speed = wind.get('speed', 0)
            wind_direction = wind.get('deg', 0)
            visibility_miles = round(closest.get('visibility', 0) / 1609.34, 2) if closest.get('visibility') else 0
            temperature_c = main.get('temp', 0)
            temperature_f = round((temperature_c * 9 / 5) + 32, 2)

            a, b = 17.27, 237.7
            alpha = ((a * temperature_c) / (b + temperature_c)) + math.log(max(humidity, 1) / 100.0)
            dew_point = round((b * alpha) / (a - alpha), 2)

            clouds_all = clouds.get('all', 0)
            weather_type = weather.get('main', "unknown")
            weather_description = weather.get('description', "none")

            is_holiday = self.is_holiday()
            is_weekend = target.weekday() >= 5

            return {
                "date_time": target_datetime,
                "is_holiday": is_holiday,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "visibility_in_miles": visibility_miles,
                "dew_point": dew_point,
                "temperature": temperature_f,
                "rain_p_h": rain,
                "snow_p_h": snow,
                "clouds_all": clouds_all,
                "weather_type": weather_type,
                "weather_description": weather_description,
                "is_weekend": is_weekend,
            }

        except Exception as e:
            print(f"Error fetching forecast data: {e}")
            return None

    def is_holiday(self):
        year = datetime.datetime.now().year
        ENDPOINT = 'https://calendarific.com/api/v2/holidays'
        parameters = {
            'api_key': '2e72745fd6562bd393296f14baacf18f92b08b3a',
            'country': 'US',
            'year': year,
        }
        try:
            resp = requests.get(ENDPOINT, params=parameters, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            holidays = data.get('response', {}).get('holidays', [])
            today = datetime.datetime.now().strftime('%Y-%m-%d')

            for holiday in holidays:
                date = holiday.get('date', {}).get('iso', '')[:10]
                if date == today:
                    return True
            return False
        except:
            return False

    def is_weekend(self):
        today = datetime.datetime.now().strftime('%A')
        return today in ['Saturday', 'Sunday']



data_path = r"D:\Rakshu\DynamicRouteRationalize\routeApplication\routeAdvisor\Train.csv"

if not os.path.exists("best_traffic_model.pkl"):
    print("Training model...")

    data = pd.read_csv(data_path)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['dayofweek'] = data['date_time'].dt.dayofweek
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    data['is_weekend'] = data['dayofweek'] >= 5

    # Convert boolean-like columns
    data['is_holiday'] = data['is_holiday'].astype(bool).astype(int) if 'is_holiday' in data.columns else 0

    X = data.drop(columns=['date_time', 'traffic_volume', 'air_pollution_index'], errors='ignore')
    y = data['traffic_volume']

    # Clean and encode
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].fillna('none').astype(str).str.strip().str.lower()
    for col in X.select_dtypes(exclude='object').columns:
        X[col] = X[col].fillna(X[col].median())

    label_cols = ['weather_type', 'weather_description']
    encoders = {}
    for col in label_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

    joblib.dump(model, "best_traffic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")
    print("Model trained and saved successfully.")
else:
    print("Model already trained and saved. Skipping training.")



def predict_from_api(api_data_dict):
    model = joblib.load("best_traffic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    df = pd.DataFrame([api_data_dict])
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(int)
    df = df.drop(columns=['date_time'], errors='ignore')

    for col in encoders.keys():
        if col in df.columns:
            le = encoders[col]
            unseen = set(df[col]) - set(le.classes_)
            if unseen:
                le.classes_ = np.append(le.classes_, list(unseen))
            df[col] = le.transform(df[col])

    df = df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(df)
    return prediction[0]



if __name__ == "__main__":
    coordinates = (40.7128, -74.0060)
    api = Api_Data(coordinates)

    target_datetime = "2025-11-05 15:00:00"
    weather_data = api.get_forecast_data(target_datetime)
    print("\nFetched API Data:")
    print(weather_data)

    if weather_data:
        predicted_volume = predict_from_api(weather_data)
        print(f"\nPredicted Traffic Volume at {target_datetime}: {predicted_volume:.2f}")
