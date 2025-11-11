import pandas as pd
import numpy as np
import joblib

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

    for col, le in encoders.items():
        if col in df.columns:
            unseen = set(df[col]) - set(le.classes_)
            if unseen:
                le.classes_ = np.append(le.classes_, list(unseen))
            df[col] = le.transform(df[col])

    df = df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(df)
    return prediction[0]
