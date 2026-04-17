import xgboost as xgb
import pandas as pd
import numpy as np

def train_trend_predictor(historical_data):
    # 1. Feature Engineering
    df = historical_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['target'] = df['search_volume'].shift(-1)
    
    train_df = df.dropna()
    X = train_df[['search_volume', 'month']]
    y = train_df['target']

    # 2. Train Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    
    # 3. 6-Month Recursive Forecast
    forecast = []
    current_val = df['search_volume'].iloc[-1]
    current_month = df['month'].iloc[-1]

    for i in range(1, 7):
        next_month = (current_month + i - 1) % 12 + 1
        # Manual prediction for the next step
        pred = model.predict(pd.DataFrame([[current_val, next_month]], columns=['search_volume', 'month']))
        forecast.append({"month": i, "value": round(float(pred[0]), 2)})
        current_val = pred[0]

    return forecast