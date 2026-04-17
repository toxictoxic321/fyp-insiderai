from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from trend_model import train_trend_predictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/predict/{keyword}")
async def get_prediction(keyword: str):
    try:
        count = 24
        
        # 1. Manually generate exactly 24 monthly timestamps
        # This avoids the "23 vs 24" error by creating them one by one
        dates = []
        current_date = datetime.now()
        for i in range(count):
            # Move back one month at a time
            date_entry = current_date - timedelta(days=30 * i)
            dates.append(date_entry)
        
        # Reverse them so they go from past to present
        dates.reverse()
        
        # 2. Generate exactly 24 random numbers
        search_volume = [np.random.randint(40, 95) for _ in range(count)]

        # 3. Create DataFrame
        historical_df = pd.DataFrame({
            'date': dates, 
            'search_volume': search_volume
        })

        # 4. Run XGBoost Prediction
        forecast = train_trend_predictor(historical_df)
        
        return {
            "keyword": keyword,
            "historical": historical_df.tail(6).to_dict(orient='records'),
            "forecast": forecast,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error Details: {e}")
        return {"status": "error", "message": str(e)}