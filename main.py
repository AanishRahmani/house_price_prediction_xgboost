from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
import numpy as np
from xgboost import XGBRegressor
import os

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




MODEL_PATH = os.path.join(os.path.dirname(__file__), "trail.json")

model = XGBRegressor()
model.load_model(MODEL_PATH)




location_map = {
    "other": 7214,
    "Whitefield": 533,
    "Sarjapur Road": 392,
    "Electronic City": 303,
    "Kanakpura Road": 264,
    "Thanisandra": 235,
    "Yelahanka": 210,
    "Uttarahalli": 186,
    "Hebbal": 176,
    "Marathahalli": 175,
    "Raja Rajeshwari Nagar": 171,
    "Bannerghatta Road": 151,
    "Hennur Road": 150,
    "7th Phase JP Nagar": 148,
    "Haralur Road": 141,
    "Electronic City Phase II": 131,
    "Rajaji Nagar": 105,
    "Chandapura": 98,
    "Bellandur": 96,
    "Hoodi": 88,
    "Electronics City Phase 1": 87,
    "KR Puram": 87,
    "Yeshwanthpur": 85,
    "Begur Road": 84,
    "Sarjapur": 80,
    "Kasavanhalli": 79,
    "Harlur": 79,
    "Banashankari": 74,
    "Hormavu": 74,
    "Ramamurthy Nagar": 72,
    "Koramangala": 72,
    "Kengeri": 72,
    "Old Madras Road": 70,
    "Varthur": 70,
    "Hosa Road": 69,
    "Jakkur": 68,
    "JP Nagar": 66,
    "Kothanur": 65,
    "Kaggadasapura": 64,
    "Nagarbhavi": 63,
    "Thigalarapalya": 62,
    "Akshaya Nagar": 62,
    "TC Palaya": 60,
    "Rachenahalli": 58,
    "Malleshwaram": 57,
    "8th Phase JP Nagar": 57,
    "Budigere": 54,
    "HSR Layout": 53,
    "Jalahalli": 52,
    "Hennur": 52,
    "Bisuvanahalli": 51,
    "Hulimavu": 51,
    "Panathur": 51
}




class PredictInput(BaseModel):
    location: str
    bhk: int
    bath: int
    total_sqft: float






@app.get("/", include_in_schema=False)
def redirect_root():
    return RedirectResponse(url="/predict")


@app.get("/predict", response_class=HTMLResponse)
def load_predict_page():
    index_file = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_file, "r") as f:
        return f.read()


@app.post("/predict")
def predict_price(data: PredictInput):
    loc = data.location.strip()

    
    if loc not in location_map:
        loc = "other"

    loc_enc = location_map[loc]

    
    has_society = 1
    balcony = 1
    price_per_sqft = 5000

    
    features = np.array([[
        data.total_sqft,
        data.bath,
        balcony,
        has_society,
        price_per_sqft,
        data.bhk,
        0, 0, 0, 0,   
        loc_enc
    ]])

    pred = model.predict(features)[0]

    return {"price_lakhs": float(pred)}
