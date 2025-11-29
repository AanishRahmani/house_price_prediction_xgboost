from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from xgboost import XGBRegressor

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = XGBRegressor()
model.load_model("/home/aanish/Desktop/ML_projects/b_house_price_prediction/trail.json")


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




# Request model
class HouseData(BaseModel):
    total_sqft: float
    bath: float
    balcony: float
    has_society: int
    price_per_sqft: float
    bhk: int
    area_type_Other: int
    area_type_Plot_Area: int
    area_type_Super_builtup_Area: int
    availability_Soon: int
    location_target_enc: float

@app.get("/")
def hello():
    return {"message": "House Price Prediction API is running"}
@app.post("/predict")
def predict_price(data: PredictInput):
    loc = data.location.strip()

    # if location not in map, use 'other'
    if loc not in location_map:
        loc = "other"

    loc_enc = location_map[loc]

    # Your derived features
    has_society = 1
    balcony = 1
    price_per_sqft = 5000  # approx, used only to fill feature vector

    # Construct full feature vector for model
    features = np.array([[
        data.total_sqft,
        data.bath,
        balcony,
        has_society,
        price_per_sqft,
        data.bhk,
        0, 0, 0, 0,   # area types + availability
        loc_enc
    ]])

    pred = model.predict(features)[0]
    return {"price_lakhs": float(pred)}
