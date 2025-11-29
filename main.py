from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
from xgboost import XGBRegressor
import os

app = FastAPI()




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

locations_sorted = sorted(location_map.keys())




templates = Jinja2Templates(directory="templates")





@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Loads the HTML UI with dropdown values.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "locations": locations_sorted}
    )


@app.post("/predict")
async def predict(
    location: str = Form(...),
    bhk: int = Form(...),
    bath: int = Form(...),
    total_sqft: float = Form(...)
):
    """
    Predict price using the ML model, return JSON.
    """

    
    loc = location_map.get(location, location_map["other"])

    
    has_society = 1
    balcony = 1
    price_per_sqft = 5000

    
    features = np.array([[
        total_sqft,
        bath,
        balcony,
        has_society,
        price_per_sqft,
        bhk,
        0, 0, 0, 0,  
        loc
    ]])

    pred = float(model.predict(features)[0])

    return JSONResponse({"predicted_price": round(pred, 2)})
