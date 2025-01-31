from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Google Generative AI API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment")

genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI app setup
app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for matches
class Match(BaseModel):
    team1: str
    team2: str

class MatchRequest(BaseModel):
    matches: list[Match]

# Prediction endpoint
@app.post("/predict")
async def predict_matches(request: MatchRequest):
    """
    Predict the upcoming outcome of soccer matches using Google Generative AI.
    """
    matches = request.matches

    if not matches:
        raise HTTPException(status_code=400, detail="No matches provided for prediction.")

    predictions = []
    for match in matches:
        try:
            # Use Generative AI to predict the match outcome
            input_prompt = (
                f"Predict the upcoming outcome of the soccer match between {match.team1} and {match.team2}. "
                f"Provide latest detailed stats including odds, last 5 games, and league position, and predict the winner."
            )
            model = genai.GenerativeModel("gemini-pro")  # Use a supported model
            prediction_result = model.generate_content(input_prompt)

            # Extract the prediction text
            prediction_text = prediction_result.text.strip()
            predictions.append({
                "team1": match.team1,
                "team2": match.team2,
                "prediction": prediction_text,
            })

        except Exception as e:
            predictions.append({
                "team1": match.team1,
                "team2": match.team2,
                "error": f"Error generating prediction: {str(e)}",
            })

    return {"predictions": predictions}

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the service is running properly.
    """
    return {"status": "ok", "message": "Prediction API is healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the King Dave Prediction API!"}
