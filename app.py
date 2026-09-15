import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from src.pipeline.prediction import PredictionPipeline
import subprocess

app = FastAPI(
    title="Financial Sentiment API", 
    description="Zero-shot FinBERT classification for financial text",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

@app.get("/", tags=["General"])
async def index():
    return {"message": "Financial Sentiment API is running. Visit /docs for the Swagger UI."}

@app.get("/train", tags=["Training Pipeline"])
async def trigger_training():
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        return {"message": "Pipeline executed successfully. Check DagsHub for metrics."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

@app.post("/predict", tags=["Prediction Pipeline"])
async def predict_sentiment(input_data: TextInput):
    try:
        predictor = PredictionPipeline()
        result = predictor.predict(input_data.text)
        
        return {
            "text": input_data.text,
            "sentiment": result["label"],
            "confidence": round(result["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)