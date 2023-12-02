from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.models.predict_model import PredictionPipeline

query: str = "What is 2+2?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        # os.system("dvc repro")
        return Response("Training done successfully!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(query):
    try:
        obj = PredictionPipeline()
        response = obj.predict(query)
        return response
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
