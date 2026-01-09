from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
import os

app = FastAPI(title="Stock Prediction API", description="API para previsão de ações usando LSTM")

# Carregar modelo e scaler ao iniciar
# Verifica se os arquivos existem antes de carregar
if os.path.exists("lstm_stock_model.h5") and os.path.exists("scaler.pkl"):
    model = load_model("lstm_stock_model.h5")
    scaler = joblib.load("scaler.pkl")
else:
    model = None
    scaler = None
    print("AVISO: Modelo ou Scaler não encontrados. Execute train_model.py primeiro.")

class StockData(BaseModel):
    last_60_days: list[float]

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/predict")
def predict(data: StockData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    
    input_data = data.last_60_days
    
    if len(input_data) != 60:
        raise HTTPException(status_code=400, detail="A lista deve conter exatamente 60 preços.")
    
    try:
        input_array = np.array(input_data).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        final_input = scaled_input.reshape(1, 60, 1)
        
        prediction = model.predict(final_input)
        predicted_price = scaler.inverse_transform(prediction)
        
        return {"prediction": float(predicted_price[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "online", "usage": "Envie um POST para /predict com os últimos 60 preços."}