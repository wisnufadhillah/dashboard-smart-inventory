from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
# Note: Idealnya scaler lu di-load pakai library joblib/pickle, 
# tapi buat contoh ini kita pakai simulasi biar lu kebayang flow-nya.

app = FastAPI(
    title="API Smart Inventory UMKM",
    description="REST API untuk melayani model Deep Learning prediksi stok",
    version="1.0"
)

# Load model AI lu (Pastiin file .keras-nya ada di folder yang sama)
try:
    model = tf.keras.models.load_model('model_inventory_umkm.keras')
except:
    model = None

# Format data yang bakal diterima API lu (30 hari riwayat penjualan)
class DataPenjualan(BaseModel):
    history_30_hari: list[float]

@app.get("/")
def home():
    return {"message": "API Smart Inventory UMKM is Running! Gas Bol!"}

@app.post("/predict")
def predict_stock(data: DataPenjualan):
    if model is None:
         return {"error": "Model AI belum di-load, cek file .keras lu!"}
    
    # 1. Siapin data history (convert ke numpy)
    input_data = np.array(data.history_30_hari)
    
    # 2. Pura-puranya ini udah di-scaling (MinMaxScaler) & di-reshape buat LSTM (1, 30, 1)
    input_reshaped = input_data.reshape((1, 30, 1))
    
    # 3. Suruh model nebak
    hasil_prediksi = model.predict(input_reshaped)
    
    # 4. Balikin hasilnya ke user
    # (Ini hasil mentah, nanti lu sesuaikan sama scaler inverse_transform lu)
    prediksi_final = int(hasil_prediksi[0][0]) 
    
    return {
        "status": "success",
        "prediksi_besok": prediksi_final,
        "pesan": "Inference sukses dijalankan via FastAPI."
    }