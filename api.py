from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import tensorflow as tf
import zipfile
from pathlib import Path
# Note: Idealnya scaler lu di-load pakai library joblib/pickle, 
# tapi buat contoh ini kita pakai simulasi biar lu kebayang flow-nya.

app = FastAPI(
    title="API Smart Inventory UMKM",
    description="REST API untuk melayani model Deep Learning prediksi stok",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_inventory_umkm.keras"
SANITIZED_MODEL_PATH = BASE_DIR / "model_inventory_umkm_sanitized.keras"
MODEL_LOAD_ERROR = None


def remove_unsupported_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for child in value.values():
            remove_unsupported_config(child)
    elif isinstance(value, list):
        for item in value:
            remove_unsupported_config(item)
    return value


def prepare_compatible_model_file():
    if (
        SANITIZED_MODEL_PATH.exists()
        and SANITIZED_MODEL_PATH.stat().st_mtime >= MODEL_PATH.stat().st_mtime
    ):
        return SANITIZED_MODEL_PATH

    with zipfile.ZipFile(MODEL_PATH, "r") as source:
        with zipfile.ZipFile(SANITIZED_MODEL_PATH, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(content.decode("utf-8"))
                    content = json.dumps(remove_unsupported_config(config)).encode("utf-8")
                target.writestr(item, content)
    return SANITIZED_MODEL_PATH


# Load model AI lu dari folder yang sama dengan api.py
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
except Exception as e:
    try:
        compatible_model_path = prepare_compatible_model_file()
        model = tf.keras.models.load_model(
            compatible_model_path,
            compile=False,
            safe_mode=False,
        )
    except Exception as fallback_error:
        model = None
        MODEL_LOAD_ERROR = f"{e}\n\nFallback error: {fallback_error}"

# Format data yang bakal diterima API lu (30 hari riwayat penjualan)
class DataPenjualan(BaseModel):
    history_30_hari: list[float]

@app.get("/")
def home():
    return {
        "message": "API Smart Inventory UMKM is Running! Gas Bol!",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
        "model_load_error": MODEL_LOAD_ERROR,
    }

@app.post("/predict")
def predict_stock(data: DataPenjualan):
    if model is None:
         return {
             "error": "Model AI belum di-load.",
             "model_path": str(MODEL_PATH),
             "detail": MODEL_LOAD_ERROR,
         }
    
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
