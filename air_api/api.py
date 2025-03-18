# api.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from load_model import load_model, process_image
from PIL import Image
import io

app = FastAPI()

# Charger le modèle une seule fois lors du démarrage
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour lancer la détection sur une image uploadée.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results, detections = process_image(model, image)
    return JSONResponse(content={"detections": detections})
