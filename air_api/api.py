from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from air_api.load_model import load_model, process_image
from PIL import Image
import io
import base64
from io import BytesIO

app = FastAPI()

# Charger le modèle une seule fois lors du démarrage
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour lancer la détection sur une image uploadée.
    """
    # Lire et convertir l'image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Traiter l'image pour obtenir les résultats et les détections
    results, detections = process_image(model, image)

    # Générer l'image annotée (ex: en dessinant les bounding boxes)
    annotated_array = results[0].plot()  # retourne un array numpy
    annotated_image = Image.fromarray(annotated_array)

    # Convertir l'image annotée en chaîne base64
    buffered = BytesIO()
    annotated_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Retourner les détections et l'image annotée encodée
    return JSONResponse(content={"detections": detections, "annotated_image": img_base64})
