# load_model.py

from ultralytics import YOLO
import numpy as np

# Chemin vers le modèle entraîné
MODEL_PATH = "models/best.pt"

def load_model():
    """
    Charge et retourne le modèle YOLO.
    """
    model = YOLO(MODEL_PATH)
    return model

def process_image(model, image):
    """
    Prend en entrée une image (PIL.Image), lance la détection et retourne
    l'objet 'results' de YOLO ainsi qu'une liste de détections.
    Chaque détection est un dictionnaire contenant la classe, la confiance et la bounding box.
    """
    # Conversion de l'image en array pour le modèle
    image_np = np.array(image)
    results = model.predict(image_np)

    detections = []
    # Extraction alternative en utilisant results[0].boxes.data
    if hasattr(results[0].boxes, "data"):
        # On convertit les données en tableau NumPy (en passant par CPU si nécessaire)
        try:
            boxes = results[0].boxes.data.cpu().numpy()
        except AttributeError:
            boxes = results[0].boxes.data.numpy()
        # Chaque 'box' contient [x1, y1, x2, y2, score, class]
        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box
            class_name = results[0].names[int(cls_id)]
            detections.append({
                "class": class_name,
                "confidence": float(score),
                "bounding_box": (x1, y1, x2, y2)
            })
    return results, detections
