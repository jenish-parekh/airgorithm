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
    if results[0].boxes and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy  # coordonnées au format [xmin, ymin, xmax, ymax]
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        for i in range(len(boxes)):
            box = boxes[i]
            conf = confidences[i]
            cls_id = int(class_ids[i])
            class_name = results[0].names[cls_id]
            detections.append({
                "class": class_name,
                "confidence": float(conf),
                "bounding_box": (box[0], box[1], box[2], box[3])
            })
    return results, detections
