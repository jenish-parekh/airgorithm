# app.py

import streamlit as st
from air_api.load_model import load_model, process_image
from PIL import Image
import numpy as np
import pandas as pd

st.title("Détection de dommages sur avions")
st.write("Chargez vos images pour détecter automatiquement les dommages.")

# Charger le modèle avec mise en cache pour éviter de le recharger à chaque interaction
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Uploader multiple d'images
uploaded_files = st.file_uploader("Charger des images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    global_detections = []  # Liste pour stocker les détections de toutes les images

    # Traitement de chaque image uploadée
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        results, detections = process_image(model, image)

        st.subheader(f"Résultats pour {uploaded_file.name}")
        # Affichage de l'image annotée
        annotated_image = results[0].plot()  # Retourne un array numpy avec les bounding boxes dessinées
        st.image(annotated_image, caption=uploaded_file.name, use_column_width=True)

        # Affichage des détections sous forme de tableau
        if detections:
            df = pd.DataFrame([{
                "Fichier": uploaded_file.name,
                "Classe": det["class"],
                "Confiance": f"{det['confidence']:.3f}",
                "Bounding Box": f"({det['bounding_box'][0]:.0f}, {det['bounding_box'][1]:.0f}, {det['bounding_box'][2]:.0f}, {det['bounding_box'][3]:.0f})"
            } for det in detections])
            st.table(df)
            global_detections.extend([{
                "Fichier": uploaded_file.name,
                "Classe": det["class"],
                "Confiance": det["confidence"]
            } for det in detections])
        else:
            st.info("Aucune détection sur cette image.")

    # Synthèse globale
    if global_detections:
        st.subheader("Synthèse des détections globales")
        df_all = pd.DataFrame(global_detections)
        st.write("Détails de toutes les détections sur l'ensemble des images :")
        st.dataframe(df_all)

        st.write("Nombre de détections par classe :")
        df_grouped = df_all.groupby("Classe").size().reset_index(name="Nombre de détections")
        st.table(df_grouped)
else:
    st.info("Veuillez charger au moins une image.")
