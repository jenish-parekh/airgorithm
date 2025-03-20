import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import datetime

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("images/logo_AirGorithm_-_option_2a-removebg-previewv1.png")

st.markdown("""
# Welcome to **AirGorithm**
## This platform helps you to identify damages on aircraft's bodies and classify the type of damages.
### Select a method to upload the pictures taken by the drone:
""")

airline_name = st.text_input("Airline name", "Air France")
st.write('You have entered the following airline', airline_name)

description = st.text_area('Audit description', '''Comprehensive audit on a selected fleet using both traditional inspection
methods and an advanced AI-powered damage detection system.
The process leverages the AirGorithm that pre-screens aircraft images to
automatically identify and assess surface damages such as dents, cracks, and
scratches. By pre-identifying potential issues, our expert inspectors can focus
their manual checks more effectively, saving significant time and enhancing
overall accuracy.

In our recent pilot audit, the AI system pre-screened a fleet of 150 aircraft
spanning 10 different models, flagging over 1,200 potential damage instances.
This preliminary detection stage reduced manual inspection hours by
approximately 40% and increased damage detection accuracy by nearly 30%
compared to conventional methods alone. These promising results underscore the
transformative potential of integrating AI-driven pre-screening into aircraft
maintenance workflows for more efficient and reliable operations.''')
st.write("Length:", len(description))

today = datetime.date.today()
d = st.date_input("Select the date of the audit", today)
st.write('You selected', d)

col1, col2 = st.columns([3, 1])
with col1:
    direction = st.radio('Select an upload method', ('Upload manually', 'Connect the drone'))
with col2:
    icon_html = """
    <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
        <span style="font-size: 50px;">{}</span>
    </div>
    """.format("🤖" if direction == 'Connect the drone' else "💾")
    st.markdown(icon_html, unsafe_allow_html=True)

st.write(f"Selected option: {direction}")

if direction == "Upload manually":
    uploaded_files = st.file_uploader("Charger des images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        global_detections = []  # Pour stocker toutes les détections
        api_url = "http://localhost:8000/predict/"  # URL de l'API

        for uploaded_file in uploaded_files:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(api_url, files=files)
            if response.status_code == 200:
                result_json = response.json()
                detections = result_json.get("detections", [])
                annotated_image_b64 = result_json.get("annotated_image", "")

                st.subheader(f"Résultats pour {uploaded_file.name}")
                if annotated_image_b64:
                    # Décoder l'image annotée
                    img_bytes = base64.b64decode(annotated_image_b64)
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption=uploaded_file.name, use_container_width=True)
                else:
                    st.info("Aucune image annotée renvoyée par l'API.")

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
                # else:
                #     st.info("Aucune détection sur cette image.")
            else:
                st.error(f"Erreur de prédiction pour {uploaded_file.name} : {response.status_code}")

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
else:
    st.image("images/457d9577a9fb58d83460e794bef5f859.gif")
