import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
 
# ---------------------------
# Load model and preprocessor
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model.keras")
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor
 
model, preprocessor = load_artifacts()
 
st.set_page_config(page_title="Spotify Hit Song Predictor", page_icon="🎵", layout="centered")
 
st.title("🎵 Spotify Hit Song Predictor")
st.write("Enter a song's audio features to predict whether it is likely to become a **Hit** (top 25% popularity).")
 
# ---------------------------
# Input fields
# ---------------------------
st.subheader("Audio Features")
 
col1, col2 = st.columns(2)
 
with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    loudness = st.slider("Loudness (dB)", -60.0, 5.0, -10.0, 0.1)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
 
with col2:
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
    valence = st.slider("Valence (positivity)", 0.0, 1.0, 0.5, 0.01)
    tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0, 1.0)
    key = st.selectbox("Key", list(range(0, 12)), index=0)
    mode = st.selectbox("Mode (0=Minor, 1=Major)", [0, 1], index=1)
    time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
 
explicit = st.checkbox("Explicit content")
duration_min = st.slider("Duration (minutes)", 0.5, 15.0, 3.5, 0.1)
track_genre = st.text_input("Track Genre", value="pop")
 
# ---------------------------
# Build input dataframe
# ---------------------------
input_dict = {
    "explicit": explicit,
    "danceability": danceability,
    "energy": energy,
    "key": key,
    "loudness": loudness,
    "mode": mode,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "time_signature": time_signature,
    "track_genre": track_genre,
    "duration_min": duration_min,
    "energy_valence_ratio": energy / (valence + 0.001),
}
 
input_df = pd.DataFrame([input_dict])
 
# ---------------------------
# Predict
# ---------------------------
if st.button("Predict Hit Probability", type="primary"):
    try:
        X_input = preprocessor.transform(input_df)
        prob = float(model.predict(X_input)[0][0])
        prediction = "🔥 HIT" if prob > 0.5 else "Not a Hit"
 
        st.subheader("Result")
        st.metric("Hit Probability", f"{prob*100:.2f}%")
 
        if prob > 0.5:
            st.success(f"Prediction: {prediction}")
        else:
            st.warning(f"Prediction: {prediction}")
 
        st.progress(min(max(prob, 0.0), 1.0))
 
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that the input columns match what the preprocessor was trained on.")
 
st.markdown("---")
st.caption("Model: ANN tuned with Optuna | Trained on Spotify Tracks Dataset (Kaggle)")