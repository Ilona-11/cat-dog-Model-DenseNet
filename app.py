import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- KONFIGURASI TEMA HIJAU ARMY ---
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #4B5320; /* Army Green */
        color: #F5F5DC; /* Cream */
    }
    [data-testid="stSidebar"] {
        background-color: #353D16; /* Darker Green */
    }
    .stMarkdown, h1, h2, h3, p {
        color: #F5F5DC !important;
    }
    .stButton>button {
        background-color: #606E32;
        color: white;
        border: 1px solid #DDE2C6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    # Mengarah ke folder 'models' sesuai struktur Anda
    model_path = 'models/model_densenet.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model tidak ditemukan di {model_path}. Pastikan file .h5 sudah ada.")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Pengaturan")
    st.write("Model: **DenseNet121**")
    rescale = st.checkbox("Rescale /255 (Normalisasi)", value=True)

# --- MAIN CONTENT ---
st.title("🐾 Cat vs Dog Classifier")
st.write("Gunakan arsitektur **DenseNet** untuk prediksi yang lebih akurat.")

uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Terpilih', width=300)
    
    # Proses Prediksi
    model = load_trained_model()
    if model:
        with st.spinner('Menganalisis gambar...'):
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            if rescale:
                img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)
            prob = prediction[0][0]
            label = "Dog" if prob > 0.5 else "Cat"
            confidence = prob if label == "Dog" else 1 - prob

            # Hasil
            st.subheader(f"Hasil: **{label}**")
            st.progress(float(confidence))
            st.write(f"Tingkat Keyakinan: {confidence*100:.2f}%")
else:
    st.info("Silakan unggah gambar kucing atau anjing.")