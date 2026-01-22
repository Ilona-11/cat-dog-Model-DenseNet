import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- KONFIGURASI HALAMAN & TEMA ---
st.set_page_config(page_title="DenseNet Animal Classifier", layout="centered")

# CSS Custom untuk Tema Hijau Army
st.markdown("""
    <style>
    .stApp {
        background-color: #4B5320; /* Army Green */
        color: #F5F5DC; /* Beige/Cream Text */
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .stMarkdown, h1, h2, h3, p, span {
        color: #F5F5DC !important;
    }
    .stFileUploader {
        background-color: #353D16;
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #DDE2C6;
    }
    .stButton>button {
        background-color: #606E32 !important;
        color: white !important;
        border-radius: 5px;
        border: 1px solid #F5F5DC;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model_densenet():
    # Mengarah ke hasil download dari Colab di folder models
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'cat_dog_classifier_densenet.h5')
    
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        st.error(f"File tidak ditemukan: {model_path}")
        return None

# --- UI UTAMA ---
st.title("🐾 Klasifikasi Kucing & Anjing")
st.write("Arsitektur: **Convolutional Neural Network (DenseNet121)**")
st.divider()

# Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar kucing atau anjing...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    # Tombol Prediksi
    if st.button("Mulai Klasifikasi"):
        model = load_model_densenet()
        
        if model:
            with st.spinner('Sedang menganalisis ciri-ciri hewan...'):
                # Preprocessing sesuai input DenseNet (224x224)
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = img_array / 255.0  # Normalisasi
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediksi
                prediction = model.predict(img_array)
                score = prediction[0][0]
                
                # Menentukan Label
                label = "Anjing (Dog)" if score > 0.5 else "Kucing (Cat)"
                confidence = score if score > 0.5 else 1 - score
                
                # Hasil Tampilan
                st.subheader(f"Hasil Prediksi: **{label}**")
                st.write(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")
                st.progress(float(confidence))
        else:
            st.warning("Pastikan file model sudah ada di folder 'models/'")

else:
    st.info("Silakan unggah file gambar untuk memulai analisis.")

# Footer Sidebar
st.sidebar.markdown("### Tentang Proyek")
st.sidebar.info("Proyek ini menggunakan Transfer Learning dengan model DenseNet121 untuk akurasi tinggi pada klasifikasi biner.")