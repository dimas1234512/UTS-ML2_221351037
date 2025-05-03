import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load scaler dan label encoder (pastikan kamu punya scaler.pkl dan label_encoder.pkl)
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="milkquality.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Kualitas Susu")
st.write("Masukkan parameter kualitas susu untuk memprediksi grade: Low, Medium, atau High.")
st.markdown("""
### Penjelasan Fitur:

- **pH**: Nilai pH susu, berkisar antara 3 hingga 9.5. Nilai ideal antara 6.25 hingga 6.90.
- **Temperatur**: Suhu susu dalam °C, berkisar dari 34°C hingga 90°C. Nilai ideal antara 34°C hingga 45.20°C.
- **Taste**: Kualitas rasa susu. Nilai 0 berarti "Buruk" dan 1 berarti "Baik".
- **Odor**: Kualitas bau susu. Nilai 0 berarti "Buruk" dan 1 berarti "Baik".
- **Fat**: Kandungan lemak. Nilai 0 berarti "Rendah" dan 1 berarti "Tinggi".
- **Turbidity**: Tingkat kekeruhan. Nilai 0 berarti "Rendah" dan 1 berarti "Tinggi".
- **Colour**: Nilai warna susu, berkisar dari 240 hingga 255. Nilai maksimum adalah 255.

- **Grade** (_target_) akan diprediksi berdasarkan input di atas, dengan kategori:
  - **Low**: Kualitas buruk
  - **Medium**: Kualitas sedang
  - **High**: Kualitas baik
""")


# Form input pengguna
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.8)
temperature = st.number_input("Temperatur (°C)", min_value=0.0, max_value=100.0, value=35.0)
taste = st.selectbox("Taste (Rasa)", [0, 1])
odor = st.selectbox("Odor (Bau)", [0, 1])
fat = st.selectbox("Fat (Lemak)", [0, 1])
turbidity = st.selectbox("Turbidity (Kekeruhan)", [0, 1])
color = st.number_input("Color (Skor Warna)", min_value=0.0, max_value=255.0, value=50.0)

# Tombol prediksi
if st.button("Prediksi Kualitas Susu"):
    input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, color]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    grade = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Prediksi Grade Susu: **{grade.upper()}**")
