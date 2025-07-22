import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 12px;
    }
    .title {
        font-size: 36px;
        text-align: center;
        font-weight: bold;
        color: #333333;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #777777;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/imageclassifier.keras')

model = load_model()

# Title
st.markdown('<div class="main"><div class="title">😊 Emotion Detection App (Happy or Sad)</div>', unsafe_allow_html=True)
st.markdown("Upload a face image and I will tell you if it's a happy or sad face! 📷")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    with st.spinner('Predicting emotion...'):
        # Preprocess
        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction) * 100, 2)
        label = "Sad 😢" if prediction > 0.5 else "Happy 😊"

    # Result
    st.success(f"**Predicted Emotion:** {label}")
    st.info(f"Confidence: **{confidence}%**")

    # Optionally add emojis or color based on result
    if label.startswith("Happy"):
        st.balloons()
    else:
        st.warning("Cheer up! 😢 You look a bit sad.")

# Footer
st.markdown('</div><div class="footer">Made with ❤️ by Abhi</div>', unsafe_allow_html=True)
