import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
#import logging
from utils.preprocess import preprocess_image

#logging.basicConfig(level=logging.DEBUG)
#st.write("✅ App has started...")
# Set page config
st.set_page_config(page_title="SmartLeaf: Crop Disease Detection Model", layout="centered")

# Title and description
st.title("🌿 SmartLeaf: Crop Disease Detection App")
st.write(
    """
    Detect crop diseases in seconds!  
    📁 Upload an image **or** 📸 take a photo of a crop leaf, and SmartLeaf will predict the crop disease.
    """
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Smart_Leaf_resnet_model.h5")

model = load_model()

# Load class labels
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# Input option: Upload or Camera
input_option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

# Get image
image_file = None
if input_option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

# Show and predict
if image_file is not None:
    st.image(image_file, caption="Selected Image", use_container_width=True)
    if st.button("Predict"):
        try:
            # Preprocess
            img_array = preprocess_image(image_file)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Show result
            st.success(f"🩺 Predicted Disease: **{predicted_class}**")
            st.info(f"📊 Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("📷 Please upload or take a photo to begin.")
