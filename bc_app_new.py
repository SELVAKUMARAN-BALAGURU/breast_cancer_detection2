import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.mixture import GaussianMixture

# Load the U-Net model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("attention_unet_best_model.keras")

model = load_model()

# Preprocess the uploaded image
def preprocess_image(image):
    """Load and preprocess image for model prediction."""
    img = image.convert("L")  # Convert to grayscale
    img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Resize
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    
    # Reshape to (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    return img_array

# Apply GMM for refining segmentation
def apply_gmm_to_segmentation(segmented_image):
    gmm = GaussianMixture(n_components=2)  # Two components for background and tumor
    reshaped = segmented_image.reshape(-1, 1)
    gmm.fit(reshaped)
    labels = gmm.predict(reshaped)
    refined_mask = labels.reshape(segmented_image.shape)
    
    # Clean noise using morphological opening
    refined_mask = cv2.morphologyEx(refined_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return refined_mask


st.title("Breast Cancer Tumor Segmentation using U-Net and GMM")
st.write("Upload an image for tumor detection.")

# Problem summary
st.subheader("Problem Summary")
st.write(
    "Breast cancer detection is crucial for early diagnosis and treatment. "
    "Using deep learning models like U-Net, we can analyze mammogram images "
    "and segment tumor regions with high accuracy."
)

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    st.write("Processing image...")
    processed_image = preprocess_image(image)

    # Predict using U-Net
    prediction = model.predict(processed_image)[0]

    # Convert to binary mask
    mask = (prediction > 0.5).astype(np.uint8) * 255

    # Apply GMM refinement
    refined_mask = apply_gmm_to_segmentation(prediction)

    # Display original and refined mask
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Initial U-Net Mask")
    ax[1].axis("off")

    ax[2].imshow(refined_mask, cmap="gray")
    ax[2].set_title("Refined Mask (GMM)")
    ax[2].axis("off")

    st.pyplot(fig)

    # Generate graph analysis of prediction parameters
    st.subheader("Graph Analysis of Prediction Parameters")
    tumor_percentage = np.mean(refined_mask) * 100
    trust_score = np.clip(10 - (tumor_percentage / 10), 1, 10)

    fig, ax = plt.subplots()
    ax.bar(["Tumor Percentage", "Trust Score"], [tumor_percentage, trust_score], color=["red", "blue"])
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # Display trustworthiness score
    st.subheader("Trustworthiness Score")
    st.write(f"On a scale of 1 to 10, this prediction has a trustworthiness score of: *{trust_score:.1f}*")

print("Streamlit app saved as bc_app_new.py")
