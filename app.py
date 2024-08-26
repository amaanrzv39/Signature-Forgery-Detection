import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances as L2

# Load your pre-trained Siamese model
model = tf.keras.models.load_model("/content/embeddings_triplet_loss.h5")
model.load_weights("/content/embeddings_triplet_loss_weights.h5")

def preprocess_image(image):
    image = image.resize((180, 180))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def compare_signatures(image1, image2):
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)
    # Use the model to predict similarity
    dist = L2(model.predict(np.expand_dims(image1,axis=0)),model.predict(np.expand_dims(image2,axis=0)))
    return dist

# Streamlit app interface
st.title("Signature Comparison using Siamese Network")

st.write("Upload two signature images to compare.")

uploaded_file1 = st.file_uploader("Choose the first image...", type="png")
uploaded_file2 = st.file_uploader("Choose the second image...", type="png")

if uploaded_file1 is not None and uploaded_file2 is not None:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    st.image(image1, caption='First Image', use_column_width=True)
    st.image(image2, caption='Second Image', use_column_width=True)

    st.write("Comparing signatures...")

    dist = compare_signatures(image1, image2)

    st.write(f"Similarity Score: {similarity_score:.2f}")

    if dist < 0.65:
        st.write("The signatures are similar.")
    else:
        st.write("The signatures are different.")
