import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👕")
st.title("Fashion-MNIST Image Classifier")

@st.cache_resource
def load_model():
    # Prefer SavedModel directory, fall back to H5 with compile=False for cross-version safety
    for path in ("cnn_fashion_mnist", "cnn_fashion_mnist.h5"):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception:
            continue
    raise FileNotFoundError("Model not found. Put 'cnn_fashion_mnist/' or 'cnn_fashion_mnist.h5' next to app.py")

model = load_model()
CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

def preprocess(img):
    img = img.convert("L")
    img = ImageOps.fit(img, (28, 28), Image.Resampling.LANCZOS)
    x = np.asarray(img).astype("float32")
    if x.mean() > 127:   # invert if background looks white
        x = 255.0 - x
    x = x / 255.0
    return np.expand_dims(x, axis=(0, -1))  # (1,28,28,1)

file = st.file_uploader("Upload a image", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file)
    st.image(img, width=256, caption="Uploaded")
    x = preprocess(img)
    p = model.predict(x, verbose=0)[0]
    if p.max() > 1:   # logits -> probabilities
        e = np.exp(p - p.max()); p = e / e.sum()
    st.subheader(f"Prediction: **{CLASS_NAMES[int(np.argmax(p))]}**")
    st.bar_chart(p)
else:
    st.info("Upload an image to get a prediction.")
