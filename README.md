# Fashion-MNIST Streamlit App

This repository contains a minimal Streamlit application that serves a **CNN image classifier** trained on the Fashion-MNIST dataset.

Open the URL that Streamlit : https://mkpr26-fashionimageclassificationusingcnns.streamlit.app/

## Project layout

```
.
├── app.py                     # Streamlit UI for inference
├── cnn_fashion_mnist.h5       # Your saved Keras model (provide this file)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

> Your training is in `Experiments.ipynb`, which fits a `cnn_model` on Fashion-MNIST CSVs. 
> Export that trained model as described below and place it alongside `app.py`.

## 1) Environment setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Save the trained model from your notebook

In your `Experiments.ipynb`, after training the CNN, run:

```python
# After cnn_model.fit(...)
cnn_model.save("cnn_fashion_mnist.h5")  # single-file Keras model
# cnn_model.save("cnn_fashion_mnist", save_format="tf")
```

This creates either:
- `cnn_fashion_mnist.h5` (single file), or
- `cnn_fashion_mnist/` directory (TensorFlow SavedModel).

Copy the produced model artifact into the same directory as `app.py`.

## 3) Launch the app

```bash
streamlit run app.py
```




