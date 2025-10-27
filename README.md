# Fashion-MNIST Streamlit App

This repository contains a minimal Streamlit application that serves a **CNN image classifier** trained on the Fashion-MNIST dataset.

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
# Alternatively, SavedModel directory:
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

Then open the URL that Streamlit prints in the terminal.

## 4) Using a different path or SavedModel directory

If you saved a directory format, keep the directory name, for example `cnn_fashion_mnist/`. 
The loader in `app.py` will work for both `.h5` files and SavedModel directories.

To point the app to a different path without editing code, create a `.streamlit/secrets.toml` file:

```toml
MODEL_PATH = "path/to/your/model_or_directory"
```

## 5) Input preprocessing contract

The app converts any uploaded image to **28×28 grayscale**, inverts if the background looks white, then normalizes to `[0, 1]`. 
This mirrors Fashion-MNIST’s expected input `(28, 28, 1)`. If you trained with a different preprocessing, match it here.

## 6) Class names

The app uses the standard Fashion-MNIST labels:

```
0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat,
5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot
```

If your class ordering differs from the standard, update `CLASS_NAMES` in `app.py` accordingly.

## 7) Reproducible build

If you plan to deploy, pin versions in `requirements.txt` for stability or use a lockfile. 
GPU is not required for inference.

---

### Troubleshooting

- **Model failed to load**: Ensure the model file or directory exists relative to `app.py`. Set `MODEL_PATH` in Streamlit secrets if needed.
- **Wrong predictions**: Check that preprocessing matches your training pipeline. Try inverting images or using a plain background.
- **Package issues**: Recreate the virtual environment and reinstall from `requirements.txt`.

Enjoy!
