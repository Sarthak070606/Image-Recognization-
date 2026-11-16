# Handwritten Digit Recognition using PyTorch + MNIST

A beginner-friendly deep-learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with **PyTorch** and a **Streamlit** frontend.

## Features
- Train a CNN on the MNIST dataset.
- Draw a digit on an interactive canvas.
- Upload a PNG/JPG/BMP image.
- Display the predicted digit, confidence, and probabilities for all ten classes.
- Automatically uses CUDA when available; otherwise uses CPU.
- Converts user images into MNIST-like 28×28 grayscale inputs.

## Project structure

```text
mnist_digit_recognizer/
├── app.py                  # Streamlit frontend
├── model.py                # CNN architecture
├── train.py                # MNIST training and evaluation
├── utils.py                # Image preprocessing and prediction
├── requirements.txt
├── README.md
├── .gitignore
└── checkpoints/            # Trained model is saved here
```

## 1. Create and activate a virtual environment

### Windows PowerShell
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, use the command recommended by the official PyTorch installation selector for your CUDA version. The CPU installation also works.

## 3. Train the model

```bash
python train.py --epochs 5
```

The first run downloads MNIST into the `data/` folder. Training prints test accuracy after each epoch and saves the checkpoint to:

```text
checkpoints/mnist_cnn.pth
```

You can change the number of epochs or batch size:

```bash
python train.py --epochs 8 --batch-size 128 --lr 0.001
```

## 4. Launch Streamlit

```bash
streamlit run app.py
```

Open the local URL printed in your terminal (usually `http://localhost:8501`).

## How it works

1. MNIST images are converted to tensors and normalized using mean `0.1307` and standard deviation `0.3081`.
2. The CNN applies two convolution + ReLU + max-pooling blocks.
3. Fully connected layers produce ten logits, one for each digit.
4. Softmax converts logits into class probabilities for display.
5. The frontend preprocesses drawn/uploaded images to resemble MNIST's 28×28 centered digit format.

## Model architecture

| Layer | Output shape (batch dimension omitted) |
|---|---:|
| Input | 1 × 28 × 28 |
| Conv2D(1, 32, 3) + ReLU | 32 × 28 × 28 |
| MaxPool2D(2) | 32 × 14 × 14 |
| Conv2D(32, 64, 3) + ReLU | 64 × 14 × 14 |
| MaxPool2D(2) | 64 × 7 × 7 |
| Flatten | 3136 |
| Linear(3136, 128) + ReLU + Dropout | 128 |
| Linear(128, 10) | 10 logits |

## Notes / troubleshooting

- **Model checkpoint missing:** run `python train.py --epochs 5` before launching the app.
- **Canvas import error:** ensure `streamlit-drawable-canvas` installed correctly. The upload option can be used as an alternative.
- **Poor prediction on photos:** MNIST contains centered white handwritten digits on a dark background. Use one clear, large handwritten digit with minimal background. Color photos are converted to grayscale automatically, but complex backgrounds may reduce accuracy.
- **CUDA out of memory:** train on CPU or reduce `--batch-size` to `64`.

## Suggested extensions
- Add confusion matrix and per-class precision/recall.
- Save user predictions to a CSV.
- Compare CNN with a fully connected neural network.
- Add model training curves (loss and accuracy).
