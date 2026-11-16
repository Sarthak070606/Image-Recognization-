
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from model import MNISTCNN
from utils import preprocess_image, predict




st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

MODEL_PATH = Path("checkpoints/mnist_cnn.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    model = MNISTCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state)

    model.eval()

    return model




st.title("🔢 Handwritten Digit Recognizer")

st.write(
    "Draw a digit from **0 to 9** or upload an image. "
    "A PyTorch CNN trained on MNIST will predict it."
)

st.caption(
    f"Running inference on: {DEVICE.type.upper()}"
)




model = load_model()

if model is None:

    st.warning(
        "The trained model checkpoint is missing."
    )

    st.info(
        "Train the model first using:"
    )

    st.code(
        "python train.py --epochs 5",
        language="bash"
    )

    st.stop()




st.subheader("1. Provide a Digit")

mode = st.radio(
    "Choose input method",
    ["Draw a digit", "Upload an image"],
    horizontal=True
)

image = None




if mode == "Draw a digit":

    try:

        from streamlit_drawable_canvas import st_canvas

        st.write(
            "Draw one large digit using the white brush."
        )

        canvas_result = st_canvas(

            fill_color="rgba(0, 0, 0, 1)",

            stroke_width=18,

            stroke_color="#FFFFFF",

            background_color="#000000",

            height=280,

            width=280,

            drawing_mode="freedraw",

            key="digit_canvas",

            return_image_data=True
        )

        if canvas_result.image_data is not None:

            canvas_array = canvas_result.image_data

            if np.max(canvas_array[:, :, :3]) > 0:

                image = Image.fromarray(
                    canvas_array.astype(np.uint8),
                    "RGBA"
                ).convert("RGB")

                st.image(
                    image,
                    caption="Your Drawing",
                    width=220
                )

    except ImportError:

        st.error(
            "Install streamlit-drawable-canvas using:"
        )

        st.code(
            "pip install streamlit-drawable-canvas"
        )



else:

    uploaded = st.file_uploader(
        "Upload a digit image",
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded is not None:

        try:

            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                width=220
            )

        except Exception as e:

            st.error(
                f"Unable to read image: {e}"
            )



if image is not None:

    if st.button(
        "Predict Digit",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Recognizing digit..."):

                input_tensor = preprocess_image(image)

                digit, confidence, probabilities = predict(
                    model,
                    input_tensor,
                    DEVICE
                )

            st.divider()

            st.subheader("2. Prediction")

            left, right = st.columns(2)

            left.metric(
                "Predicted Digit",
                str(digit)
            )

            right.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

   

            st.write("### Class Probabilities")

            fig, ax = plt.subplots(figsize=(8, 3))

            ax.bar(
                range(10),
                probabilities
            )

            ax.set_xticks(range(10))

            ax.set_xlabel("Digit")

            ax.set_ylabel("Probability")

            ax.set_ylim(0, 1)

            ax.grid(
                axis="y",
                alpha=0.2
            )

            st.pyplot(fig)

            plt.close(fig)



            with st.expander(
                "View Preprocessed 28×28 Image"
            ):

                display = (
                    input_tensor[0, 0].detach().cpu().numpy()
                    * 0.3081 + 0.1307
                )

                display = display.clip(0, 1)

                st.image(
                    display,
                    clamp=True,
                    caption="MNIST-style input",
                    width=180
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

else:

    st.info(
        "Draw or upload a digit to enable prediction."
    )


with st.expander("About This Project"):

    st.markdown("""
    - **Dataset:** MNIST
    - **Training Images:** 60,000
    - **Testing Images:** 10,000
    - **Framework:** PyTorch
    - **Model:** Convolutional Neural Network (CNN)
    - **Input:** 28 × 28 grayscale image
    - **Output:** Digit from 0 to 9
    - **Preprocessing:** Grayscale conversion, inversion,
      cropping, resizing, centering and normalization.
    """)
