from PIL import Image, ImageOps
import numpy as np
import torch

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Accept RGB/color or grayscale images and convert them into MNIST-style
    tensors: one centered digit, white strokes on black background, 28x28.
    """
    # Handle transparency by compositing onto white before grayscale conversion.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")

    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image).astype(np.uint8)

    # Determine whether the digit is dark-on-light or light-on-dark.
    # MNIST convention: bright digit strokes on a dark background.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(border.mean()) > float(arr.mean()):
        arr = 255 - arr

    image = Image.fromarray(arr)

    # Remove empty margins by cropping to the visible digit strokes.
    threshold = max(20, int(np.percentile(arr, 75)))
    bbox = image.point(lambda p: 255 if p > threshold else 0).getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize while preserving aspect ratio; MNIST digits occupy about 20x20 pixels.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    tensor = torch.from_numpy(np.asarray(canvas).copy()).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    return tensor

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, digit = probabilities.max(dim=0)
    return int(digit.item()), float(confidence.item()), probabilities.cpu().numpy()
