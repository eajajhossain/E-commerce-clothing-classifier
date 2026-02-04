import numpy as np
from PIL import Image

IMG_SIZE = (256, 256)  # MUST match training

def preprocess_image(image: Image.Image):
    """
    Takes a PIL image and returns a model-ready tensor
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0  # MUST match training

    image_array = np.expand_dims(image_array, axis=0)
    return image_array
