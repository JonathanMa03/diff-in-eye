import numpy as np
from PIL import Image

from .config import TARGET_SIZE


def preprocess_image(path, target_size=TARGET_SIZE):
    img = Image.open(path).convert("L")
    img = img.resize(target_size, resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0

    return arr