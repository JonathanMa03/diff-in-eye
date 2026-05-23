import numpy as np
from .config import STRIPE_HEIGHT


def center_mask(shape, mask_size=(64, 64)):
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)

    mh, mw = mask_size
    top = (h - mh) // 2
    left = (w - mw) // 2

    mask[top:top + mh, left:left + mw] = 0.0

    return mask


def horizontal_stripe_mask(shape, stripe_height=STRIPE_HEIGHT):
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)

    center = h // 2
    top = center - stripe_height // 2

    mask[top:top + stripe_height, :] = 0.0

    return mask


def random_square_mask(shape, square_size=48):
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)

    top = np.random.randint(0, h - square_size)
    left = np.random.randint(0, w - square_size)

    mask[top:top + square_size, left:left + square_size] = 0.0

    return mask