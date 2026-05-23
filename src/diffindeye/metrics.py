import numpy as np


def masked_mse(original, reconstruction, mask):
    missing_region = mask == 0
    return np.mean((original[missing_region] - reconstruction[missing_region]) ** 2)


def masked_mae(original, reconstruction, mask):
    missing_region = mask == 0
    return np.mean(np.abs(original[missing_region] - reconstruction[missing_region]))


def psnr(original, reconstruction):
    mse = np.mean((original - reconstruction) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(1.0 / np.sqrt(mse))