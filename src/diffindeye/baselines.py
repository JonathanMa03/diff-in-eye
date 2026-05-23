import numpy as np
import cv2


def zero_fill_reconstruction(masked, mask=None):
    return masked.copy()


def gaussian_reconstruction(masked, mask, kernel_size=(21, 21), sigma_x=0):
    blurred = cv2.GaussianBlur(
        masked,
        ksize=kernel_size,
        sigmaX=sigma_x,
    )

    recon = masked.copy()
    recon[mask == 0] = blurred[mask == 0]

    return recon


def telea_reconstruction(masked, mask, inpaint_radius=3):
    masked_uint8 = (masked * 255).astype(np.uint8)
    opencv_mask = ((1 - mask) * 255).astype(np.uint8)

    recon = cv2.inpaint(
        masked_uint8,
        opencv_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    return recon.astype(np.float32) / 255.0


def pde_diffusion_reconstruction(
    masked_image,
    mask,
    n_iters=500,
    dt=0.1,
):
    u = masked_image.copy()

    known_pixels = mask == 1
    missing_pixels = mask == 0

    for _ in range(n_iters):
        laplacian = (
            np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
            - 4 * u
        )

        u[missing_pixels] = u[missing_pixels] + dt * laplacian[missing_pixels]
        u[known_pixels] = masked_image[known_pixels]

    return u