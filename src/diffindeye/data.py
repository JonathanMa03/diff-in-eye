from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE
from .preprocess import preprocess_image
from .masks import horizontal_stripe_mask


def get_image_paths(split_dir):
    split_dir = Path(split_dir)
    return sorted(list(split_dir.glob("*/*.jpg")))


def get_default_paths():
    return {
        "train": get_image_paths(TRAIN_DIR),
        "val": get_image_paths(VAL_DIR),
        "test": get_image_paths(TEST_DIR),
    }


class OCTInpaintingDataset(Dataset):
    def __init__(
        self,
        image_paths,
        preprocess_fn=preprocess_image,
        mask_fn=horizontal_stripe_mask,
    ):
        self.image_paths = image_paths
        self.preprocess_fn = preprocess_fn
        self.mask_fn = mask_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = self.preprocess_fn(path)
        mask = self.mask_fn(image.shape)
        masked_image = image * mask

        image = torch.tensor(image).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        masked_image = torch.tensor(masked_image).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "path": str(path),
        }


def make_dataloader(
    image_paths,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    preprocess_fn=preprocess_image,
    mask_fn=horizontal_stripe_mask,
):
    dataset = OCTInpaintingDataset(
        image_paths=image_paths,
        preprocess_fn=preprocess_fn,
        mask_fn=mask_fn,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader