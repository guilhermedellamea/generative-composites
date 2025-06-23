import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from core.paths import BASE_DATA_DIR
from core.utils import SimulationHandler, load_data


def seed_everything(seed: int = 1903) -> None:
    """
    Seed random number generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading microstructure images and labels.
    Optionally returns the simulation ID with each sample.
    """

    def __init__(
        self,
        samples: List[str],
        dataset_name: str,
        input_var: str,
        label_var: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_id: bool = False,
    ) -> None:
        self.samples = samples
        self.dataset_name = dataset_name
        self.input_var = input_var
        self.label_var = label_var
        self.transform = transform
        self.target_transform = target_transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, str],
        Tuple[torch.Tensor, str],
    ]:
        sim_id = self.samples[idx]
        input_np = load_data(self.dataset_name, sim_id, self.input_var)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        if self.label_var:
            label_np = load_data(self.dataset_name, sim_id, self.label_var)
            label_tensor = torch.tensor(label_np).float()
            if label_tensor.ndim == 2:
                label_tensor = label_tensor.unsqueeze(0)
            if self.target_transform:
                label_tensor = self.target_transform(label_tensor)

            if self.return_id:
                return input_tensor, label_tensor, sim_id
            return input_tensor, label_tensor

        return (input_tensor, sim_id) if self.return_id else input_tensor


# Normalization factories


def get_normalizer(
    method: str,
    mean: float,
    std: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a normalization function based on the specified method.

    Supported methods:
    - 'mean_std': standard score normalization (z-score)
    - 'minmax': maps values to [0, 1]
    - 'minmax_signed': maps values to [-1, 1]
    """

    if method == "mean_std":

        def _norm(x: torch.Tensor) -> torch.Tensor:
            return (x - mean) / std

        return _norm

    elif method == "negative_mean_std":

        def _norm(x: torch.Tensor) -> torch.Tensor:
            return -(x - mean) / std

        return _norm

    elif method == "minmax":

        def _norm(x: torch.Tensor) -> torch.Tensor:
            return (x - min_val) / (max_val - min_val)

        return _norm

    elif method == "minmax_signed":

        def _norm(x: torch.Tensor) -> torch.Tensor:
            return 2 * (x - min_val) / (max_val - min_val) - 1

        return _norm

    else:
        raise ValueError(f"Unsupported normalization method: '{method}'")


# Data splitting


def split_simulations(
    dataset: str,
    val_frac: float = 0.1,
    seed: int = 1903,
) -> Dict[str, List[str]]:
    """
    Split a dataset's simulation IDs into train/validation/test sets.

    Args:
        dataset: Name of the dataset to split.
        val_frac: Fraction of the data to reserve for validation (and equally for test).
        seed: Random seed for reproducibility.

    Returns:
        A dict with keys 'train', 'validation', 'test' mapping to lists of simulation IDs.
    """
    seed_everything(seed)
    handler = SimulationHandler(BASE_DATA_DIR, dataset)
    sims = handler.read_simulations()

    print(handler.reinforcement_dir)

    random.shuffle(sims)
    n = len(sims)
    k = int(n * val_frac)

    return {
        "train": sims[: n - 2 * k],
        "validation": sims[n - 2 * k : n - k],
        "test": sims[n - k :],
    }


# Loader factory


def get_dataloaders(
    dataset: str,
    input_var: str,
    label_var: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    normalize: str = "mean_std",
    seed: int = 1903,
    return_id: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, torch.Tensor]]:
    """
    Return train/val/test dataloaders and label statistics for one dataset,
    a single input variable and a single label variable.
    """
    seed_everything(seed)

    # split the single dataset
    splits = split_simulations(dataset)

    # compute statistics for the single label
    values = [
        load_data(dataset, simulation_id, label_var)
        for simulation_id in splits["train"]
    ]
    arr = np.array(values)
    label_stats = {
        "mean": torch.tensor(arr.mean()),
        "std": torch.tensor(arr.std()),
        "min": torch.tensor(arr.min()),
        "max": torch.tensor(arr.max()),
        "splits": splits,
    }

    # prepare transforms
    in_transforms: List[Callable] = []
    if normalize == "mean_std":
        in_transforms.append(transforms.Normalize([0.5], [0.5]))
    input_transform = transforms.Compose(in_transforms) if in_transforms else None

    target_transform = get_normalizer(
        normalize,
        label_stats["mean"].item(),
        label_stats["std"].item(),
        label_stats["min"].item(),
        label_stats["max"].item(),
    )

    # build loaders for each split
    loaders: Dict[str, DataLoader] = {}
    for split_name, samples in splits.items():
        ds = CustomDataset(
            samples,
            dataset,
            input_var,
            label_var,
            transform=input_transform,
            target_transform=target_transform,
            return_id=return_id,
        )

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            generator=torch.Generator(device).manual_seed(seed),
        )

    return loaders, label_stats
