import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label
from scipy.spatial.distance import cdist

from .paths import *


def set_device(device_id: int = None):
    """
    Set the default device for torch operations.

    Parameters:
    - device_id (int, optional): Specify a CUDA device ID (e.g., 0 or 1). If None, auto-selects the best device.

    Returns:
    - torch.device: The device set as default.
    """
    if torch.cuda.is_available():
        if device_id is not None and device_id.isnumeric():
            if device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid device_id {device_id}. Only {torch.cuda.device_count()} CUDA device(s) available."
                )
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device(device_id)
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)
    print(f"===> Using device: {device}", flush=True)
    return device


def timed_print(text, func=""):
    hour = str(datetime.now())
    if func != "":
        func = "[ {} ]".format(func)
    text_to_print = "{} {} {}".format(hour, func, text)
    print(text_to_print, flush=True)


def print_args(args, func=""):
    print("=" * 40)
    print(f"{func} Configuration:")
    for key, value in vars(args).items():
        print(f"{key:>25}: {value}")
    print("=" * 40)


class SimulationHandler:
    """
    Handles listing and filtering of simulation files under:
        BASE_DATA_DIR / dataset_name / "reinforcement" /
    Each file is assumed to represent one simulation and the filename (without extension) is used as ID.
    """

    def __init__(self, base_dir: Path, dataset_name: str):
        """
        Args:
            base_dir: Root folder containing all datasets.
            dataset_name: Subfolder name for the dataset (e.g., 'dataset_elastic').
        """
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.reinforcement_dir = self.base_dir / self.dataset_name / "reinforcement"
        self.reinforcement_dir.mkdir(parents=True, exist_ok=True)

    def read_simulations(self) -> List[str]:
        """
        Return sorted list of simulation IDs (filenames without extension).
        """
        return sorted({p.stem for p in self.reinforcement_dir.glob("*") if p.is_file()})


def load_data(dataset: str, simulation: str, variable: str) -> np.ndarray:
    """
    Load a .npy data file for the given dataset, simulation ID, and variable.

    Args:
        dataset: Name of the dataset (e.g., 'dataset_elastic').
        simulation: Simulation ID (e.g., '1').
        variable: Variable name (e.g., 'Ey', 'reinforcement').

    Returns:
        A numpy array loaded from the .npy file.
    """
    file_path = BASE_DATA_DIR / dataset / variable / f"{simulation}.npy"
    return np.load(file_path)
