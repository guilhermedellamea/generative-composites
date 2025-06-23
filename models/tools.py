import torch

from core.paths import get_saved_models_path
from models.base_models import ResNet


def load_predictor(
    label_var: str,
    device: torch.device,
    initial_channels: int = 16,
    ndown: int = 7,
    mlp_hidden_layers: int = 256,
    leakyrelu: bool = False,
    norm_type: str = "batch",
) -> torch.nn.Module:
    """
    Load a trained predictor model from disk.

    Args:
        label_var (str): Target variable the model was trained on (e.g., "Ey", "energy_when_broken_Y").
        device (str): Device to load the model to, e.g., "cuda" or "cpu".
        initial_channels (int): Number of channels in the first convolutional layer.
        ndown (int): Number of downsampling blocks in the ResNet.
        mlp_hidden_layers (int): Number of hidden units in the MLP head.
        leakyrelu (bool): Whether LeakyReLU is used instead of ReLU.
        norm_type (str): Type of normalization ("batch", "instance", etc.).

    Returns:
        torch.nn.Module: The loaded ResNet model in evaluation mode.
    """
    # Reconstruct the model architecture
    model = ResNet(
        n_channels=1,
        n_labels=1,
        resolution=512,
        initial_channels=initial_channels,
        ndown=ndown,
        mlp_hidden_layers=mlp_hidden_layers,
        leakyrelu=leakyrelu,
        norm_type=norm_type,
    ).to(device)

    # Load the model weights
    path = get_saved_models_path(f"predictor_{label_var}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return model
