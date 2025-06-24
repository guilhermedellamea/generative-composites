import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam

from core.dataloaders import get_dataloaders
from core.paths import get_saved_models_path
from core.utils import print_args, set_device, timed_print
from models.base_models import ResNet


class PredictorTrainer:
    """
    Trainer class for supervised learning of scalar properties from microstructure images.

    Args:
        args (argparse.Namespace): Training arguments parsed from command line.
    """

    def __init__(self, args: argparse.Namespace):
        print_args(args, func="PredictorTrainer")  # Print arguments in a structured way
        self.args = args
        self.device = set_device(args.device)  # torch.device("cuda" or "cpu")
        self.timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Timestamp for file saving

        self._init_data()

    def _init_data(self):
        """
        Initialize data loaders for training and validation sets.
        The target variable depends on the dataset type.
        """
        input_var = "reinforcement"  # Always the same input variable in this setting

        if self.args.dataset == "dataset_elastic":
            self.label_var = "Ey"  # Target is Young's modulus
            normalize = "mean_std"
        elif self.args.dataset == "dataset_damage":
            self.label_var = "toughness"  # Target is toughness
            normalize = "negative_mean_std"
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        # Load dataloaders with corresponding input/output and normalization
        self.loaders, _ = get_dataloaders(
            dataset=self.args.dataset,
            input_var=input_var,
            label_var=self.label_var,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            normalize=normalize,
            device=self.device,
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders["validation"]

    def train(self):
        """
        Train the model using MSE loss and Adam optimizer.
        Best model is saved automatically based on validation loss.
        """
        model = ResNet(
            n_channels=1,
            n_labels=1,
            resolution=512,
            initial_channels=self.args.initial_channels,
            ndown=self.args.ndown,
            mlp_hidden_layers=self.args.mlp_hidden_layers,
            leakyrelu=self.args.leakyrelu,
            norm_type=self.args.norm_type,
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        timed_print(f"Model has {n_params} parameters")

        optimizer = Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            model.train()
            train_loss = self._train_one_epoch(model, optimizer, criterion)

            model.eval()
            val_loss = self._val_one_epoch(model, criterion)

            timed_print(
                f"[{epoch+1}/{self.args.epochs}] Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = get_saved_models_path(f"predictor_{self.label_var}")
                torch.save(model.state_dict(), path)

    def _train_one_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Train the model for one epoch.

        Returns:
            float: Average training loss over the epoch.
        """
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs[:, 0], labels.to(self.device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _val_one_epoch(self, model: nn.Module, criterion: nn.Module) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = model(inputs.to(self.device))
                loss = criterion(outputs[:, 0], labels.to(self.device))
                total_loss += loss.item()

        return total_loss / len(self.val_loader)


if __name__ == "__main__":
    # Define command-line arguments and their types
    parser = argparse.ArgumentParser(description="Train predictor on material data")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dataset_elastic", "dataset_damage"],
        default="dataset_elastic",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-validation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--initial-channels", type=int, default=16)
    parser.add_argument("--ndown", type=int, default=7)
    parser.add_argument("--leakyrelu", action="store_true")
    parser.add_argument("--mlp-hidden-layers", type=int, default=256)
    parser.add_argument("--norm-type", type=str, default="batch")

    args = parser.parse_args()

    train = PredictorTrainer(args)

    train.train()
