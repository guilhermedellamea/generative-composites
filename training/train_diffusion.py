import argparse
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision.utils import make_grid

from core.dataloaders import get_dataloaders
from core.paths import get_saved_figure_path, get_saved_models_path
from core.utils import load_data, print_args, set_device, timed_print
from models.base_models import ContextualUNet, DamageConditionedUNet
from models.ddpm import DDPM, DDPM_Damage


class DDPMTrainer:
    def __init__(self, args):
        print_args(args, "DDPM Training Arguments")
        self.args = args
        self.device = set_device(args.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._init_data()
        self._init_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _init_data(self):
        if self.args.mode == "elastic":
            self.dataset = "dataset_elastic"
            self.input_var = "reinforcement"
            self.label_var = "Ey"
            self.mode = "elastic"
        elif self.args.mode == "damage":
            self.dataset = "dataset_damage"
            self.input_var = "reinforcement"
            self.label_var = "damage"
            self.mode = "damage"
        else:
            raise ValueError("Invalid mode")

        self.loaders, self.dataset_stats = get_dataloaders(
            dataset=self.dataset,
            input_var=self.input_var,
            label_var=self.label_var,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            normalize="mean_std",
            device=self.device,
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders["validation"]

    def _init_model(self):
        if self.mode == "elastic":
            net = ContextualUNet(
                1,
                n_feat=self.args.n_feat,
                n_down=self.args.ndown,
                norm_type=self.args.norm,
            )
            model = DDPM(
                model=net,
                betas=(self.args.beta1, self.args.beta2),
                num_steps=self.args.n_T,
                device=self.device,
            )
        elif self.mode == "damage":
            net = DamageConditionedUNet(
                1, base_channels=self.args.n_feat, norm_type=self.args.norm
            )
            model = DDPM_Damage(
                model=net,
                betas=(self.args.beta1, self.args.beta2),
                num_steps=self.args.n_T,
                device=self.device,
            )

        self.n_gpu = torch.cuda.device_count()
        self.model = model
        self.module_to_sample_from = model
        if self.n_gpu > 1:
            print(f"Using {self.n_gpu} GPUs")
            self.model = torch.nn.DataParallel(model)
            self.module_to_sample_from = self.model.module

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        moving_loss = None

        for x, context in self.train_loader:
            x, context = x.to(self.device), context.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(x, context)
            loss = loss.mean()

            loss.backward()

            if moving_loss is None:
                moving_loss = loss.item()
            else:
                moving_loss = 0.95 * moving_loss + 0.05 * loss.item()

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def val_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, context in self.val_loader:
                x, context = x.to(self.device), context.to(self.device)
                loss = self.model(x, context)
                loss = loss.mean()
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch):
        name = f"ddpm_{self.mode}_{self.label_var}"
        path = get_saved_models_path(name)
        torch.save(self.module_to_sample_from.state_dict(), path)
        timed_print(f"[Epoch {epoch}] Checkpoint saved to {path}")

    def save_samples(self, epoch):
        self.model.eval()

        if self.args.mode == "elastic":
            context_extremes = (-1, 1)
        elif self.args.mode == "damage":
            sim_id_validation = random.choice(
                self.dataset_stats["splits"]["validation"]
            )
            gradcam_heatmap = load_data(
                self.dataset,
                sim_id_validation,
                "gradcam",
            )
            context_extremes = (
                gradcam_heatmap,
                gradcam_heatmap * self.args.weight_damage,
            )

        for guide_weight in self.args.guide_weights:
            timed_print(
                f"[Epoch {epoch}] Generating samples with guide weight {guide_weight}"
            )

            with torch.no_grad():
                samples, _ = self.module_to_sample_from.sample(
                    n_samples=self.args.n_sample,
                    image_shape=(1, 512, 512),
                    guide_weight=guide_weight,
                    context_extremes=context_extremes,
                )
                grid = make_grid(samples, nrow=4, normalize=True)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(grid.cpu().permute(1, 2, 0), cmap="gray")
                ax.axis("off")
                save_path = get_saved_figure_path(
                    f"ddpm_generated_guide_w_{guide_weight}_epoch_{epoch:03d}"
                )
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.close()
                timed_print(f"[Epoch {epoch}] Sample images saved to {save_path}")

    def run(self):
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch()

            val_loss = self.val_one_epoch()
            timed_print(
                f"Epoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if (epoch + 1) % self.args.sample_each_epoch == 0:
                self.save_samples(epoch + 1)
                self.save_checkpoint(epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM Trainer")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-size-validation", type=int, default=16)
    parser.add_argument("--n_sample", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n-feat", type=int, default=32)
    parser.add_argument("--beta1", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.02)
    parser.add_argument("--n-T", type=int, default=500)
    parser.add_argument("--norm", type=str, default="instance")
    parser.add_argument("--ndown", type=int, default=2)
    parser.add_argument(
        "--mode", type=str, choices=["elastic", "damage"], default="elastic"
    )
    parser.add_argument("--sample_each_epoch", type=int, default=10)
    parser.add_argument("--guide_weights", type=list, default=[0.0, 0.5, 4.0])
    parser.add_argument("--weight_damage", type=float, default=0.1)

    args = parser.parse_args([])
    timed_print(f"Training DDPM started at {datetime.now()}")

    self = DDPMTrainer(args)

    self.run()
