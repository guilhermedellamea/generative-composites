import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from core.dataloaders import get_dataloaders
from core.paths import get_saved_figure_path, get_saved_models_path
from core.utils import print_args, set_device, timed_print
from models.cwgan import ConditionalGenerator, Critic
from models.tools import load_predictor


def gradient_penalty(discriminator, real_images, fake_images, device):
    alpha = torch.rand((real_images.size(0), 1, 1, 1), device=device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(
        True
    )
    mixed_scores = discriminator(interpolated)

    grad = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(grad.size(0), -1)
    grad_norm = grad.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp, grad_norm.mean()


class CWGANTrainer:
    def __init__(self, args):
        print_args(args, "CWGAN Training Arguments")

        self.args = args
        self.device = set_device(args.device)
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float("inf")

        self.time_training = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._init_data()
        self._init_models()
        self._init_optimizers()

    def _init_data(self):
        self.input_var, self.label_var = "reinforcement", "Ey"
        self.loaders, _ = get_dataloaders(
            dataset="dataset_elastic",
            input_var=self.input_var,
            label_var=self.label_var,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            normalize="mean_std",
            device=self.device,
        )
        self.train_loader = self.loaders[f"train"]
        self.val_loader = self.loaders[f"validation"]
        self.target_vf = 0.05

    def _init_models(self):
        self.generator = ConditionalGenerator(
            n_channels=1,
            resolution=self.args.resolution,
            initial_channels=self.args.initial_channels_G,
            latent_dim=self.args.latent_dim,
            condition_size=self.args.condition_size,
            nup=self.args.nup,
        ).to(self.device)

        self.critic = Critic(
            n_channels=1,
            resolution=self.args.resolution,
            initial_channels=self.args.initial_channels_D,
            kernel_size=self.args.kernel_size_D,
            ndown=self.args.ndown,
            norm=self.args.norm,
        ).to(self.device)

        self.predictor = load_predictor(
            label_var=self.label_var,
            device=self.device,
            initial_channels=16,
            ndown=7,
            mlp_hidden_layers=256,
            leakyrelu=True,
            norm_type="batch",
        )

    def _init_optimizers(self):
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.9)
        )
        self.optimizer_D = optim.Adam(
            self.critic.parameters(), lr=self.args.lr, betas=(0.5, 0.9)
        )

    def train_one_epoch(self):
        self.generator.train()
        self.critic.train()

        total_D, total_G, total_gp, total_grad_norm = 0, 0, 0, 0
        latent_dim = self.args.latent_dim

        z_train = torch.randn(len(self.train_loader.dataset), latent_dim)
        z_loader = torch.utils.data.DataLoader(z_train, batch_size=self.args.batch_size)

        for (real_imgs, labels), z in zip(self.train_loader, z_loader):
            real_imgs, labels = real_imgs.to(self.device), labels.to(self.device)
            z = z.to(self.device)
            labels = labels.unsqueeze(1)

            for _ in range(self.args.n_disc):
                fake_imgs = self.generator(torch.randn_like(z), labels)
                real_score = self.critic(real_imgs).mean()
                fake_score = self.critic(fake_imgs.detach()).mean()
                gp, grad_norm = gradient_penalty(
                    self.critic, real_imgs, fake_imgs.detach(), self.device
                )

                d_loss = fake_score - real_score + self.args.lambda_gp * gp

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

                total_D += d_loss.item()
                total_gp += gp.item()
                total_grad_norm += grad_norm.item()

            fake_imgs = self.generator(z, labels)
            gen_score = self.critic(fake_imgs)
            predicted = self.predictor(fake_imgs)
            t_loss = self.loss_fn(predicted, labels)
            g_loss = -gen_score.mean() + self.args.lambda_target * t_loss

            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            total_G += g_loss.item()

        n_batches = len(z_loader)
        return (
            total_D / (n_batches * self.args.n_disc),
            total_G / n_batches,
            total_gp / (n_batches * self.args.n_disc),
            total_grad_norm / (n_batches * self.args.n_disc),
            t_loss.item(),
        )

    def validation_one_epoch(self):
        self.generator.eval()
        self.critic.eval()

        total_D, total_G, total_gp, total_grad_norm, total_vf = 0, 0, 0, 0, 0
        z_val = torch.randn(len(self.val_loader.dataset), self.args.latent_dim)
        z_loader = torch.utils.data.DataLoader(
            z_val, batch_size=self.args.batch_size_validation
        )

        for (real_imgs, labels), z in zip(self.val_loader, z_loader):
            with torch.no_grad():
                labels = labels.unsqueeze(1)

                z = z.to(self.device)
                real_imgs, labels = real_imgs.to(self.device), labels.to(self.device)
                fake_imgs = self.generator(z, labels)
                real_score = self.critic(real_imgs).mean()
                fake_score = self.critic(fake_imgs).mean()

                predicted = self.predictor(fake_imgs)
                t_loss = self.loss_fn(predicted, labels)

                # Estimate volume fraction (assuming binary image)
                vf_batch = (fake_imgs > 0).float().mean().item()
                total_vf += vf_batch

            with torch.enable_grad():
                gp, grad_norm = gradient_penalty(
                    self.critic, real_imgs, fake_imgs, self.device
                )
            gp = gp.detach()
            grad_norm = grad_norm.detach()

            d_loss = fake_score - real_score + self.args.lambda_gp * gp.item()
            g_loss = -fake_score + self.args.lambda_target * t_loss

            total_D += d_loss.item()
            total_G += g_loss.item()
            total_gp += gp.item()
            total_grad_norm += grad_norm.item()

        n_batches = len(z_loader)
        avg_vf = total_vf / n_batches
        return (
            total_D / n_batches,
            total_G / n_batches,
            total_gp / n_batches,
            total_grad_norm / n_batches,
            t_loss.item(),
            avg_vf,  # Added volume fraction
        )

    def save_checkpoint(self, epoch):
        path = get_saved_models_path(f"cwgan_{self.label_var}")
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )
        timed_print(f"[Epoch {epoch}] Model checkpoint saved to {path}")

    def save_generated_images(self, epoch: int):
        """
        Generate and save a grid of fake samples from the generator.
        """
        self.generator.eval()
        with torch.no_grad():
            n_samples = 16
            z = torch.randn(n_samples, self.args.latent_dim).to(self.device)
            # Use a fixed target value (e.g., mean Ey) as conditioning
            cs = torch.linspace(-1, 1, steps=n_samples, device=self.device).unsqueeze(1)

            fake_imgs = self.generator(z, cs)

            # Normalize and convert to grid
            grid = make_grid(fake_imgs, nrow=4, normalize=True, pad_value=1)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(grid.cpu().permute(1, 2, 0), cmap="gray")
            ax.axis("off")
            save_path = get_saved_figure_path(f"wgan_generated_epoch_{epoch:03d}")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            timed_print(f"[Epoch {epoch}] Generated images saved to {save_path}")

    def run(self):
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validation_one_epoch()

            val_d_loss, val_g_loss, val_gp, val_grad_norm, t_loss, avg_vf = val_loss

            vf_error = abs(avg_vf - self.target_vf) / self.target_vf
            within_vf_margin = vf_error <= 0.10
            improved_loss = t_loss < self.best_val_loss

            timed_print(
                f"Epoch {epoch+1}/{self.args.epochs} | "
                f"Train D: {train_loss[0]:.4f} | Val D: {val_d_loss:.4f} | "
                f"Train G: {train_loss[1]:.4f} | Val G: {val_g_loss:.4f} | "
                f"Avg VF: {avg_vf:.4f} (Target: {self.target_vf:.4f}, Error: {vf_error:.1%}) | "
                f"Target Loss: {t_loss:.6f}"
            )

            if within_vf_margin and improved_loss:
                self.best_val_loss = t_loss
                self.save_checkpoint(epoch)

            if (
                self.args.save_image_frequency > 0
                and (epoch + 1) % self.args.save_image_frequency == 0
            ):
                self.save_generated_images(epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional WGAN-GP Trainer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-validation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-gp", type=float, default=300.0)
    parser.add_argument("--lambda-target", type=float, default=3.0)
    parser.add_argument("--n-disc", type=int, default=1)
    parser.add_argument("--kernel-size-D", type=int, default=3)
    parser.add_argument("--initial-channels-D", type=int, default=16)
    parser.add_argument("--initial-channels-G", type=int, default=4)
    parser.add_argument("--ndown", type=int, default=6)
    parser.add_argument("--nup", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--condition-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--norm", type=str, default="instance")
    parser.add_argument(
        "--save-image-frequency",
        type=int,
        default=1,
        help="Number of epochs between saving generated images.",
    )

    args = parser.parse_args()
    timed_print(f"Training started at {datetime.now()}")
    trainer = CWGANTrainer(args)

    trainer.run()
