"""
This code is modified from https://github.com/TeaPearce/Conditional_Diffusion_MNIST/tree/main

"""

import numpy as np
import torch
import torch.nn as nn

from core.utils import timed_print
from models.base_models import ContextualUNet, DamageConditionedUNet


def compute_ddpm_schedule(beta_start: float, beta_end: float, num_steps: int) -> dict:
    """
    Compute the DDPM schedule including all coefficients required for training and sampling.

    Args:
        beta_start (float): Initial beta value for the noise schedule.
        beta_end (float): Final beta value for the noise schedule.
        num_steps (int): Total number of diffusion steps.

    Returns:
        dict: Dictionary with precomputed tensors for use in the DDPM process.
    """
    betas = torch.linspace(beta_start, beta_end, num_steps + 1)
    alphas = 1.0 - betas
    log_alphas = torch.log(alphas)
    alpha_bars = torch.exp(torch.cumsum(log_alphas, dim=0))

    return {
        "beta_t": betas,
        "alpha_t": alphas,
        "oneover_sqrta": 1.0 / torch.sqrt(alphas),
        "sqrt_beta_t": torch.sqrt(betas),
        "alphabar_t": alpha_bars,
        "sqrtab": torch.sqrt(alpha_bars),
        "sqrtmab": torch.sqrt(1.0 - alpha_bars),
        "mab_over_sqrtmab": (1.0 - alphas) / torch.sqrt(1.0 - alpha_bars),
    }


class DDPM(nn.Module):
    """
    Conditional Denoising Diffusion Probabilistic Model using contextual guidance.
    """

    def __init__(
        self,
        model: nn.Module,
        betas: tuple,
        num_steps: int,
        device: torch.device,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps
        self.drop_prob = drop_prob
        self.loss_fn = nn.MSELoss()

        schedule = compute_ddpm_schedule(betas[0], betas[1], num_steps)
        for key, value in schedule.items():
            self.register_buffer(key, value)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Perform one DDPM forward step.

        Args:
            x (Tensor): Ground truth image tensor.
            context (Tensor): Conditioning scalar.

        Returns:
            Tensor: MSE loss between predicted and true noise.
        """
        batch_size = x.shape[0]
        t_indices = torch.randint(
            1, self.num_steps + 1, (batch_size,), device=self.device
        )
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[t_indices, None, None, None] * x
            + self.sqrtmab[t_indices, None, None, None] * noise
        )
        t_normalized = t_indices / self.num_steps

        return self.loss_fn(noise, self.model(x_t, context, t_normalized))

    def sample(
        self,
        n_samples: int = 2,
        image_shape: tuple = (1, 512, 512),
        guide_weight: float = 0.0,
        context_extremes: tuple = (-1.0, 1.0),
    ):
        """
        Generate new samples using the learned DDPM model.

        Args:
            n_samples (int): Number of samples to generate.
            image_shape (tuple): Shape of each sample (C, H, W).
            guide_weight (float): Strength of classifier-free guidance.
            context_range (tuple): Range for condition values.

        Returns:
            Tuple[Tensor, np.ndarray]: Final sample tensor and stack of intermediate steps.
        """
        # n_samples should be at least 2 and even
        if n_samples < 2 or n_samples % 2 != 0:
            raise ValueError("n_samples must be at least 2 and even.")

        x = torch.randn(n_samples, *image_shape).to(self.device)

        # Create condition tensor
        conditions = torch.linspace(
            context_extremes[0], context_extremes[1], n_samples, device=self.device
        )

        if guide_weight > 0:
            conditions = torch.cat([conditions, conditions.flip(0)])
            x = x.repeat(2, 1, 1, 1)

        sample_evolution = []
        timed_print("Sampling with DDPM...")

        for t in reversed(range(1, self.num_steps + 1)):
            if t % 50 == 0:
                timed_print(f"Sampling step {t}...")

            t_tensor = torch.full((x.size(0),), t / self.num_steps, device=self.device)
            z = torch.randn(n_samples, *image_shape).to(self.device) if t > 1 else 0

            eps = self.model(x, conditions, t_tensor)

            if guide_weight > 0:
                eps_cond, eps_uncond = eps.chunk(2)
                eps = (1 + guide_weight) * eps_cond - guide_weight * eps_uncond
                x = x[:n_samples]

            x = (
                self.oneover_sqrta[t] * (x - eps * self.mab_over_sqrtmab[t])
                + self.sqrt_beta_t[t] * z
            )

            if t % 20 == 0 or t == self.num_steps or t < 8:
                sample_evolution.append(x.detach().cpu().numpy())

        return x, np.array(sample_evolution)


class DDPM_Damage(nn.Module):
    """
    DDPM variant that conditions on spatial damage masks.
    """

    def __init__(
        self,
        model: nn.Module,
        betas: tuple,
        num_steps: int,
        device: torch.device,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.num_steps = num_steps
        self.drop_prob = drop_prob
        self.loss_fn = nn.MSELoss()

        schedule = compute_ddpm_schedule(betas[0], betas[1], num_steps)
        for key, value in schedule.items():
            self.register_buffer(key, value)

    def forward(self, x: torch.Tensor, damage_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with damage conditioning.

        Args:
            x (Tensor): Input image.
            damage_mask (Tensor): Conditioning tensor.

        Returns:
            Tensor: MSE loss between predicted and true noise.
        """
        batch_size = x.shape[0]
        t_indices = torch.randint(
            1, self.num_steps + 1, (batch_size,), device=self.device
        )
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[t_indices, None, None, None] * x
            + self.sqrtmab[t_indices, None, None, None] * noise
        )
        t_normalized = t_indices / self.num_steps

        return self.loss_fn(noise, self.model(x_t, t_normalized, damage_mask))

    def sample(
        self,
        n_samples: int = 2,
        image_shape: tuple = (1, 512, 512),
        guide_weight: float = 0.0,
        context_extremes: tuple = (None, None),
    ):
        """
        Sample with spatial guidance based on damage or avoidance masks.

        Args:
            n_samples (int): Number of samples to generate.
            image_shape (tuple): Image shape.
            guide_weight (float): Strength of contrastive guidance.
            context_extremes (tuple): Max and min heatmaps.

        Returns:
            Tuple[Tensor, np.ndarray]: Final samples and evolution over time.
        """

        # n_samples should be at least 2 and even
        if n_samples < 2 or n_samples % 2 != 0:
            raise ValueError("n_samples must be at least 2 and even.")

        x = torch.randn(n_samples, *image_shape).to(self.device)
        sample_evolution = []

        context = torch.cat([context_extremes[0], context_extremes[1]], dim=0)
        if context.shape[0] < n_samples:
            context = context.repeat(n_samples // 2, 1)

        # Prepare masks for contrastive guidance
        if guide_weight > 0:
            damage_mask = torch.cat([context, context], dim=0)
            x = x.repeat(2, 1, 1, 1)

        timed_print("Sampling with damage-aware DDPM...")
        for t in reversed(range(1, self.num_steps + 1)):
            if t % 50 == 0:
                timed_print(f"Sampling step {t}...")

            t_tensor = torch.full((x.size(0),), t / self.num_steps, device=self.device)
            z = torch.randn(n_samples, *image_shape).to(self.device) if t > 1 else 0

            eps = self.model(x, t_tensor, damage_mask)

            if guide_weight > 0:
                eps_cond, eps_uncond = eps.chunk(2)
                eps = (1 + guide_weight) * eps_cond - guide_weight * eps_uncond
                x = x[:n_samples]

            x = (
                self.oneover_sqrta[t] * (x - eps * self.mab_over_sqrtmab[t])
                + self.sqrt_beta_t[t] * z
            )

            if t % 20 == 0 or t == self.num_steps or t < 8:
                sample_evolution.append(x.detach().cpu().numpy())

        return x, np.array(sample_evolution)
