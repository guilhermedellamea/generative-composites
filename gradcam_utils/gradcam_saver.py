import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from core.dataloaders import get_dataloaders
from core.paths import get_gradcam_folder, get_saved_figure_path, get_saved_models_path
from core.utils import print_args, set_device
from gradcam_utils.gradcam import GradCAMplusplus
from models.base_models import ResNet


def generate_and_save_gradcam(args):
    print_args(args, func="generate_and_save_gradcam")
    device = set_device(args.device)

    # Define dataset-specific parameters
    if args.dataset == "dataset_elastic":
        label_var = "Ey"
        normalize = "mean_std"
        n_target_layer = args.n_target_layer if args.n_target_layer else 0
    elif args.dataset == "dataset_damage":
        label_var = "toughness"
        normalize = "negative_mean_std"
        n_target_layer = args.n_target_layer if args.n_target_layer else 4
    else:
        raise ValueError("Unsupported dataset")

    # Load data
    dataloaders, _ = get_dataloaders(
        dataset=args.dataset,
        input_var="reinforcement",
        label_var=label_var,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=normalize,
        device=device,
        return_id=True,
    )

    # Load pretrained predictor
    model = ResNet(
        n_channels=1,
        n_labels=1,
        resolution=512,
        initial_channels=args.initial_channels,
        ndown=args.ndown,
        mlp_hidden_layers=args.mlp_hidden_layers,
        leakyrelu=args.leakyrelu,
        norm_type=args.norm_type,
    ).to(device)

    ckpt_path = get_saved_models_path(f"predictor_{label_var}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Register hooks and initialize GradCAM++
    target_layer = model.downs[n_target_layer]
    gradcam = GradCAMplusplus(model=model, target_layer=target_layer)

    # Output path setup
    output_folder = get_gradcam_folder(args.dataset)

    # Run on all data (train + val)
    fig_counter = 0
    for split_name, loader in dataloaders.items():

        for idx, (inputs, labels, sim_ids) in enumerate(
            tqdm(loader, desc=f"{split_name}")
        ):
            inputs = inputs.to(device)
            masks, _ = gradcam(inputs)

            for i, sim_id in enumerate(sim_ids):
                mask = masks[i, 0].cpu().numpy()
                input_img = inputs[i, 0].detach().cpu().numpy()

                # Save raw mask as .npy
                mask_save_path = output_folder / f"{sim_id}.npy"
                np.save(mask_save_path, mask)

                fig_counter += 1

                if fig_counter % args.save_figures_each != 0:
                    continue

                # Plot input and mask side by side
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(input_img, cmap="gray")
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                axes[1].imshow(mask, cmap="jet")
                axes[1].set_title("GradCAM++ Heatmap")
                axes[1].axis("off")

                # Save the figure
                fig_path = get_saved_figure_path(f"{args.dataset}_{sim_id}_gradcam.png")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150)
                plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GradCAM++ maps")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dataset_elastic", "dataset_damage"],
        default="dataset_damage",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # ResNet-specific arguments
    parser.add_argument("--initial-channels", type=int, default=16)
    parser.add_argument("--ndown", type=int, default=7)
    parser.add_argument("--mlp-hidden-layers", type=int, default=256)
    parser.add_argument("--leakyrelu", action="store_true")
    parser.add_argument("--norm-type", type=str, default="batch")
    parser.add_argument(
        "--n_target_layer", type=int, default=None, help="Target layer for GradCAM++"
    )
    parser.add_argument(
        "--save-figures-each",
        type=int,
        default=100,
        help="Save figures every N cases",
    )

    args = parser.parse_args()
    generate_and_save_gradcam(args)
