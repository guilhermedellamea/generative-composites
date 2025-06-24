import torch
import torch.nn.functional as F

"""
Adapted from: https://github.com/1Konny/gradcam_plus_plus-pytorch
"""


class GradCAMplusplus:
    def __init__(self, model, target_layer):

        self.model = model

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients["value"] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations["value"] = output
            return None

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        return self.forward(input_tensor, class_idx, retain_graph)

    def forward(
        self,
        input_tensor,
        class_idx=0,
        normalize=True,
    ):
        """
        Computes GradCAM++ saliency masks.

        Args:
            input_tensor (Tensor): Input images of shape (B, C, H, W)
            class_idx (int or list[int], optional): Target class indices
            normalize (bool): Normalize mask between 0 and 1

        Returns:
            masks (Tensor): Saliency maps of shape (B, 1, H, W)
            logits (Tensor): Model outputs of shape (B, num_classes)
        """
        batch_size, _, height, width = input_tensor.size()
        logits = self.model(input_tensor)

        class_idx = [class_idx] * batch_size

        masks = []

        for i in range(batch_size):

            self.model.zero_grad()
            target_score = logits[i, class_idx[i]].squeeze()
            target_score.backward(retain_graph=True)

            grads = self.gradients["value"][i].unsqueeze(0)  # dS/dA
            activs = self.activations["value"][i].unsqueeze(0)  # A
            _, channels, h_feat, w_feat = grads.size()

            # Compute alpha weights
            alpha_num = grads.pow(2)
            activs_sum = activs.view(1, channels, -1).sum(-1).view(1, channels, 1, 1)
            alpha_denom = 2 * grads.pow(2) + activs_sum * grads.pow(3)
            alpha_denom = torch.where(
                alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom)
            )
            alphas = alpha_num / (alpha_denom + 1e-7)

            # Weights are alpha * ReLU(grad)
            positive_grads = F.relu(grads)
            weights = (
                (alphas * positive_grads)
                .view(1, channels, -1)
                .sum(-1)
                .view(1, channels, 1, 1)
            )

            # Generate saliency map
            saliency = (weights * activs).sum(1, keepdim=True)
            saliency = F.relu(saliency)
            saliency = F.interpolate(
                saliency, size=(height, width), mode="bilinear", align_corners=False
            )

            if normalize:
                saliency_min = saliency.min()
                saliency_max = saliency.max()
                if saliency_max > saliency_min:
                    saliency = (saliency - saliency_min) / (saliency_max - saliency_min)

            masks.append(saliency.detach())

        masks = torch.cat(masks, dim=0)
        return masks, logits.detach()
