import math
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

def normalize_tensor(t):
    t_min = t.min()
    t_max = t.max()
    if t_max - t_min < 1e-6:
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def spatial_attention_progression(attn_dict, cols=6, figsize_per_plot=(4, 4)):
    items = list(attn_dict.items())
    num_items = len(items)
    rows = math.ceil(num_items / cols)
    figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_items > 1 else [axes]

    for ax, (name, attn_tensor) in zip(axes, items):
        try:
            received_attention = attn_tensor.mean(1).mean(1)  # [B, N]
            num_windows = attn_tensor.shape[0]
            num_tokens = received_attention.shape[1]
            spatial_tokens = int(num_tokens ** 0.5)

            h_windows = w_windows = int(num_windows ** 0.5)
            attn_2d = received_attention.reshape(h_windows, w_windows, spatial_tokens, spatial_tokens)
            attn_2d = attn_2d.permute(0, 2, 1, 3).reshape(h_windows * spatial_tokens, w_windows * spatial_tokens)

            ax.imshow(normalize_tensor(attn_2d), cmap='jet')
            ax.set_title(name)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"Failed: {name}")
            ax.axis('off')

    # Hide any unused subplots
    for i in range(len(items), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("swin_spatial_attentions.png", dpi=600, bbox_inches='tight')
    plt.show()


def image_w_spatial_attention_progression(input_tensor, attn_dict, cols=3, alpha=0.5, figsize_per_plot=(4, 4)):
    def denormalize(img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        return (img * std + mean).clamp(0, 1)

    def tensor_to_pil(img_tensor):
        return to_pil_image(img_tensor.squeeze(0).cpu())  # [1, 3, H, W] -> [3, H, W]

    def apply_overlay(base_img_tensor, heatmap_tensor):
        heatmap_tensor = normalize_tensor(heatmap_tensor)
        heatmap_np = heatmap_tensor.detach().cpu().numpy()
        if heatmap_np.ndim != 2:
            raise ValueError(f"Expected 2D heatmap, got shape {heatmap_np.shape}")

        heatmap_color = plt.cm.jet(heatmap_np)[:, :, :3]  # RGBA -> RGB
        heatmap_img = Image.fromarray((heatmap_color * 255).astype(np.uint8)).resize(base_img_tensor.shape[-2:][::-1])

        base_img = tensor_to_pil(base_img_tensor)
        return Image.blend(base_img, heatmap_img, alpha=alpha)

    items = list(attn_dict.items())
    num_items = len(items)
    rows = math.ceil(num_items / cols)
    figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_items > 1 else [axes]

    base_img_tensor = denormalize(input_tensor[0])  # remove batch dim

    for ax, (name, attn_tensor) in zip(axes, items):
        try:
            received_attention = attn_tensor.mean(1).mean(1)  # [B, N]
            num_windows = attn_tensor.shape[0]
            num_tokens = received_attention.shape[1]
            spatial_tokens = int(num_tokens ** 0.5)

            h_windows = w_windows = int(num_windows ** 0.5)
            attn_2d = received_attention.reshape(h_windows, w_windows, spatial_tokens, spatial_tokens)
            attn_2d = attn_2d.permute(0, 2, 1, 3).reshape(h_windows * spatial_tokens, w_windows * spatial_tokens)

            overlay = apply_overlay(base_img_tensor, attn_2d)
            ax.imshow(overlay)
            ax.set_title(name)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"Failed: {name}")
            ax.axis('off')
            print(f"[ERROR] {name}: {e}")

    for i in range(len(items), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("swin_spatial_attentions_w_image.png", dpi=600, bbox_inches='tight')
    plt.show()


def channel_activation_progression(channel_activation_dict, figsize=(14, 2)):
    # Group activations by stage
    stage_activations = defaultdict(list)
    for name, act in channel_activation_dict.items():
        if act is None or act.numel() == 0:
            continue
        # Parse stage from name like "stage_1_block_2"
        stage_name = "_".join(name.split("_")[:2])
        stage_activations[stage_name].append(act)

    # Average across blocks within each stage
    averaged_stage_activations = {}
    for stage, acts in stage_activations.items():
        stacked = torch.stack(acts)
        averaged = stacked.mean(dim=0)
        averaged_stage_activations[stage] = averaged

    # Sort by stage index
    sorted_items = sorted(averaged_stage_activations.items(), key=lambda x: int(x[0].split('_')[1]))

    num_stages = len(sorted_items)
    fig, axes = plt.subplots(num_stages, 1, figsize=(figsize[0], figsize[1] * num_stages))

    if num_stages == 1:
        axes = [axes]

    for ax, (stage, act) in zip(axes, sorted_items):
        act = (act - act.min()) / (act.max() - act.min() + 1e-6)
        act = act.flatten().unsqueeze(0)  # ensures shape (1, C)
        ax.imshow(act.numpy(), aspect='auto', cmap='viridis')
        ax.set_title(stage)
        ax.set_ylabel("Layer")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("swin_channel_activations.png", dpi=600, bbox_inches='tight')
    plt.show()


def attention_matrix_progression(attn_maps, figsize=(16, 4)):
    """
    Aggregates attention maps per stage and plots one mean attention map per stage.
    """

    def group_attn_by_stage(attn_maps):
        """
        Groups attention tensors by stage name from a flat attn_maps dict.
        Returns: dict { "stage_0": [tensor, tensor, ...], ... }
        """
        stagewise_attn_dict = defaultdict(list)
        for name, attn in attn_maps.items():
            stage_name = "_".join(name.split("_")[:2])
            stagewise_attn_dict[stage_name].append(attn)
        return stagewise_attn_dict

    stagewise_attn_dict = group_attn_by_stage(attn_maps)
    stage_items = sorted(stagewise_attn_dict.items(), key=lambda x: int(x[0].split('_')[1]))

    fig, axes = plt.subplots(1, len(stage_items), figsize=figsize)
    if len(stage_items) == 1:
        axes = [axes]

    for ax, (stage_name, attn_list) in zip(axes, stage_items):
        try:
            stacked = torch.stack(attn_list)  # (num_blocks, B, H, N, N)
            attn_tensor = stacked.mean(dim=0)  # (B, H, N, N)
            avg_attn = attn_tensor.mean(0).mean(0)  # (N, N)

            ax.imshow(normalize_tensor(avg_attn), cmap='viridis')
            ax.set_title(stage_name)
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"Failed: {stage_name}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("swin_attention_matrices.png", dpi=600, bbox_inches='tight')
    plt.show()
