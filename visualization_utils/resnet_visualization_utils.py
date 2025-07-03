import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt

def normalize_tensor(t):
    t_min = t.min()
    t_max = t.max()
    return (t - t_min) / (t_max - t_min + 1e-6)


def overlay_heatmap_on_image(img_tensor, heatmap_tensor, alpha=0.5, cmap='jet'):
    img_np = to_pil_image(img_tensor.cpu()).convert("RGB")

    heatmap = normalize_tensor(heatmap_tensor).cpu().numpy()
    heatmap_colored = plt.get_cmap(cmap)(heatmap)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)

    heatmap_img = Image.fromarray(heatmap_colored)
    heatmap_img = heatmap_img.resize(img_np.size, resample=Image.BILINEAR)

    blended = Image.blend(img_np, heatmap_img, alpha=alpha)
    return blended


def spatial_activation_progression(spatial_maps, input_image, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(spatial_maps), figsize=figsize)
    for ax, (name, fmap) in zip(axes, spatial_maps.items()):
        img = normalize_tensor(fmap[0]).cpu().numpy()
        ax.imshow(img, cmap='jet')
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("resnet_spatial_activations.png", dpi=600, bbox_inches='tight')
    plt.show()


def image_w_spatial_activation_progression(input_image, focus_images, titles=None, figsize=(12, 8)):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(input_image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(input_image.device)
    denormalized_input = (input_image * std) + mean

    images = [denormalized_input.permute(1, 2, 0).clamp(0, 1).cpu().numpy()] + focus_images
    titles = ["Input image"] + (titles if titles else [""] * len(focus_images))

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i in range(6):
        if i < len(images):
            axes[i].imshow(images[i])
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("resnet_spatial_activations_w_image.png", dpi=600, bbox_inches='tight')
    plt.show()


def channel_activation_progression(channel_maps, figsize=(12, 6)):
    fig, axes = plt.subplots(len(channel_maps), 1, figsize=figsize)
    if len(channel_maps) == 1:
        axes = [axes]
    for ax, (name, cmap) in zip(axes, channel_maps.items()):
        img = normalize_tensor(cmap[0].unsqueeze(0)).cpu().numpy()
        ax.imshow(img, aspect='auto', cmap='viridis')
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("resnet_channel_activations.png", dpi=600, bbox_inches='tight')
    plt.show()
