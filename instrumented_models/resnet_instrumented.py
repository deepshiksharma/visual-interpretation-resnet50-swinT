import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

class ResNet50_InternalRepresentation(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super(ResNet50_InternalRepresentation, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        self.activations = OrderedDict()
        self.hooks = []

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.focus_layers = OrderedDict({
            "conv1": self.model.conv1,
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        })

        self._register_hooks()

    def _save_activation(self, name):
        def hook(module, input, output):
            if not self.training:
                self.activations[name] = output.detach().cpu()
        return hook

    def _register_hooks(self):
        for name, layer in self.focus_layers.items():
            hook = layer.register_forward_hook(self._save_activation(name))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activation_maps(self):
        spatial_maps = OrderedDict()
        channel_maps = OrderedDict()
        for name, act in self.activations.items():
            spatial_map = act.mean(dim=1)         # [B, H, W]
            channel_map = act.mean(dim=(2, 3))    # [B, C]
            spatial_maps[name] = spatial_map
            channel_maps[name] = channel_map
        return spatial_maps, channel_maps

    def forward(self, x):
        return self.model(x)
