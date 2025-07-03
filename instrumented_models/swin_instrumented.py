import torch
import torch.nn as nn
import timm

class SwinTiny_InternalRepresentation(nn.Module):
    def __init__(self, num_classes=None, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.attention_maps = {}
        self.channel_activations = {}
        self.token_activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, stage in enumerate(self.model.layers):
            for j, block in enumerate(stage.blocks):
                # Hook for attention
                def make_attn_hook(stage_idx=i, block_idx=j):
                    def hook(module, input, output):
                        attn = module.last_attn_weights.detach().cpu()
                        self.attention_maps[f'stage_{stage_idx}_block_{block_idx}'] = attn
                    return hook

                attn_module = block.attn
                if not hasattr(attn_module, 'original_forward'):
                    attn_module.original_forward = attn_module.forward
                    attn_module.forward = self._make_custom_attn_forward(attn_module)

                self.hooks.append(attn_module.register_forward_hook(make_attn_hook()))

                # Hook for MLP output
                def make_mlp_activation_hook(stage_idx=i, block_idx=j):
                    def hook(module, input, output):
                        # output: [B, N, C]
                        if isinstance(output, tuple):
                            output = output[0]
                        if output.dim() == 3:
                            output = output.detach().cpu()
                            self.token_activations[f'stage_{stage_idx}_block_{block_idx}'] = output
                            self.channel_activations[f'stage_{stage_idx}_block_{block_idx}'] = output.mean(dim=1)
                    return hook

                self.hooks.append(block.mlp.register_forward_hook(make_mlp_activation_hook()))

    def _make_custom_attn_forward(self, attn_module):
        def custom_forward(x, mask=None):
            B_, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B_, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * attn_module.scale
            attn = (q @ k.transpose(-2, -1))

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, attn_module.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, attn_module.num_heads, N, N)

            attn = attn.softmax(dim=-1)
            attn_module.last_attn_weights = attn.detach()

            out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            out = attn_module.proj(out)
            out = attn_module.proj_drop(out)

            return out
        return custom_forward

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def forward(self, x):
        self.attention_maps = {}
        self.channel_activations = {}
        self.token_activations = {}
        return self.model(x)

    def get_attention_maps(self):
        return self.attention_maps

    def get_channel_activations(self):
        return self.channel_activations

    def get_token_activations(self):
        return self.token_activations
