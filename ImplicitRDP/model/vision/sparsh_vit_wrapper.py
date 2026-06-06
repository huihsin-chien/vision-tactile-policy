"""
SparshViTWrapper — wraps SparSH DINO-small ViT for use inside TransformerObsEncoder.

Checkpoint facts (facebook/sparsh-dino-small):
  - in_chans=6 (2 stacked frames during pretraining)
  - num_register_tokens=1, pos_embed_fn='sinusoidal'
  - Teacher backbone key prefix: 'teacher_encoder.backbone.'
  - forward() output: (B, 196, 384)  [register token stripped internally]

At inference we duplicate the single 3-ch tactile image → 6-ch input,
which simulates the zero-motion case and keeps the pretrained weight intact.
"""

import sys
import os
import torch
import torch.nn as nn

SPARSH_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'sparsh')

class SparshViTWrapper(nn.Module):
    """
    Thin wrapper so TransformerObsEncoder can use SparSH ViT as a backbone.

    Input:  (B, 3, 224, 224)  float32 in [0, 1]
    Output: (B, 196, 384)     patch tokens
    """

    model_name = 'vit_sparsh_small'   # starts with 'vit' → aggregate_feature treats as ViT

    def __init__(self, ckpt_path: str, frozen: bool = True):
        super().__init__()

        if str(os.path.abspath(SPARSH_ROOT)) not in sys.path:
            sys.path.insert(0, str(os.path.abspath(SPARSH_ROOT)))

        from tactile_ssl.model.vision_transformer import vit_small

        self.vit = vit_small(
            in_chans=6,
            num_register_tokens=1,
            pos_embed_fn='sinusoidal',
        )

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        prefix = 'teacher_encoder.backbone.'
        backbone_sd = {
            k[len(prefix):]: v
            for k, v in ckpt['model'].items()
            if k.startswith(prefix)
        }
        missing, unexpected = self.vit.load_state_dict(backbone_sd, strict=False)
        assert len(missing) == 0, f"Missing keys loading SparSH: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys loading SparSH: {unexpected}"

        if frozen:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → duplicate to 6-ch for pretrained patch embedding
        x6 = torch.cat([x, x], dim=1)          # (B, 6, H, W)
        return self.vit(x6)                      # (B, 196, 384)
