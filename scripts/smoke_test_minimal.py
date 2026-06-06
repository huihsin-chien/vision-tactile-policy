#!/usr/bin/env python3

"""Minimal forward smoke test for ImplicitRDP and SparSH.

Run from the ImplicitRDP repo root:
    /home/user/huihsin/ImplicitRDP/implicitrdp_venv/bin/python scripts/smoke_test_minimal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SPARSH_ROOT = WORKSPACE_ROOT / "sparsh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SPARSH_ROOT) not in sys.path:
    sys.path.insert(0, str(SPARSH_ROOT))

from ImplicitRDP.model.vision.transformer_obs_encoder import TransformerObsEncoder
from tactile_ssl.model.vision_transformer import vit_base, vit_small


def run_implicitrdp_smoke_test() -> tuple[int, ...]:
    shape_meta = {
        "obs": {
            "right_wrist_img": {"shape": [3, 360, 640], "type": "rgb"},
            "right_robot_tcp_pose": {"shape": [9], "type": "low_dim"},
            "right_robot_tcp_wrench": {"shape": [6], "type": "low_dim"},
        }
    }
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta,
        obs_horizon=2,
        model_name="resnet34",
        pretrained=False,
        frozen=False,
        n_emb=768,
        use_group_norm=True,
        share_rgb_model=False,
        feature_aggregation=None,
        downsample_ratio=32,
    )
    encoder.eval()

    obs = {
        "right_wrist_img": torch.zeros(2, 3, 360, 640),
        "right_robot_tcp_pose": torch.zeros(2, 9),
        "right_robot_tcp_wrench": torch.zeros(2, 6),
    }
    with torch.no_grad():
        out = encoder(obs)
    return tuple(out.shape)


def run_sparsh_smoke_test(model_name: str, model: torch.nn.Module) -> tuple[int, ...]:
    model.eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"{model_name} output shape: {tuple(out.shape)}")
    print(f"{model_name} output embedding dim: {out.shape[-1]}")
    print(
        f"{model_name} projection to ImplicitRDP n_emb=768: "
        f"Linear({out.shape[-1]} -> 768)"
    )
    return tuple(out.shape)


def main() -> None:
    print(f"ImplicitRDP repo root: {REPO_ROOT}")
    print(f"SparSH repo root: {SPARSH_ROOT}")

    imp_shape = run_implicitrdp_smoke_test()
    print(f"ImplicitRDP obs encoder output shape: {imp_shape}")

    run_sparsh_smoke_test("vit_small", vit_small())
    run_sparsh_smoke_test("vit_base", vit_base())


if __name__ == "__main__":
    main()