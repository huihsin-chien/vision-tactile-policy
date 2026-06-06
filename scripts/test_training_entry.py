#!/usr/bin/env python3
"""
Dry-run test: dataset → normalizer → one forward pass → loss.

Verifies 6/4 (config) and 6/5 (data loader) end-to-end without starting
a full training run. No GPU required for shape checks; moves to CUDA if
available.

Run:
    source /home/user/huihsin/ImplicitRDP/implicitrdp_venv/bin/activate
    cd /home/user/huihsin/ImplicitRDP
    python scripts/test_training_entry.py
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OmegaConf.register_new_resolver("eval", eval, replace=True)
CONFIG_DIR = str(REPO_ROOT / "ImplicitRDP" / "config")


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train_umi_digit_implicitrdp_workspace")
    OmegaConf.resolve(cfg)

    print("=== Config resolved OK ===")
    print(f"  name:        {cfg.name}")
    print(f"  task:        {cfg.task_name}")
    print(f"  action_dim:  {cfg.task.shape_meta.action.shape}")
    print(f"  horizon:     {cfg.horizon}")
    print(f"  n_obs_steps: {cfg.n_obs_steps}")
    print(f"  use_rnn:     {cfg.policy.use_rnn_obs_encoder}")

    # ── 1. Dataset ────────────────────────────────────────────────────
    print("\n=== Dataset ===")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"  Train samples: {len(dataset)}")
    item = dataset[0]
    print("  obs keys / shapes:")
    for k, v in item["obs"].items():
        print(f"    {k}: {tuple(v.shape)}")
    print("  extended_obs keys / shapes:")
    for k, v in item["extended_obs"].items():
        print(f"    {k}: {tuple(v.shape)}")
    print(f"  action: {tuple(item['action'].shape)}")

    # ── 2. DataLoader — one batch ──────────────────────────────────────
    print("\n=== DataLoader ===")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("  batch['obs'] shapes:")
    for k, v in batch["obs"].items():
        print(f"    {k}: {tuple(v.shape)}")
    print("  batch['extended_obs'] shapes:")
    for k, v in batch["extended_obs"].items():
        print(f"    {k}: {tuple(v.shape)}")
    print(f"  batch['action']: {tuple(batch['action'].shape)}")

    # ── 3. Normalizer ─────────────────────────────────────────────────
    print("\n=== Normalizer ===")
    normalizer = dataset.get_normalizer()
    print(f"  Keys: {list(normalizer.params_dict.keys())}")

    # ── 4. Policy forward + loss ──────────────────────────────────────
    print("\n=== Policy ===")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    policy = policy.to(device)

    from ImplicitRDP.common.pytorch_util import dict_apply
    batch = dict_apply(batch, lambda x: x.to(device))

    policy.eval()
    with torch.no_grad():
        loss = policy.compute_diffusion_loss(batch)
    print(f"  compute_diffusion_loss: {loss.item():.4f}  ✅")

    # Also test predict_action (reactive path)
    obs_dict = batch["obs"]
    extended_obs_dict = batch["extended_obs"]
    result = policy.predict_action(obs_dict, extended_obs_dict=extended_obs_dict)
    print(f"  predict_action output shape: {tuple(result['action_pred'].shape)}  ✅")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
