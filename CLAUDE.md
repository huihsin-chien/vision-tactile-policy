# CLAUDE.md — ImplicitRDP Repo Guide

Paper: "ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning"
arXiv: 2512.10946 | Project: https://implicit-rdp.github.io

---

## What ImplicitRDP Is

A **Reactive Diffusion Policy** that uses a slow-fast architecture:
- **Slow path**: Diffusion Transformer generates an action chunk conditioned on RGB visual tokens.
- **Fast path**: GRU-based RNN encodes high-frequency force-torque (wrench) history as temporal conditioning, allowing the diffusion model to reactively adjust actions mid-execution.

The key innovation over vanilla Diffusion Policy: the wrench RNN provides real-time force feedback *inside* the diffusion sampling loop, so the policy can handle contact-rich manipulation without rerunning the full diffusion from scratch.

---

## Repo Structure

```
ImplicitRDP/
├── ImplicitRDP/              # main Python package
│   ├── config/               # Hydra configs (see below)
│   ├── dataset/              # RealImageTactileDataset, zarr-based
│   ├── env_runner/           # real_stable_reactive_runner.py (eval loop)
│   ├── model/
│   │   ├── diffusion/        # TransformerForDiffusion, UNet, EMA, DDIM
│   │   ├── force/            # RNN (GRU for wrench temporal conditioning)
│   │   ├── vae/              # VAE for latent action (used by AT policy)
│   │   ├── vision/           # TransformerObsEncoder (main visual encoder)
│   │   └── common/           # normalizer, lr scheduler, etc.
│   ├── policy/               # DiffusionTransformerImagePolicy (main), UNet, LDP variants
│   ├── real_world/           # robot server, camera publishers, kineteach controller
│   └── workspace/            # training loop classes (TrainDiffusionTransformerImageWorkspace, etc.)
├── scripts/                  # smoke_test_minimal.py and other utils
├── control.py                # real-robot inference entry point
├── eval_real_robot_flexiv.py # eval entry point
└── train_implicitrdp.sh      # training launch script
```

---

## Core Classes

### `TransformerObsEncoder` — `ImplicitRDP/model/vision/transformer_obs_encoder.py`

Encodes all observation modalities into a unified `(B, N_tokens, n_emb=768)` sequence.

- For `type: rgb` keys: runs a timm/ViT backbone → `aggregate_feature()` → `Linear(feat_dim→768)`
- For `type: low_dim` keys: `Linear(dim→768)`
- All tokens are concatenated along the token dimension
- `share_rgb_model=False` (default): separate backbone per camera

**Integration point for SparSH**: replace the timm model in `key_model_map[tactile_key]` with a SparSH VisionTransformer.

### `TransformerForDiffusion` — `ImplicitRDP/model/diffusion/transformer_for_diffusion.py`

Decoder-only Transformer that denoises an action trajectory.

- Input: noisy action `(B, horizon, action_dim)` + positional embedding
- Condition: obs tokens `(B, N, 768)` + optional temporal_cond `(B, T, 768)` from RNN
- Architecture: 7 decoder layers, 4 heads, n_emb=768
- Inference: DDIM, 5 steps (fast)

### `RNN` — `ImplicitRDP/model/force/rnn.py`

Simple GRU encoder for high-frequency sensor streams (wrench, or DIGIT PCA in our case).

```python
# Original ImplicitRDP (wrench):
GRU(input_dim=6, hidden_dim=512, num_layers=1) → Linear(512→768)

# Our adaptation (dual-DIGIT PCA, input_dim auto-derived from shape_meta.extended_obs):
GRU(input_dim=64, hidden_dim=512, num_layers=1) → Linear(512→768)
# input_dim=64 = pca_dim(32) × 2 sensors; concatenated by get_temporal_cond()
# output: (B, T, 768) temporal conditioning tokens
```

### `DiffusionTransformerImagePolicy` — `ImplicitRDP/policy/diffusion_transformer_image_policy.py`

Ties everything together. Key methods:
- `predict_action(obs_dict, extended_obs_dict)` — full inference
- `predict_obs_tokens(obs_dict)` — encode obs only (for reactive fast-path)
- `predict_from_obs_tokens_and_noisy_trajectory(...)` — decode given pre-computed obs tokens
- `compute_diffusion_loss(batch)` — training loss (MSE on epsilon prediction)
- `get_temporal_cond(extended_obs_dict)` — wrench → GRU → temporal conditioning

---

## Config System (Hydra)

### Config folder structure

```
config/
├── train_reactive_diffusion_transformer_real_image_workspace.yaml  ← ImplicitRDP (main)
├── train_diffusion_transformer_real_image_workspace.yaml           ← plain DPT baseline
├── train_diffusion_unet_real_image_workspace.yaml                  ← DP-UNet baseline
├── train_latent_diffusion_unet_real_image_workspace.yaml           ← LDP baseline
├── train_at_workspace.yaml                                         ← AT/VAE training
├── task/
│   ├── real_flip_image_wrench_implicitrdp_10fps.yaml   ← flip task, wrench+img, ImplicitRDP
│   ├── real_flip_image_wrench_dpt_10fps.yaml           ← flip task, DPT baseline
│   ├── real_toggle_image_wrench_implicitrdp_10fps.yaml ← toggle task, ImplicitRDP
│   ├── real_robot_env.yaml                             ← base robot/camera hardware config
│   └── real_flip_one_usb_camera_kineteach_10fps.yaml   ← flip hardware config
├── at/
│   ├── at_flip.yaml    ← VAE HPs for flip task
│   └── at_toggle.yaml  ← VAE HPs for toggle task
├── real_world_env.yaml   ← top-level env entry (shortcut)
└── legacy/               ← old peel/wipe/lift experiments (do not use)
```

### Task config naming convention

`real_{task}_{sensors}_{policy}_{fps}.yaml`

- `task`: `flip` | `toggle`
- `sensors`: `image` | `image_wrench` | `image_dp` | `image_wrench_implicitrdp` | etc.
- `policy`: `dp` | `dpt` | `ldp` | `at` | `implicitrdp` | `implicitrdp_6d` | `implicitrdp_fp`
- `fps`: `10fps`

### shape_meta layout (task yaml)

```yaml
shape_meta:
  obs:
    right_wrist_img:       {shape: [3, 360, 640], type: rgb}
    right_robot_tcp_pose:  {shape: [9], type: low_dim}
    right_robot_tcp_wrench:{shape: [6], type: low_dim}
  extended_obs:            # for the wrench RNN (fast path)
    right_robot_tcp_wrench:{shape: [6], type: low_dim}
  action:
    shape: [19]  # or [13] in model_shape_meta with RPY

# To add tactile:
#   right_digit_img: {shape: [3, 224, 224], type: rgb}
```

---

## Action Space

### Original ImplicitRDP (Flexiv Rizon 4s, impedance control)

| Config | Dims | Format |
|---|---|---|
| `shape_meta.action` | 19 | xyz(3)+6d_rot(6)+virtual_xyz(3)+virtual_6d_rot(6)+stiffness(1) |
| `model_shape_meta.action` | 13 | xyz(3)+RPY(3)+virtual_xyz(3)+virtual_RPY(3)+stiffness(1) |

`virtual_xyz/RPY` = spring anchor for Cartesian Impedance Control.
`stiffness` = spring constant K (F = K × (virtual − real_TCP)).
`rpy_for_rotation: True` — uses RPY, not 6D rotation.

### Our adaptation (UMI + position control, no F/T sensor)

| Config | Dims | Format |
|---|---|---|
| `shape_meta.action` | 7 | xyz(3)+axis_angle(3)+gripper_width(1) |

`virtual_xyz/RPY/stiffness` are removed — not applicable without Flexiv impedance control.
Contact awareness comes from dual-DIGIT fast-path GRU instead.
`rpy_for_rotation: False`, `relative_action: False`.

---

## Training

```bash
source ImplicitRDP/implicitrdp_venv/bin/activate
cd ImplicitRDP
bash train_implicitrdp.sh   # wraps: python train.py --config-name train_reactive_diffusion_transformer_real_image_workspace task=real_flip_image_wrench_implicitrdp_10fps
```

Key training HPs (from workspace yaml):
- `lr=3e-4`, `obs_encoder_lr=3e-5` (slower LR for backbone)
- `batch_size=64`, `num_epochs=600`, `use_ema=True`
- `checkpoint_every=10`, `val_every=1`

---

## Adding SparSH Tactile Encoder (Integration Checklist)

1. **task yaml** — add a new `type: rgb` key under `shape_meta.obs` for the tactile image
2. **`TransformerObsEncoder`** — in `key_model_map`, assign a `SparshViTWrapper` (wraps SparSH VisionTransformer and returns `(BT, 196, 384)` tokens) for that key
3. **`key_projection_map`** — auto-created as `Linear(384→768)` if `feature_size != n_emb`
4. **`aggregate_feature()`** — treat SparSH output the same as ViT (return all patch tokens, `feature_aggregation=None`)
5. **`extended_obs`** — optionally add tactile embedding to fast-path RNN instead of or alongside wrench

---

## Real-World Deployment Stack

```
control.py / eval_real_robot_flexiv.py
  └─ RealStableReactiveRunner (env_runner/real_stable_reactive_runner.py)
       ├─ bimanual_robot_publisher.py  (Flexiv robot state at 10fps)
       ├─ usb_camera_publisher.py      (wrist camera at 10fps)
       ├─ realsense_camera_publisher.py (optional)
       ├─ gelsight_camera_publisher.py  (optional tactile)
       └─ mctac_camera_publisher.py     (optional tactile)
```

Robot: Flexiv Rizon 4s, server at `192.168.2.x`
Controller: KineteachController at port 8082

---

## Datasets

- Format: zarr (`ReplayBuffer` wrapper in `ImplicitRDP/common/replay_buffer.py`)
- Dataset class: `RealImageTactileDataset` (`dataset/real_image_tactile_dataset.py`)
- Paths in task configs: `data/flip_v1_zarr`, `data/toggle_v1_zarr`
- Train/val split: `val_ratio=0.1`, seed=42
- `relative_action=True`, `delta_action=False`
