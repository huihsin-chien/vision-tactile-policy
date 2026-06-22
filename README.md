# vision-tactile-policy

> 🚧 **Work in Progress** — currently in data collection phase.

Adapting [ImplicitRDP](https://implicit-rdp.github.io/)'s slow-fast diffusion policy to use **UMI hand-held demonstrations** with **dual DIGIT tactile sensors** — targeting contact-aware grasping of fragile objects (egg, tofu, test tube).

## Motivation

Fragile object manipulation requires knowing *how hard* you're gripping. Visual feedback alone isn't enough — this project adds tactile sensing to the fast-path temporal conditioning of ImplicitRDP, replacing the Flexiv-specific wrist F/T sensor with commodity tactile sensors.

## Architecture

ImplicitRDP uses a **slow-fast** diffusion policy:
- **Slow path**: diffusion transformer over visual + proprioceptive observations
- **Fast path**: GRU that conditions on high-frequency sensor data → temporal embedding injected into the slow path

Our adaptation replaces the fast-path sensor:

```
Original (Flexiv):  wrench F/T (6D)       → GRU(6, 512)  → Linear(512→768)
Ours (UMI+DIGIT):   dual-DIGIT PCA (64D)  → GRU(64, 512) → Linear(512→768)
                    └── pca_dim=32 × 2 sensors
```

Action space: **19D** (Flexiv joint impedance) → **7D** (xyz + axis_angle + gripper) for position-controlled arms.

```python
# config/task/umi_digit.yaml
action_dim: 7            # xyz(3) + axis_angle(3) + gripper(1)
tactile_pca_dim: 32      # per DIGIT sensor
fast_path_input_dim: 64  # 32 * 2 sensors
```

## Data Collection

**Hardware**: UMI gripper + GoPro (3rd-person) + 2× DIGIT tactile sensors (fingertips)

**Sync**: Pressure spike on DIGIT at grasp start → detect Δt → align GoPro video + dual tactile streams offline.

**Target objects**: egg · tofu · test tube

## Planned: SparSH Integration

Replace the `timm` ViT backbone in `TransformerObsEncoder` with a `SparshViTWrapper` for DIGIT image keys — cross-attention fusion of tactile tokens with visual tokens into the slow-path diffusion transformer.

```python
# Instead of:
encoder = TimmViT(model_name="vit_base_patch16_224")

# Use:
encoder = SparshViTWrapper(
    pretrained="sparsh-base",
    tactile_keys=["digit_left", "digit_right"],
    fusion="cross_attention",
)
```

## Evaluation (planned)

Ablation study: **w/ tactile** vs. **w/o tactile** on fragile grasping.

| Object    | w/ tactile | w/o tactile |
|-----------|------------|-------------|
| Egg       | TBD        | TBD         |
| Tofu      | TBD        | TBD         |
| Test tube | TBD        | TBD         |

## Repository Structure

```
vision-tactile-policy/
├── config/
│   ├── task/umi_digit.yaml         # action/obs space config
│   └── model/slow_fast_digit.yaml  # fast-path GRU config
├── models/
│   ├── fast_path_gru.py            # dual-DIGIT PCA → GRU
│   ├── sparsh_wrapper.py           # (planned) SparSH ViT adapter
│   └── policy.py                   # slow-fast diffusion policy
├── data_collection/
│   ├── sync_streams.py             # GoPro + DIGIT sync via pressure spike
│   └── pca_fit.py                  # fit PCA on DIGIT tactile images
└── CLAUDE.md                       # technical context for AI assistants
```

## Dependencies

```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/digit-interface
pip install git+https://github.com/columbia-ai-robotics/umi
# SparSH (planned)
pip install git+https://github.com/facebookresearch/sparsh
```

## Related Work

- [ImplicitRDP](https://implicit-rdp.github.io/) — original slow-fast diffusion policy (Flexiv)
- [UMI](https://umi-gripper.github.io/) — hand-held demonstration system
- [DIGIT](https://digit.ml/) — compact tactile sensor by Meta AI
- [SparSH](https://sparsh-ssl.github.io/) — tactile representation learning

## Status

| Component | Status |
|-----------|--------|
| Architecture adaptation (7D action, dual-DIGIT GRU) | ✅ Done |
| Data collection pipeline (sync) | ✅ Done |
| Training data collection | 🔄 In progress |
| Policy training | ⏳ Pending |
| SparSH integration | ⏳ Planned |
| Real-robot evaluation | ⏳ Planned |

---

**Lydia Chien** · NYCU EE · Independent Research · 2026
