# vision-tactile-policy

> 🚧 **Work in Progress** — data collection ongoing

**UMI + DIGIT tactile adaptation of [ImplicitRDP](https://implicit-rdp.github.io)**  
Independent research — Lydia Chien @ NYCU EE · Jan 2026 – Present

---

## What This Is

This repo adapts ImplicitRDP's slow-fast diffusion policy for a **UMI + DIGIT tactile** setup — replacing the original Flexiv force/torque sensor with dual-DIGIT tactile sensors, and the impedance-control action space with a position-controlled 7-DOF action space compatible with UMI hand-held demonstrations.

**Goal:** learn a contact-aware grasping policy for fragile objects (egg, tofu, test tube), using tactile feedback as the reactive signal.

---

## Key Adaptations from Original ImplicitRDP

| Component | Original (SJTU) | This Fork |
|---|---|---|
| Robot | Flexiv Rizon 4s | UMI hand-held + position-ctrl arm |
| Fast-path sensor | 6-DOF wrist wrench | Dual DIGIT tactile (PCA, dim=64) |
| Action space | 19D (xyz + 6d\_rot + virtual + stiffness) | 7D (xyz + axis\_angle + gripper) |
| Data collection | Kinematic teaching | UMI GoPro + DIGIT sync |
| Tactile encoder | — | SparSH ViT *(planned)* |

### Fast-path GRU

```python
# Original: 6D wrench → GRU → temporal conditioning
GRU(input_dim=6,  hidden_dim=512) → Linear(512→768)

# This fork: dual-DIGIT PCA → GRU → temporal conditioning
GRU(input_dim=64, hidden_dim=512) → Linear(512→768)
# input_dim = pca_dim(32) × 2 DIGIT sensors
```

### Data sync pipeline

Clapperboard-analog sync: pressure spike on DIGIT → detect Δt → align GoPro video + tactile streams.

*Demo: data collection video — coming soon*

---

## Planned: SparSH Tactile Encoder

Cross-attention fusion of SparSH tactile tokens into the slow-path Diffusion Transformer:
- Assign `SparshViTWrapper` in `TransformerObsEncoder` for DIGIT image keys
- Output: `(BT, 196, 384)` tokens → projected to 768-dim, concat with visual tokens

---

## Evaluation (planned)

Ablation: **w/ tactile** vs **w/o tactile** on fragile object grasping  
Objects: egg · tofu · test tube

---

## Base Paper

> Chen et al., "ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning," arXiv:2512.10946, 2025.  
> Project: https://implicit-rdp.github.io🚧——·–——
