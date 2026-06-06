#!/usr/bin/env python3
"""
Convert voilab dataset.zarr.zip → ImplicitRDP replay_buffer.zarr

Usage (must use voilab venv which has imagecodecs_jpegxl):
  /home/user/huihsin/voilab/.venv/bin/python scripts/zarr_convert.py \
      --src /home/user/huihsin/voilab/video/dataset.zarr.zip \
      --dst /home/user/huihsin/voilab/video/umi_digit_zarr \
      --pca-dim 32

Key mapping (both DIGIT sensors):
  camera0_rgb              → wrist_rgb         (H,W,C uint8, blosc)
  robot0_tactile_left      → digit_img_left    (H,W,C uint8, blosc)
  robot0_tactile_right     → digit_img_right   (H,W,C uint8, blosc)
  eef_pos+rot_aa+gripper   → robot_tcp_pose    (7-dim float32)
  same                     → action            (7-dim float32)
  PCA(left@64×64)          → digit_pca_left    (pca_dim float32) ← fast-path GRU
  PCA(right@64×64)         → digit_pca_right   (pca_dim float32) ← fast-path GRU
  episode_ends             → fixed (drop trailing 0)

PCA design:
  - Separate PCA per sensor (left/right contact patterns differ semantically)
  - Input: 64×64×3 = 12288 dims (INTER_AREA downsample from 224×224)
  - pca_dim=32 per sensor → 96.2% variance explained (current 2-demo dataset)
  - Total fast-path GRU input_dim = 32+32 = 64
  - Refit PCA when dataset grows to ~150 demos (Phase 3)
  - PCA matrices saved as digit_pca_left.npz / digit_pca_right.npz in dst/

Action space (7-dim, no virtual target / stiffness):
  action = [eef_xyz(3), eef_axis_angle(3), gripper_width(1)]
  No virtual_xyz/RPY/stiffness — those are Flexiv impedance-control specific.
  We use position control; contact awareness comes from DIGIT fast-path instead.
"""

import argparse, os
import numpy as np
import zarr
import cv2
from numcodecs import Blosc
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Register imagecodecs codecs (needed for imagecodecs_jpegxl)
try:
    import imagecodecs.numcodecs
    imagecodecs.numcodecs.register_codecs()
except Exception as e:
    print(f"Warning: could not register imagecodecs: {e}")


THUMB = 64   # resize to 64×64 before PCA (= 64*64*3 = 12288 dims)
TACTILE_KEYS = ['robot0_tactile_left', 'robot0_tactile_right']
OUT_IMG_KEYS = ['digit_img_left', 'digit_img_right']
OUT_PCA_KEYS = ['digit_pca_left', 'digit_pca_right']


def load_src_zarr(src_path: str):
    store = zarr.ZipStore(src_path, mode='r')
    return zarr.open(store, mode='r')


def fix_episode_ends(raw: np.ndarray, total_frames: int) -> np.ndarray:
    """Drop trailing zeros; verify last value equals total_frames."""
    ends = raw[raw > 0]
    assert ends[-1] == total_frames, \
        f"episode_ends last value {ends[-1]} != total_frames {total_frames}"
    return ends


def make_tcp_pose(eef_pos, eef_rot_aa, gripper):
    """Concatenate EEF state → 7-dim absolute pose."""
    return np.concatenate([eef_pos, eef_rot_aa, gripper], axis=-1).astype(np.float32)


def thumb_batch(images: np.ndarray, size: int = THUMB) -> np.ndarray:
    """(N, H, W, 3) uint8 → (N, size*size*3) float32 in [0,1]."""
    out = np.empty((len(images), size * size * 3), dtype=np.float32)
    for i, img in enumerate(images):
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = small.flatten().astype(np.float32) / 255.0
    return out


def fit_pca_single(src_root, src_key: str, out_key: str,
                   pca_dim: int, plot_dir: str):
    """Fit PCA on one DIGIT sensor, return (pca, embeddings)."""
    print(f"[PCA] {src_key} → {out_key}: loading {THUMB}×{THUMB} thumbnails …")
    raw = src_root['data'][src_key]
    N = raw.shape[0]
    X = thumb_batch(raw[:], THUMB)   # (N, 12288)

    print(f"[PCA] Fitting PCA: {N} frames → {pca_dim} components …")
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Explained variance:")
    for k in [8, 16, 32, 64]:
        if k <= pca_dim:
            print(f"    {k:3d} components: {cumvar[k-1]*100:.1f}%")

    # Save explained-variance plot per sensor
    fig, ax = plt.subplots()
    ax.plot(range(1, pca_dim + 1), cumvar * 100)
    ax.axhline(95, color='r', linestyle='--', label='95%')
    ax.set_xlabel('n_components')
    ax.set_ylabel('Cumulative explained variance (%)')
    ax.set_title(f'DIGIT PCA — {out_key}')
    ax.legend()
    plot_path = os.path.join(plot_dir, f'pca_variance_{out_key}.png')
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  Variance plot → {plot_path}")

    embeddings = pca.transform(X).astype(np.float32)
    return pca, embeddings


def write_dst_zarr(dst_dir: str, src_root,
                   pca_left: np.ndarray, pca_right: np.ndarray,
                   episode_ends: np.ndarray):
    zarr_path = os.path.join(dst_dir, 'replay_buffer.zarr')
    os.makedirs(zarr_path, exist_ok=True)
    print(f"\n[Write] Creating {zarr_path} …")

    img_comp = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    num_comp = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    root = zarr.open(zarr.DirectoryStore(zarr_path), mode='w')
    data = root.require_group('data')
    meta = root.require_group('meta')

    N = int(episode_ends[-1])
    print(f"  Total frames: {N}, episodes: {len(episode_ends)}")

    # ── wrist RGB ─────────────────────────────────────────────────────
    print("  wrist_rgb …")
    data.array('wrist_rgb', src_root['data']['camera0_rgb'][:N],
               chunks=(1, 224, 224, 3), compressor=img_comp, dtype='u1')

    # ── DIGIT images (both sensors) ───────────────────────────────────
    for src_key, out_key in zip(TACTILE_KEYS, OUT_IMG_KEYS):
        print(f"  {out_key} (← {src_key}) …")
        data.array(out_key, src_root['data'][src_key][:N],
                   chunks=(1, 224, 224, 3), compressor=img_comp, dtype='u1')

    # ── proprioception → robot_tcp_pose (7-dim) ───────────────────────
    print("  robot_tcp_pose …")
    tcp_pose = make_tcp_pose(
        src_root['data']['robot0_eef_pos'][:N],
        src_root['data']['robot0_eef_rot_axis_angle'][:N],
        src_root['data']['robot0_gripper_width'][:N],
    )
    data.array('robot_tcp_pose', tcp_pose, chunks=(N, 7),
               compressor=num_comp, dtype='f4')

    # ── action = absolute EEF state (same as robot_tcp_pose) ─────────
    print("  action …")
    data.array('action', tcp_pose.copy(), chunks=(N, 7),
               compressor=num_comp, dtype='f4')

    # ── DIGIT PCA embeddings (fast-path GRU input) ────────────────────
    for emb, out_key in zip([pca_left, pca_right], OUT_PCA_KEYS):
        print(f"  {out_key} …")
        data.array(out_key, emb[:N], chunks=(N, emb.shape[1]),
                   compressor=num_comp, dtype='f4')

    # ── episode_ends ──────────────────────────────────────────────────
    meta.array('episode_ends', episode_ends.astype(np.int64),
               chunks=(len(episode_ends),), compressor=None)

    print("\n[Write] Done. Summary:")
    for key in ['wrist_rgb', 'digit_img_left', 'digit_img_right',
                'robot_tcp_pose', 'action', 'digit_pca_left', 'digit_pca_right']:
        arr = data[key]
        print(f"  {key}: {arr.shape}  {arr.dtype}")
    print(f"  episode_ends: {meta['episode_ends'][:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/home/user/huihsin/voilab/video/dataset.zarr.zip')
    parser.add_argument('--dst', default='/home/user/huihsin/voilab/video/umi_digit_zarr')
    parser.add_argument('--pca-dim', type=int, default=32,
                        help='PCA components per sensor (32 → 96.2%% variance on current dataset)')
    args = parser.parse_args()

    print(f"Source:  {args.src}")
    print(f"Dest:    {args.dst}")
    print(f"PCA dim: {args.pca_dim} per sensor (total GRU input = {args.pca_dim*2})\n")

    src_root = load_src_zarr(args.src)
    N = src_root['data']['camera0_rgb'].shape[0]
    episode_ends = fix_episode_ends(src_root['meta']['episode_ends'][:], N)
    print(f"Episodes: {len(episode_ends)}, total frames: {N}")
    print(f"episode_ends (fixed): {episode_ends}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pca_l, emb_l = fit_pca_single(src_root, 'robot0_tactile_left',  'digit_pca_left',  args.pca_dim, script_dir)
    pca_r, emb_r = fit_pca_single(src_root, 'robot0_tactile_right', 'digit_pca_right', args.pca_dim, script_dir)

    write_dst_zarr(args.dst, src_root, emb_l, emb_r, episode_ends)

    # Save PCA matrices (needed at inference time for online embedding)
    for pca, name in [(pca_l, 'digit_pca_left'), (pca_r, 'digit_pca_right')]:
        save_path = os.path.join(args.dst, f'{name}.npz')
        np.savez(save_path,
                 components=pca.components_,
                 mean=pca.mean_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        print(f"PCA matrix saved → {save_path}")


if __name__ == '__main__':
    main()
