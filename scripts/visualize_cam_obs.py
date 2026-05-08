"""
Visualize multi-view camera images + pose for 2 episodes.

Layout per PNG (one file per episode):
  [obs_key label] | t=0 | t=1 | t=2 | t=3 | t=4 | [pos / fwd / euler rotation]

Camera position and forward direction are recovered from the stored
Plücker ray maps:  ray = [d_base, m_base]  where  m_base = C_base × d_base.
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

# ── config ──────────────────────────────────────────────────────────────────
TASK     = "CloseDrawer"
PATH     = ("/home/qian-wang/depth_vla/robocasa/datasets/v0.1/single_stage"
            "/kitchen_drawer/CloseDrawer/2024-04-30/small_multiview_mobilebase.hdf5")
EPISODES = ["demo_23", "demo_28"]
N_FRAMES = 5
OUT_DIR  = "/home/qian-wang/custom_robocasa"

# ── layout constants (inches) ────────────────────────────────────────────────
LABEL_W = 3.2
IMG_W   = 1.45
INFO_W  = 5.2
ROW_H   = 1.55
# ────────────────────────────────────────────────────────────────────────────


def plucker_cam_pos(ray_frame: np.ndarray, stride: int = 8) -> np.ndarray:
    """Recover camera origin C from a Plücker ray map (H×W×6).

    Each pixel stores (d, m) where d is the unit ray direction and m = C×d.
    Summing the per-pixel identity: (I – d dᵀ)C = d×m over all pixels gives
    a well-conditioned 3×3 linear system for C.
    """
    d = ray_frame[::stride, ::stride, :3].reshape(-1, 3).astype(np.float64)
    m = ray_frame[::stride, ::stride, 3:].reshape(-1, 3).astype(np.float64)
    norms = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
    d /= norms                                    # re-normalise (float16 rounding)
    n  = len(d)
    A  = n * np.eye(3) - d.T @ d                  # (3,3)
    b  = np.cross(d, m).sum(axis=0)               # (3,)
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C                                       # camera position in mobilebase frame


def plucker_rotation(ray_frame: np.ndarray):
    """Return (fwd_vec, euler_ZYX_deg) from the centre-pixel ray direction.

    fwd_vec  : (3,) unit forward direction in mobilebase frame
    euler_ZYX: (yaw, pitch, roll) in degrees, built via the same look-at
               convention used by RobocasaLocalSphereCameraWrapper.
    """
    H, W = ray_frame.shape[:2]
    fwd = ray_frame[H // 2, W // 2, :3].astype(np.float64)
    fwd /= np.linalg.norm(fwd) + 1e-8

    z_ax = -fwd
    up   = np.array([0.0, 0.0, 1.0])
    x_ax = np.cross(up, z_ax)
    if np.linalg.norm(x_ax) < 1e-6:
        up   = np.array([0.0, 1.0, 0.0])
        x_ax = np.cross(up, z_ax)
    x_ax /= np.linalg.norm(x_ax)
    y_ax  = np.cross(z_ax, x_ax)
    R     = np.column_stack([x_ax, y_ax, z_ax])   # camera-to-mobilebase rotation
    euler = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return fwd, euler                              # (yaw, pitch, roll)


def sort_image_keys(keys):
    priority = {
        "robot0_agentview_left_image":     (0, 0),
        "robot0_agentview_left_01_image":  (0, 1),
        "robot0_agentview_left_02_image":  (0, 2),
        "robot0_agentview_left_03_image":  (0, 3),
        "robot0_agentview_right_image":    (1, 0),
        "robot0_agentview_right_01_image": (1, 1),
        "robot0_agentview_right_02_image": (1, 2),
        "robot0_agentview_right_03_image": (1, 3),
        "robot0_eye_in_hand_image":        (2, 0),
    }
    return sorted(keys, key=lambda k: priority.get(k, (9, 0)))


def make_episode_figure(hdf5, task, ep, image_keys, n_frames):
    n_rows  = len(image_keys)
    fig_w   = LABEL_W + IMG_W * n_frames + INFO_W
    fig_h   = ROW_H * n_rows + 0.65

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(f"Task: {task}   |   Episode: {ep}",
                 fontsize=11, fontweight="bold", y=0.998)

    gs = gridspec.GridSpec(
        n_rows, n_frames + 2,
        width_ratios=[LABEL_W] + [IMG_W] * n_frames + [INFO_W],
        left=0.005, right=0.995, top=0.965, bottom=0.005,
        hspace=0.07, wspace=0.025,
    )

    obs_grp = f"data/{ep}/obs"

    for row, img_key in enumerate(image_keys):
        cam_name  = img_key[:-6]          # strip "_image"
        ray_key   = f"{cam_name}_ray"
        valid_key = f"{cam_name}_valid"

        # ── label cell ──────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_facecolor("#dde5f8")
        ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
        ax_lbl.axis("off")
        # break long names at "_" boundaries so they wrap in the column
        label_text = cam_name.replace("robot0_", "").replace("agentview_", "agentview\n")
        ax_lbl.text(0.5, 0.5, label_text,
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    linespacing=1.35, transform=ax_lbl.transAxes)

        # ── 5 frame images ──────────────────────────────────────────────────
        imgs = hdf5[f"{obs_grp}/{img_key}"][()]
        for t in range(n_frames):
            ax = fig.add_subplot(gs[row, t + 1])
            ax.imshow(imgs[t])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if row == 0:
                ax.set_title(f"t = {t}", fontsize=7.5, pad=2)

        # ── camera info cell ────────────────────────────────────────────────
        ax_info = fig.add_subplot(gs[row, n_frames + 1])
        ax_info.set_facecolor("#f5ede0")
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
        ax_info.axis("off")

        # check validity flag (only virtual cameras have one)
        is_valid = True
        if f"{obs_grp}/{valid_key}" in hdf5:
            is_valid = bool(int(hdf5[f"{obs_grp}/{valid_key}"][0].item()))

        if f"{obs_grp}/{ray_key}" in hdf5 and is_valid:
            ray0 = hdf5[f"{obs_grp}/{ray_key}"][0].astype(np.float32)
            C    = plucker_cam_pos(ray0)
            fwd, euler = plucker_rotation(ray0)
            info = (
                f"pos   ({C[0]:+.3f},  {C[1]:+.3f},  {C[2]:+.3f})\n"
                f"fwd   ({fwd[0]:+.3f},  {fwd[1]:+.3f},  {fwd[2]:+.3f})\n"
                f"yaw = {euler[0]:+.1f}°   pitch = {euler[1]:+.1f}°   roll = {euler[2]:+.1f}°"
            )
            color = "#1a1a2e"
        elif not is_valid:
            info  = "[ occluded / invalid ]"
            color = "#aa2222"
        else:
            info  = "no ray data"
            color = "#666666"

        ax_info.text(0.04, 0.5, info, ha="left", va="center",
                     fontsize=6.8, family="monospace", color=color,
                     linespacing=1.85, transform=ax_info.transAxes)

    return fig


def main():
    f = h5py.File(PATH, "r")
    all_obs = list(f[f"data/{EPISODES[0]}/obs"].keys())
    image_keys = sort_image_keys([k for k in all_obs if k.endswith("_image")])

    print(f"Cameras to visualise ({len(image_keys)}):")
    for k in image_keys:
        print(f"  {k}")
    print()

    for ep in EPISODES:
        fig = make_episode_figure(f, TASK, ep, image_keys, N_FRAMES)
        out = os.path.join(OUT_DIR, f"camera_vis_{TASK}_{ep}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved → {out}")

    f.close()
    print("Done.")


if __name__ == "__main__":
    main()
