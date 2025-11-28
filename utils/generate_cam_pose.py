import numpy as np
from robosuite.utils.transform_utils import quat2mat
import cv2
import mujoco
from robocasa.utils.camera_utils import CAM_CONFIGS
from robocasa.utils.env_utils import create_env
from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)

# ---------- helpers ----------
def pose_compose(T_wb, T_bc):
    """Compose world←base with base←cam to get world←cam."""
    R_wb, t_wb = T_wb[:3,:3], T_wb[:3,3]
    R_bc, t_bc = T_bc[:3,:3], T_bc[:3,3]
    R_wc = R_wb @ R_bc
    t_wc = t_wb + R_wb @ t_bc
    T_wc = np.eye(4); T_wc[:3,:3]=R_wc; T_wc[:3,3]=t_wc
    return T_wc

def T_from_pos_quat(pos, quat):
    """MuJoCo gives quats as [w,x,y,z]."""
    R = quat2mat(quat)         # world←frame rotation
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=np.array(pos)
    return T

def camera_forward_world(R_wc):
    """Principal ray direction for a MuJoCo camera (looks along -Z)."""
    return (R_wc @ np.array([0.0, 0.0, -1.0])).astype(float)

def closest_points_between_rays(p1, d1, p2, d2, eps=1e-8):
    """
    Ray1: p1 + s d1, Ray2: p2 + t d2. Returns midpoint of closest approach, and (s,t).
    d1,d2 should be unit.
    """
    d1 = d1 / (np.linalg.norm(d1) + eps)
    d2 = d2 / (np.linalg.norm(d2) + eps)
    w0 = p1 - p2
    a = np.dot(d1, d1)         # = 1
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)         # = 1
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a*c - b*b
    if abs(denom) < 1e-6:
        # Nearly parallel — project p2 onto ray1
        s = -d
        t = 0.0
    else:
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
    p_on_1 = p1 + s * d1
    p_on_2 = p2 + t * d2
    mid = 0.5 * (p_on_1 + p_on_2)
    return mid, (s, t), np.linalg.norm(p_on_1 - p_on_2)

# ---------- main utility ----------
def virtual_workspace_center_from_two_cams(
    env,
    base_body_name="mobilebase0_support",
    left_cam_rel_pos=None, left_cam_rel_quat=None,
    right_cam_rel_pos=None, right_cam_rel_quat=None,
    clamp_z=None,   # e.g., 0.75–0.85 for countertop. Set to a float or None.
):
    """
    left/right cam poses are RELATIVE to base (as in CAM_CONFIGS parent_body=mobilebase0_support).
    Returns: world-space virtual center (3,)
    """
    sim = env.sim

    # World pose of the base
    bid = sim.model.body_name2id(base_body_name)
    p_wb = sim.data.xpos[bid].copy()       # (3,)
    q_wb = sim.data.xquat[bid].copy()      # [w,x,y,z]
    T_wb = T_from_pos_quat(p_wb, q_wb)

    # Build base←camera transforms from rel pos/quat
    T_bl = T_from_pos_quat(np.array(left_cam_rel_pos),  np.array(left_cam_rel_quat))
    T_br = T_from_pos_quat(np.array(right_cam_rel_pos), np.array(right_cam_rel_quat))

    # Compose to world
    T_wl = pose_compose(T_wb, T_bl)
    T_wr = pose_compose(T_wb, T_br)

    # Camera origins and directions (world)
    pL = T_wl[:3,3]; RL = T_wl[:3,:3]; dL = camera_forward_world(RL)
    pR = T_wr[:3,3]; RR = T_wr[:3,:3]; dR = camera_forward_world(RR)

    center_w, (sL, sR), gap = closest_points_between_rays(pL, dL, pR, dR)

    # Optional: clamp Z to a plausible surface height
    if clamp_z is not None:
        center_w[2] = clamp_z

    return center_w, dict(ray_gap=gap, s_left=sL, s_right=sR)


if __name__ == "__main__":
    # Suppose your two cams are the default left/right agent views (relative to base)
    l = CAM_CONFIGS["DEFAULT"]["robot0_agentview_left"]
    r = CAM_CONFIGS["DEFAULT"]["robot0_agentview_right"]

    c = CAM_CONFIGS["DEFAULT"]["robot0_agentview_center"]  # just for reference

    camera_names = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
        "robot0_agentview_center"
    ],

    env_name = np.random.choice(
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_env(env_name=env_name, camera_names=camera_names)
    env.reset()
    center_w, debug = virtual_workspace_center_from_two_cams(
        env,
        left_cam_rel_pos=l["pos"], left_cam_rel_quat=l["quat"],
        right_cam_rel_pos=r["pos"], right_cam_rel_quat=r["quat"],
        clamp_z=None  # or 0.80 if you want to fix height
    )
    print("Virtual center (world):", center_w, " Debug:", debug)