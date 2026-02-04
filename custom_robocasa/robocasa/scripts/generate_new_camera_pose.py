import numpy as np
from scipy.spatial.transform import Rotation
from robocasa.utils.env_utils import create_env

TASKS = ["TurnSinkSpout", "TurnOnSinkFaucet", "TurnOffSinkFaucet"]
CAM = "robot0_agentview_left"
PARENT_BODY = "mobilebase0_support"

def joint_world_pos(sim, joint_name: str) -> np.ndarray:
    jnt_id = sim.model.joint_name2id(joint_name)
    body_id = sim.model.jnt_bodyid[jnt_id]
    jnt_pos_local = sim.model.jnt_pos[jnt_id]
    body_pos = sim.data.body_xpos[body_id]
    body_rot = sim.data.body_xmat[body_id].reshape(3, 3)
    return body_pos + body_rot @ jnt_pos_local

def compute_pivot(sim, env) -> np.ndarray:
    sink = env.get_fixture(env.FixtureType.SINK)
    prefix = sink.naming_prefix
    water_site = f"{prefix}water"
    handle_joint = f"{prefix}handle_joint"
    spout_joint = f"{prefix}spout_joint"

    pts = [
        sim.data.get_site_xpos(water_site).copy(),
        joint_world_pos(sim, handle_joint),
        joint_world_pos(sim, spout_joint),
    ]
    return np.mean(np.stack(pts, axis=0), axis=0)

for task in TASKS:
    env = create_env(env_name=task, camera_depths=False)
    env.reset()
    sim = env.sim

    cam_id = sim.model.camera_name2id(CAM)

    # parent body world pose
    pb_id = sim.model.body_name2id(PARENT_BODY)
    pb_pos = sim.data.body_xpos[pb_id].copy()
    pb_rot = Rotation.from_matrix(sim.data.body_xmat[pb_id].reshape(3, 3))
    pb_rot_inv = pb_rot.inv()

    # camera local pose (parent frame)
    cam_pos_local = sim.model.cam_pos[cam_id].copy()
    cam_quat_local = sim.model.cam_quat[cam_id].copy()
    cam_rot_local = Rotation.from_quat(cam_quat_local, scalar_first=True)

    # local -> world
    cam_pos_world = pb_pos + pb_rot.apply(cam_pos_local)
    cam_rot_world = pb_rot * cam_rot_local

    pivot = compute_pivot(sim, env)

    # rotate 90 deg left about pivot in world
    rot_z = Rotation.from_euler("z", 90, degrees=True)
    rel = cam_pos_world - pivot
    new_cam_pos_world = pivot + rot_z.apply(rel)

    # keep world z unchanged
    new_cam_pos_world[2] = cam_pos_world[2]

    new_cam_rot_world = rot_z * cam_rot_world

    # world -> local
    new_cam_pos_local = pb_rot_inv.apply(new_cam_pos_world - pb_pos)
    new_cam_rot_local = pb_rot_inv * new_cam_rot_world
    new_cam_quat_local = new_cam_rot_local.as_quat(scalar_first=True)

    print(f"\n=== {task} ===")
    print("pivot_world:", np.round(pivot, 4))
    print("new_pos_local:", np.round(new_cam_pos_local, 6))
    print("new_quat_local:", np.round(new_cam_quat_local, 6))

    env.close()
