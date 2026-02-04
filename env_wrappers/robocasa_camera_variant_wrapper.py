from __future__ import annotations

import gym
import numpy as np
from scipy.spatial.transform import Rotation


class RobocasaCameraVariantWrapper(gym.Wrapper):
    """
    Rotate robot0_agentview_left around pivot = env.init_robot_base_pos.pos (world),
    keeping world Z constant.
    """

    def __init__(self, env: gym.Env, yaw_deg: float):
        super().__init__(env)
        self.yaw_deg = float(yaw_deg)
        self.cam_name = "robot0_agentview_left"
        self._sim_id = None
        self._cam_id = None
        self._cam_body_id = None

    @property
    def sim(self):
        return self.env.sim

    def _refresh_cam_ids(self):
        sim_id = id(self.sim)
        if self._sim_id == sim_id:
            return
        self._sim_id = sim_id
        self._cam_id = self.sim.model.camera_name2id(self.cam_name)
        self._cam_body_id = int(self.sim.model.cam_bodyid[self._cam_id])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._apply_rotation()
        return self._refresh_observations(obs)

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        self._apply_rotation()
        return self._refresh_observations(obs)

    def step(self, action):
        return self.env.step(action)

    def _apply_rotation(self):
        self._refresh_cam_ids()

        # pivot from init_robot_base_pos fixture
        ref_fixture = self.env.init_robot_base_pos
        pivot = np.array(ref_fixture.pos, dtype=np.float64)

        # parent body world pose
        pb_pos = self.sim.data.body_xpos[self._cam_body_id].copy()
        pb_rot = Rotation.from_matrix(
            self.sim.data.body_xmat[self._cam_body_id].reshape(3, 3)
        )
        pb_rot_inv = pb_rot.inv()

        # local camera pose
        cam_pos_local = self.sim.model.cam_pos[self._cam_id].copy()
        cam_quat_local = self.sim.model.cam_quat[self._cam_id].copy()
        # MuJoCo stores quaternions as (w, x, y, z). SciPy expects (x, y, z, w).
        cam_quat_xyzw = np.array(
            [
                cam_quat_local[1],
                cam_quat_local[2],
                cam_quat_local[3],
                cam_quat_local[0],
            ],
            dtype=np.float64,
        )
        cam_rot_local = Rotation.from_quat(cam_quat_xyzw)

        # local -> world
        cam_pos_world = pb_pos + pb_rot.apply(cam_pos_local)
        cam_rot_world = pb_rot * cam_rot_local

        # rotate around pivot in world
        rot_z = Rotation.from_euler("z", np.deg2rad(self.yaw_deg))
        rel = cam_pos_world - pivot
        new_cam_pos_world = pivot + rot_z.apply(rel)

        # keep world Z unchanged
        new_cam_pos_world[2] = cam_pos_world[2]

        new_cam_rot_world = rot_z * cam_rot_world

        # world -> local
        new_cam_pos_local = pb_rot_inv.apply(new_cam_pos_world - pb_pos)
        new_cam_rot_local = pb_rot_inv * new_cam_rot_world
        # SciPy returns (x, y, z, w) — convert back to MuJoCo (w, x, y, z)
        new_cam_quat_xyzw = new_cam_rot_local.as_quat()
        new_cam_quat_local = np.array(
            [
                new_cam_quat_xyzw[3],
                new_cam_quat_xyzw[0],
                new_cam_quat_xyzw[1],
                new_cam_quat_xyzw[2],
            ],
            dtype=np.float64,
        )

        # write back in parent-body frame
        self.sim.model.cam_pos[self._cam_id] = new_cam_pos_local
        self.sim.model.cam_quat[self._cam_id] = new_cam_quat_local

        self.sim.forward()

    def _refresh_observations(self, obs):
        if hasattr(self.env, "_update_observables"):
            self.env._update_observables(force=True)
        if hasattr(self.env, "_get_observations"):
            return self.env._get_observations()
        return obs


if __name__ == "__main__":
      from robocasa.utils.env_utils import create_env
      import cv2
      
      base_env = create_env(
            env_name="TurnSinkSpout"
      )
      wrapped_env = RobocasaCameraVariantWrapper(base_env, yaw_deg=-45.0)
      
      _ = wrapped_env.reset()
      
      low, high = wrapped_env.action_spec
      action = np.random.uniform(low=low, high=high)
      obs, reward, done, info = wrapped_env.step(action)

      img, img2 = obs["robot0_agentview_left_image"], obs["robot0_agentview_right_image"]
      img, img2 = img[::-1], img2[::-1]
      cv2.imshow("Rotated Camera View Left", img)
      cv2.imshow("Original Camera View Right", img2) 
      cv2.waitKey(0)
      cv2.destroyAllWindows()

