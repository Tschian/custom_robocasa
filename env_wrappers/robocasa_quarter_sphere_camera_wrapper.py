from __future__ import annotations

import gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING


class RobocasaQuarterSphereCameraWrapper(gym.Wrapper):
    """
    Adds extra rendered views by sampling camera poses on left / right quarter-sphere
    surfaces around a reference fixture. Cameras are rendered by temporarily moving
    existing base cameras (left / right) and then restoring them.
    """

    def __init__(
        self,
        env: gym.Env,
        num_cameras: int = 9,
        left_cam_name: str = "robot0_agentview_left",
        right_cam_name: str = "robot0_agentview_right",
        left_prefix: str = "robot0_agentview_left",
        right_prefix: str = "robot0_agentview_right",
        include_depth: bool = True,
        seed: int | None = None,
        use_quarter: bool = True,
        camera_width: int | None = None,
        camera_height: int | None = None,
        left_az_range: tuple[float, float] | None = None,
        left_el_range: tuple[float, float] | None = None,
        right_az_range: tuple[float, float] | None = None,
        right_el_range: tuple[float, float] | None = None,
    ):
        super().__init__(env)
        self.num_cameras = int(num_cameras)
        self.left_cam_name = left_cam_name
        self.right_cam_name = right_cam_name
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.include_depth = bool(include_depth)
        self.use_quarter = bool(use_quarter)
        self.camera_width = camera_width
        self.camera_height = camera_height

        self._rng = np.random.default_rng(seed)
        self._sim_id = None
        self._cam_cache = {}

        # whether pre-give azimuth/elevation ranges or use defaults
        self._left_poses = None
        self._right_poses = None

        # cached reference axes for sampling camera poses
        self.ref_forward = None
        self.ref_right = None
        self.ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._virtual_pose_map = {}
        self._virtual_valid = {}

        self._set_azimuth_elevation_ranges(
            left_az_range=left_az_range,
            left_el_range=left_el_range,
            right_az_range=right_az_range,
            right_el_range=right_el_range,
        )

        self._left_names = [f"{self.left_prefix}_{i:02d}" for i in range(1, self.num_cameras + 1)]
        self._right_names = [f"{self.right_prefix}_{i:02d}" for i in range(1, self.num_cameras + 1)]

    @property
    def sim(self):
        return self.env.sim

    def _refresh_cam_ids(self):
        sim_id = id(self.sim)
        if self._sim_id == sim_id:
            return
        self._sim_id = sim_id
        self._cam_cache = {}
        for name in (self.left_cam_name, self.right_cam_name):
            cam_id = self.sim.model.camera_name2id(name)
            cam_body_id = int(self.sim.model.cam_bodyid[cam_id])
            self._cam_cache[name] = (cam_id, cam_body_id)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._left_poses = None
        self._right_poses = None
        self._sample_camera_poses()
        return self._add_extra_camera_obs(obs)

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        if "model" in state:
            # Full episode reset: new kitchen layout loaded, resample virtual camera poses
            self._sample_camera_poses()
        elif self._left_poses is None or self._right_poses is None:
            # First call without a prior reset() (e.g. standalone use)
            self._sample_camera_poses()
        return self._add_extra_camera_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._add_extra_camera_obs(obs)
        return obs, reward, done, info

    def _resolve_render_size(self, cam_name: str):
        if self.camera_width is not None and self.camera_height is not None:
            return self.camera_width, self.camera_height
        if hasattr(self.env, "camera_names"):
            for idx, name in enumerate(self.env.camera_names):
                if name == cam_name:
                    return self.env.camera_widths[idx], self.env.camera_heights[idx]
            if len(self.env.camera_widths) > 0:
                return self.env.camera_widths[0], self.env.camera_heights[0]
        return 256, 256
    
    def _set_azimuth_elevation_ranges(
        self,
        left_az_range: tuple[float, float] | None = None,
        left_el_range: tuple[float, float] | None = None,
        right_az_range: tuple[float, float] | None = None,
        right_el_range: tuple[float, float] | None = None,
    ):
        self.left_az_range = left_az_range if left_az_range is not None else (0, 90)
        self.left_el_range = left_el_range if left_el_range is not None else (0, 90)
        self.right_az_range = right_az_range if right_az_range is not None else (0, -90)
        self.right_el_range = right_el_range if right_el_range is not None else (0, 90)

    def _add_extra_camera_obs(self, obs):
        if self.num_cameras <= 0:
            return obs
        if self._left_poses is None or self._right_poses is None:
            self._sample_camera_poses()

        self._refresh_cam_ids()

        left_cam_id, left_body_id = self._cam_cache[self.left_cam_name]
        right_cam_id, right_body_id = self._cam_cache[self.right_cam_name]

        left_pos_local = self.sim.model.cam_pos[left_cam_id].copy()
        left_quat_local = self.sim.model.cam_quat[left_cam_id].copy()
        right_pos_local = self.sim.model.cam_pos[right_cam_id].copy()
        right_quat_local = self.sim.model.cam_quat[right_cam_id].copy()

        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        if self.num_cameras > 0:
            width, height = self._resolve_render_size(self.left_cam_name)
            base_left_img = obs.get(f"{self.left_cam_name}_image")
            base_left_depth = obs.get(f"{self.left_cam_name}_depth")
            for name, (pos_w, rot_w) in zip(self._left_names, self._left_poses):
                if not self._virtual_valid.get(name, True):
                    if base_left_img is not None:
                        obs[f"{name}_image"] = base_left_img
                    if self.include_depth and base_left_depth is not None:
                        obs[f"{name}_depth"] = base_left_depth
                    obs[f"{name}_valid"] = np.array([0], dtype=np.uint8)
                    continue

                self._set_camera_world_pose(left_cam_id, left_body_id, pos_w, rot_w)
                self.sim.forward()
                img = self.sim.render(
                    camera_name=self.left_cam_name,
                    width=width,
                    height=height,
                    depth=self.include_depth,
                )
                if self.include_depth:
                    rgb, depth = img
                    obs[f"{name}_image"] = rgb[::convention]
                    obs[f"{name}_depth"] = np.expand_dims(depth[::convention], axis=-1)
                else:
                    obs[f"{name}_image"] = img[::convention]
                obs[f"{name}_valid"] = np.array([1], dtype=np.uint8)

        if self.num_cameras > 0:
            width, height = self._resolve_render_size(self.right_cam_name)
            base_right_img = obs.get(f"{self.right_cam_name}_image")
            base_right_depth = obs.get(f"{self.right_cam_name}_depth")
            for name, (pos_w, rot_w) in zip(self._right_names, self._right_poses):
                if not self._virtual_valid.get(name, True):
                    if base_right_img is not None:
                        obs[f"{name}_image"] = base_right_img
                    if self.include_depth and base_right_depth is not None:
                        obs[f"{name}_depth"] = base_right_depth
                    obs[f"{name}_valid"] = np.array([0], dtype=np.uint8)
                    continue

                self._set_camera_world_pose(right_cam_id, right_body_id, pos_w, rot_w)
                self.sim.forward()
                img = self.sim.render(
                    camera_name=self.right_cam_name,
                    width=width,
                    height=height,
                    depth=self.include_depth,
                )
                if self.include_depth:
                    rgb, depth = img
                    obs[f"{name}_image"] = rgb[::convention]
                    obs[f"{name}_depth"] = np.expand_dims(depth[::convention], axis=-1)
                else:
                    obs[f"{name}_image"] = img[::convention]
                obs[f"{name}_valid"] = np.array([1], dtype=np.uint8)

        self.sim.model.cam_pos[left_cam_id] = left_pos_local
        self.sim.model.cam_quat[left_cam_id] = left_quat_local
        self.sim.model.cam_pos[right_cam_id] = right_pos_local
        self.sim.model.cam_quat[right_cam_id] = right_quat_local
        self.sim.forward()

        return obs

    def _sample_camera_poses(self):
        self._refresh_cam_ids()

        ref_fixture = self.env.init_robot_base_pos
        pivot = np.array(ref_fixture.pos, dtype=np.float32)
        print(f"Sampling camera poses around current pivot point: {pivot}")

        left_pos_w, _ = self._get_camera_world_pose(self.left_cam_name)
        right_pos_w, _ = self._get_camera_world_pose(self.right_cam_name)

        v_left = left_pos_w - pivot
        v_right = right_pos_w - pivot

        left_radius = np.linalg.norm(v_left)
        right_radius = np.linalg.norm(v_right)
        if left_radius <= 0 or right_radius <= 0:
            raise ValueError("Camera radius is zero; check reference fixture and camera positions.")

        robot_pos = self.env.robots[0].base_pos
        if robot_pos is None:
            robot_pos = self.sim.data.get_body_xpos(self.env.robots[0].robot_model.root_body)
        robot_pos = np.array(robot_pos, dtype=np.float32)

        forward = robot_pos - pivot
        forward[2] = 0.0
        if np.linalg.norm(forward) < 1e-6:
            forward = v_left.copy()
            forward[2] = 0.0
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.ref_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = right / np.linalg.norm(right)

        self.ref_forward = forward
        self.ref_right = right

        left_dirs = self._sample_unit_vectors_homogeneous(
            self.num_cameras,
            az_range=self.left_az_range,
            el_range=self.left_el_range,
        )
        right_dirs = self._sample_unit_vectors_homogeneous(
            self.num_cameras,
            az_range=self.right_az_range,
            el_range=self.right_el_range,
        )

        self._left_poses = []
        self._right_poses = []
        for direction in left_dirs:
            pos_w = pivot + left_radius * direction
            if self._is_occluded(pos_w, pivot):
                self._left_poses.append((pos_w, None))
                continue
            rot_w = self._look_at_rotation(pos_w, pivot)
            self._left_poses.append((pos_w, rot_w))

        for direction in right_dirs:
            pos_w = pivot + right_radius * direction
            if self._is_occluded(pos_w, pivot):
                self._right_poses.append((pos_w, None))
                continue
            rot_w = self._look_at_rotation(pos_w, pivot)
            self._right_poses.append((pos_w, rot_w))

        self._virtual_pose_map = {
            **{name: pose for name, pose in zip(self._left_names, self._left_poses)},
            **{name: pose for name, pose in zip(self._right_names, self._right_poses)},
        }
        self._virtual_valid = {name: (rot_w is not None) for name, (_, rot_w) in self._virtual_pose_map.items()}

    def _sample_unit_vectors_random(
        self,
        count: int,
        az_range: tuple[float, float] | None = None,
        el_range: tuple[float, float] | None = None,
    ):
        """
        Samples directions using azimuth / elevation around the (forward, right, up) frame.
        Azimuth is measured in the forward-right plane; elevation is measured above that plane.
        """
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        if self.ref_forward is None or self.ref_right is None:
            raise RuntimeError("Reference axes not set; call _sample_camera_poses() first.")

        az_min, az_max = az_range
        el_min, el_max = el_range

        if not self.use_quarter:
            el_min = min(el_min, -90.0)
            el_max = max(el_max, 90.0)

        az = np.deg2rad(self._rng.uniform(az_min, az_max, size=count))
        el = np.deg2rad(self._rng.uniform(el_min, el_max, size=count))

        cos_el = np.cos(el)
        sin_el = np.sin(el)
        cos_az = np.cos(az)
        sin_az = np.sin(az)

        dirs = (
            cos_el[:, None] * (cos_az[:, None] * self.ref_forward + sin_az[:, None] * self.ref_right)
            + sin_el[:, None] * self.ref_up
        )
        return dirs

    def _sample_unit_vectors_homogeneous(
        self,
        count: int,
        az_range: tuple[float, float],
        el_range: tuple[float, float],
    ):
        """
        Homogeneous (approximately equal-area) sampling over a spherical patch defined by azimuth/elevation ranges.
        Uses a stratified grid in azimuth and sin(elevation) to make area per sample roughly uniform.
        """
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        if self.ref_forward is None or self.ref_right is None:
            raise RuntimeError("Reference axes not set; call _sample_camera_poses() first.")

        az_min, az_max = az_range
        el_min, el_max = el_range

        # choose a grid that covers at least 'count' samples
        n_el = max(1, int(round(np.sqrt(count))))
        n_az = int(np.ceil(count / n_el))

        # cell centers in [0,1]
        u = (np.arange(n_az) + 0.5) / n_az
        v = (np.arange(n_el) + 0.5) / n_el
        uu, vv = np.meshgrid(u, v, indexing="xy")
        uu = uu.flatten()
        vv = vv.flatten()

        # map to azimuth range
        az = np.deg2rad(az_min + (az_max - az_min) * uu)

        # map to elevation range with equal-area in elevation
        sin_el_min = np.sin(np.deg2rad(el_min))
        sin_el_max = np.sin(np.deg2rad(el_max))
        sin_el = sin_el_min + (sin_el_max - sin_el_min) * vv
        el = np.arcsin(sin_el)

        cos_el = np.cos(el)
        sin_el = np.sin(el)
        cos_az = np.cos(az)
        sin_az = np.sin(az)

        dirs = (
            cos_el[:, None] * (cos_az[:, None] * self.ref_forward + sin_az[:, None] * self.ref_right)
            + sin_el[:, None] * self.ref_up
        )

        return dirs[:count]

    def _look_at_rotation(self, pos_w: np.ndarray, target_w: np.ndarray):
        forward = target_w - pos_w
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            raise ValueError("Camera position coincides with target; cannot compute look-at rotation.")
        forward = forward / norm
        z_axis = -forward
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack((x_axis, y_axis, z_axis))
        return Rotation.from_matrix(rot)

    def _get_camera_world_pose(self, cam_name: str):
        self._refresh_cam_ids()
        cam_id, cam_body_id = self._cam_cache[cam_name]
        pb_pos = self.sim.data.body_xpos[cam_body_id].copy()
        pb_rot = Rotation.from_matrix(self.sim.data.body_xmat[cam_body_id].reshape(3, 3))

        cam_pos_local = self.sim.model.cam_pos[cam_id].copy()
        cam_quat_local = self.sim.model.cam_quat[cam_id].copy()
        cam_quat_xyzw = np.array(
            [cam_quat_local[1], cam_quat_local[2], cam_quat_local[3], cam_quat_local[0]],
            dtype=np.float64,
        )
        cam_rot_local = Rotation.from_quat(cam_quat_xyzw)

        cam_pos_world = pb_pos + pb_rot.apply(cam_pos_local)
        cam_rot_world = pb_rot * cam_rot_local
        return cam_pos_world, cam_rot_world

    def _set_camera_world_pose(self, cam_id: int, cam_body_id: int, pos_w: np.ndarray, rot_w: Rotation):
        pb_pos = self.sim.data.body_xpos[cam_body_id].copy()
        pb_rot = Rotation.from_matrix(self.sim.data.body_xmat[cam_body_id].reshape(3, 3))
        pb_rot_inv = pb_rot.inv()

        cam_pos_local = pb_rot_inv.apply(pos_w - pb_pos)
        cam_rot_local = pb_rot_inv * rot_w
        cam_quat_xyzw = cam_rot_local.as_quat()
        cam_quat_local = np.array(
            [cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]],
            dtype=np.float64,
        )

        self.sim.model.cam_pos[cam_id] = cam_pos_local
        self.sim.model.cam_quat[cam_id] = cam_quat_local

    def get_virtual_camera_names(self):
        return list(self._left_names) + list(self._right_names)

    def get_virtual_camera_pose(self, cam_name: str):
        if cam_name not in self._virtual_pose_map:
            raise KeyError(f"Unknown virtual camera name: {cam_name}")
        return self._virtual_pose_map[cam_name]

    def virtual_intrinsic_fn(self, cam_name: str, width: int, height: int):
        """
        Intrinsics for virtual cameras. Uses base camera fovy (left/right) to compute K.
        """
        self._refresh_cam_ids()
        base_cam = self.left_cam_name if cam_name.startswith(self.left_prefix) else self.right_cam_name
        cam_id = self.sim.model.camera_name2id(base_cam)
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def virtual_extrinsic_fn(self, cam_name: str):
        """
        Extrinsics for virtual cameras. Returns camera->world pose with robosuite axis correction.
        """
        pos_w, rot_w = self.get_virtual_camera_pose(cam_name)
        if rot_w is None:
            base_cam = self.left_cam_name if cam_name.startswith(self.left_prefix) else self.right_cam_name
            cam_id = self.sim.model.camera_name2id(base_cam)
            camera_pos = self.sim.data.cam_xpos[cam_id]
            camera_rot = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = camera_rot
            T[:3, 3] = camera_pos
            corr = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            return T @ corr
        R = rot_w.as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos_w
        corr = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return T @ corr

    def _is_occluded(self, cam_pos: np.ndarray, target_pos: np.ndarray, eps: float = 1e-3) -> bool:
        ray_dir = target_pos - cam_pos
        dist = np.linalg.norm(ray_dir)
        if dist < 1e-6:
            return False
        ray_dir = ray_dir / dist
        pnt = np.asarray(cam_pos, dtype=np.float64).reshape(3, 1)
        vec = np.asarray(ray_dir, dtype=np.float64).reshape(3, 1)
        geomgroup = np.zeros((6, 1), dtype=np.uint8)
        geomid = np.array([[-1]], dtype=np.int32)
        hit_dist = mujoco.mj_ray(
            self.sim.model._model,
            self.sim.data._data,
            pnt,
            vec,
            geomgroup,
            1,
            -1,
            geomid,
        )
        if geomid[0, 0] < 0:
            return False
        return float(hit_dist) < dist - eps

    def _check_success(self):
        return self.env._check_success()


if __name__ == "__main__":
    from robocasa.utils.env_utils import create_env
    import cv2

    base_env = create_env(env_name="TurnSinkSpout")
    wrapped_env = RobocasaQuarterSphereCameraWrapper(
        base_env,
        num_cameras=8,
        include_depth=False,
        seed=0,
    )

    _ = wrapped_env.reset()

    low, high = wrapped_env.action_spec
    action = np.random.uniform(low=low, high=high)
    obs, reward, done, info = wrapped_env.step(action)

    images = {"robot0_agentview_left_image": obs["robot0_agentview_left_image"], 
              "robot0_agentview_right_image": obs["robot0_agentview_right_image"]}

    for i in range(1, wrapped_env.num_cameras + 1):
        left_name = f"{wrapped_env.left_prefix}_{i:02d}"
        right_name = f"{wrapped_env.right_prefix}_{i:02d}"
        images[f"{left_name}"] = obs[f"{left_name}_image"]
        images[f"{right_name}"] = obs[f"{right_name}_image"]

        if f"{left_name}_valid" in obs and obs[f"{left_name}_valid"][0] == 1:
            print(f"Virtual camera {left_name} is valid.")
        if f"{right_name}_valid" in obs and obs[f"{right_name}_valid"][0] == 1:
            print(f"Virtual camera {right_name} is valid.")


    # for key, img in images.items():
    #     img = img[::-1]
    #     cv2.imshow(f"{key}", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
