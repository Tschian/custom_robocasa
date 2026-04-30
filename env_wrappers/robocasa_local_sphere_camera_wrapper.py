from __future__ import annotations

import gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING


class RobocasaLocalSphereCameraWrapper(gym.Wrapper):
    """
    Adds rendered views by randomly sampling camera poses in small spherical caps
    centered on the default left/right camera positions.

    At each episode reset, num_cameras positions are drawn uniformly from
        azimuth  in [center_az - az_half, center_az + az_half]
        elevation in [center_el - el_half, center_el + el_half]
    where center_az / center_el are the azimuth and elevation of the actual default
    left/right cameras relative to the reference fixture (pivot) and the pivot->robot
    forward axis. The sphere radius equals the distance from the pivot to each default
    camera, matching the original camera placement distance.

    Key differences from RobocasaQuarterSphereCameraWrapper:
    - Sampling region is a small cap around the default camera rather than the full
      quarter-sphere, controlled by az_half and el_half in degrees.
    - Sampling is uniformly random (not stratified/homogeneous) and re-drawn at every
      episode reset, giving per-episode camera diversity within the cap.
    - No pre-specified az/el ranges needed: centers are computed from actual camera poses.
    """

    def __init__(
        self,
        env: gym.Env,
        num_cameras: int = 5,
        left_cam_name: str = "robot0_agentview_left",
        right_cam_name: str = "robot0_agentview_right",
        left_prefix: str = "robot0_agentview_left",
        right_prefix: str = "robot0_agentview_right",
        include_depth: bool = True,
        seed: int | None = None,
        camera_width: int | None = None,
        camera_height: int | None = None,
        az_half: float = 15.0,
        el_half: float = 10.0,
        radius_range: tuple[float, float] = (1.0, 1.2),
    ):
        """
        Parameters
        ----------
        env             : wrapped Gym environment
        num_cameras     : number of virtual cameras per side (left and right)
        left_cam_name   : MuJoCo camera used to render left virtual views
        right_cam_name  : MuJoCo camera used to render right virtual views
        left_prefix     : prefix for left virtual camera observation keys
        right_prefix    : prefix for right virtual camera observation keys
        include_depth   : whether to also render depth images
        seed            : RNG seed (fixed seed gives same per-episode sequence)
        camera_width    : override render width (default: inherit from env)
        camera_height   : override render height (default: inherit from env)
        az_half         : half-width of azimuth sampling range in degrees
        el_half         : half-width of elevation sampling range in degrees
        radius_range    : (min_scale, max_scale) multiplied onto the base radius per camera
        """
        super().__init__(env)
        self.num_cameras = int(num_cameras)
        self.left_cam_name = left_cam_name
        self.right_cam_name = right_cam_name
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.include_depth = bool(include_depth)
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.az_half = float(az_half)
        self.el_half = float(el_half)
        self.radius_range = (float(radius_range[0]), float(radius_range[1]))

        self._rng = np.random.default_rng(seed)
        self._sim_id: int | None = None
        self._cam_cache: dict = {}

        self._left_poses: list | None = None
        self._right_poses: list | None = None

        self.ref_forward: np.ndarray | None = None
        self.ref_right: np.ndarray | None = None
        self.ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.reference_pivot: np.ndarray | None = None

        # Computed from default camera positions each episode
        self._left_az_center: float | None = None
        self._left_el_center: float | None = None
        self._right_az_center: float | None = None
        self._right_el_center: float | None = None

        self._virtual_pose_map: dict = {}
        self._virtual_valid: dict = {}

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

    def _resolve_render_size(self, cam_name: str) -> tuple[int, int]:
        if self.camera_width is not None and self.camera_height is not None:
            return self.camera_width, self.camera_height
        if hasattr(self.env, "camera_names"):
            for idx, name in enumerate(self.env.camera_names):
                if name == cam_name:
                    return self.env.camera_widths[idx], self.env.camera_heights[idx]
            if len(self.env.camera_widths) > 0:
                return self.env.camera_widths[0], self.env.camera_heights[0]
        return 256, 256

    def _add_extra_camera_obs(self, obs: dict) -> dict:
        if self.num_cameras <= 0:
            return obs
        if self._left_poses is None or self._right_poses is None:
            self._sample_camera_poses()

        self._refresh_cam_ids()

        left_cam_id, left_body_id = self._cam_cache[self.left_cam_name]
        right_cam_id, right_body_id = self._cam_cache[self.right_cam_name]

        # Save base camera local poses for restoration after rendering
        left_pos_local = self.sim.model.cam_pos[left_cam_id].copy()
        left_quat_local = self.sim.model.cam_quat[left_cam_id].copy()
        right_pos_local = self.sim.model.cam_pos[right_cam_id].copy()
        right_quat_local = self.sim.model.cam_quat[right_cam_id].copy()

        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Render left virtual cameras
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
                # obs[f"{name}_image"] = rgb[::convention]
                # obs[f"{name}_depth"] = np.expand_dims(depth[::convention], axis=-1)
                obs[f"{name}_image"] = np.flip(rgb, axis=0)
                obs[f"{name}_depth"] = np.expand_dims(np.flip(depth, axis=0), axis=-1)
            else:
                # obs[f"{name}_image"] = img[::convention]
                obs[f"{name}_image"] = np.flip(img, axis=0)
            obs[f"{name}_valid"] = np.array([1], dtype=np.uint8)

        # Render right virtual cameras
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
                # obs[f"{name}_image"] = rgb[::convention]
                # obs[f"{name}_depth"] = np.expand_dims(depth[::convention], axis=-1)
                obs[f"{name}_image"] = np.flip(rgb, axis=0)
                obs[f"{name}_depth"] = np.expand_dims(np.flip(depth, axis=0), axis=-1)
            else:
                # obs[f"{name}_image"] = img[::convention]
                obs[f"{name}_image"] = np.flip(img, axis=0)
            obs[f"{name}_valid"] = np.array([1], dtype=np.uint8)

        # Restore base cameras
        self.sim.model.cam_pos[left_cam_id] = left_pos_local
        self.sim.model.cam_quat[left_cam_id] = left_quat_local
        self.sim.model.cam_pos[right_cam_id] = right_pos_local
        self.sim.model.cam_quat[right_cam_id] = right_quat_local
        self.sim.forward()

        return obs

    def _compute_az_el(self, unit_dir: np.ndarray) -> tuple[float, float]:
        """
        Decomposes unit_dir into (azimuth_deg, elevation_deg) in the (forward, right, up) frame.
        Azimuth 0 = forward, +90 = right, -90 = left. Elevation 0 = horizontal, +90 = straight up.
        """
        unit_dir = np.asarray(unit_dir, dtype=np.float64)
        sin_el = float(np.clip(np.dot(unit_dir, self.ref_up), -1.0, 1.0))
        el_deg = float(np.degrees(np.arcsin(sin_el)))

        v_horiz = unit_dir - sin_el * self.ref_up.astype(np.float64)
        norm = float(np.linalg.norm(v_horiz))
        if norm < 1e-6:
            return 0.0, el_deg  # at a pole; azimuth is undefined

        v_horiz /= norm
        cos_az = float(np.clip(np.dot(v_horiz, self.ref_forward), -1.0, 1.0))
        sin_az = float(np.dot(v_horiz, self.ref_right))
        az_deg = float(np.degrees(np.arctan2(sin_az, cos_az)))
        return az_deg, el_deg

    def _sample_unit_vectors_random(
        self,
        count: int,
        az_range: tuple[float, float],
        el_range: tuple[float, float],
    ) -> np.ndarray:
        """
        Randomly samples 'count' unit directions uniformly within the given
        azimuth/elevation range (in degrees) in the (forward, right, up) frame.
        Elevation is clamped to (-89.9, 89.9) to avoid gimbal lock at the poles.
        """
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        az_min, az_max = az_range
        el_min = max(float(el_range[0]), 0.0)
        el_max = min(float(el_range[1]),  89.9)

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
        return dirs.astype(np.float32)

    def _sample_camera_poses(self):
        """
        Resample all virtual camera poses for the current episode.

        Steps:
          1. Find pivot from init_robot_base_pos.
          2. Compute forward (pivot->robot, horizontal) and right axes.
          3. Compute azimuth/elevation of default left/right cameras from pivot.
          4. Randomly sample num_cameras directions per side within az_half / el_half of those centers.
          5. Build world-space poses (position + look-at rotation), marking occluded ones invalid.
        """
        self._refresh_cam_ids()

        ref_fixture = self.env.init_robot_base_pos
        pivot = np.array(ref_fixture.pos, dtype=np.float32)
        self.reference_pivot = pivot

        left_pos_w, _ = self._get_camera_world_pose(self.left_cam_name)
        right_pos_w, _ = self._get_camera_world_pose(self.right_cam_name)

        v_left = (left_pos_w - pivot).astype(np.float32)
        v_right = (right_pos_w - pivot).astype(np.float32)

        left_radius = float(np.linalg.norm(v_left))
        right_radius = float(np.linalg.norm(v_right))
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

        right = np.cross(forward, self.ref_up).astype(np.float32)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = right / np.linalg.norm(right)

        self.ref_forward = forward
        self.ref_right = right

        # Determine azimuth / elevation of the default camera positions relative to pivot
        self._left_az_center, self._left_el_center = self._compute_az_el(v_left / left_radius)
        self._right_az_center, self._right_el_center = self._compute_az_el(v_right / right_radius)

        # Randomly sample new positions for this episode within the small cap
        left_dirs = self._sample_unit_vectors_random(
            self.num_cameras,
            az_range=(self._left_az_center - self.az_half, self._left_az_center + self.az_half),
            el_range=(self._left_el_center - self.el_half, self._left_el_center + self.el_half),
        )
        right_dirs = self._sample_unit_vectors_random(
            self.num_cameras,
            az_range=(self._right_az_center - self.az_half, self._right_az_center + self.az_half),
            el_range=(self._right_el_center - self.el_half, self._right_el_center + self.el_half),
        )

        r_lo, r_hi = self.radius_range
        left_scales = self._rng.uniform(r_lo, r_hi, size=len(left_dirs))
        right_scales = self._rng.uniform(r_lo, r_hi, size=len(right_dirs))

        self._left_poses = []
        self._right_poses = []
        for direction, scale in zip(left_dirs, left_scales):
            pos_w = pivot + left_radius * scale * direction
            if self._is_occluded(pos_w, pivot):
                self._left_poses.append((pos_w, None))
                continue
            rot_w = self._look_at_rotation(pos_w, pivot)
            self._left_poses.append((pos_w, rot_w))

        for direction, scale in zip(right_dirs, right_scales):
            pos_w = pivot + right_radius * scale * direction
            if self._is_occluded(pos_w, pivot):
                self._right_poses.append((pos_w, None))
                continue
            rot_w = self._look_at_rotation(pos_w, pivot)
            self._right_poses.append((pos_w, rot_w))

        self._virtual_pose_map = {
            **{name: pose for name, pose in zip(self._left_names, self._left_poses)},
            **{name: pose for name, pose in zip(self._right_names, self._right_poses)},
        }
        self._virtual_valid = {
            name: (rot_w is not None) for name, (_, rot_w) in self._virtual_pose_map.items()
        }

    def _look_at_rotation(self, pos_w: np.ndarray, target_w: np.ndarray) -> Rotation:
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

    def get_virtual_camera_names(self) -> list[str]:
        return list(self._left_names) + list(self._right_names)

    def get_virtual_camera_pose(self, cam_name: str):
        if cam_name not in self._virtual_pose_map:
            raise KeyError(f"Unknown virtual camera name: {cam_name}")
        return self._virtual_pose_map[cam_name]

    def virtual_intrinsic_fn(self, cam_name: str, width: int, height: int) -> np.ndarray:
        self._refresh_cam_ids()
        base_cam = self.left_cam_name if cam_name.startswith(self.left_prefix) else self.right_cam_name
        cam_id = self.sim.model.camera_name2id(base_cam)
        fovy = self.sim.model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def virtual_extrinsic_fn(self, cam_name: str) -> np.ndarray:
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

    def _check_success(self):
        return self.env._check_success()


if __name__ == "__main__":
    from robocasa.utils.env_utils import create_env
    import cv2

    base_env = create_env(env_name="TurnOnMicrowave")
    wrapped_env = RobocasaLocalSphereCameraWrapper(
        base_env,
        num_cameras=3,
        include_depth=False,
        seed=None,
        az_half=15.0,
        el_half=15.0,
    )

    obs = wrapped_env.reset()
    print(
        f"Left camera center:  az={wrapped_env._left_az_center:.1f}°  el={wrapped_env._left_el_center:.1f}°"
    )
    print(
        f"Right camera center: az={wrapped_env._right_az_center:.1f}°  el={wrapped_env._right_el_center:.1f}°"
    )


    low, high = wrapped_env.action_spec
    action = np.random.uniform(low=low, high=high)
    obs, reward, done, info = wrapped_env.step(action)

    images = {
        "robot0_agentview_left_image": obs["robot0_agentview_left_image"],
        "robot0_agentview_right_image": obs["robot0_agentview_right_image"],
    }

    for i in range(1, wrapped_env.num_cameras + 1):
        left_name = f"{wrapped_env.left_prefix}_{i:02d}"
        right_name = f"{wrapped_env.right_prefix}_{i:02d}"
        images[left_name] = obs[f"{left_name}_image"]
        images[right_name] = obs[f"{right_name}_image"]

        if obs.get(f"{left_name}_valid", np.array([0]))[0] == 1:
            pos_w, _ = wrapped_env.get_virtual_camera_pose(left_name)
            az, el = wrapped_env._compute_az_el((pos_w - wrapped_env.reference_pivot) / np.linalg.norm(pos_w - wrapped_env.reference_pivot))
            print(f"  {left_name}  pos={np.round(pos_w, 3)}  az={az:.1f}°  el={el:.1f}°")
        else:
            print(f"  {left_name}  [occluded]")
        if obs.get(f"{right_name}_valid", np.array([0]))[0] == 1:
            pos_w, _ = wrapped_env.get_virtual_camera_pose(right_name)
            az, el = wrapped_env._compute_az_el((pos_w - wrapped_env.reference_pivot) / np.linalg.norm(pos_w - wrapped_env.reference_pivot))
            print(f"  {right_name}  pos={np.round(pos_w, 3)}  az={az:.1f}°  el={el:.1f}°")
        else:
            print(f"  {right_name}  [occluded]")

    for key, img in images.items():
        cv2.imshow(key, img[::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
