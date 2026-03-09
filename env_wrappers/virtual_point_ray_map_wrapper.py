import gym
import numpy as np
import open3d as o3d
import einops

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import robosuite.utils.transform_utils as T
from robosuite import macros


class VirtualPointRayMapWrapper(gym.Wrapper):
    """
    Point / ray map wrapper that supports virtual cameras (not registered in sim.model).

    You must provide:
    - virtual_cam_names: list of virtual camera names
    - virtual_intrinsic_fn(cam_name, width, height) -> 3x3 K
    - virtual_extrinsic_fn(cam_name) -> 4x4 camera-to-world pose
    """

    def __init__(
        self,
        env,
        virtual_cam_names,
        virtual_intrinsic_fn,
        virtual_extrinsic_fn,
        include_real_cams=True,
        global_frame=False,
    ):
        super().__init__(env)
        self.virtual_cam_names = list(virtual_cam_names)
        self.virtual_intrinsic_fn = virtual_intrinsic_fn
        self.virtual_extrinsic_fn = virtual_extrinsic_fn
        self.include_real_cams = bool(include_real_cams)
        self.global_frame = bool(global_frame)

        self.real_cam_names = list(env.camera_names) if self.include_real_cams else []
        self.cam_names = self.real_cam_names + self.virtual_cam_names

        # assume all cameras share the same render size as env.camera_widths/heights[0]
        self.img_width = env.camera_widths[0]
        self.img_height = env.camera_heights[0]

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]

        return obs_dict

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]

        return obs_dict

    def _get_cam_intrinsic_matrix(self, cam_name: str):
        if cam_name in self.real_cam_names:
            return get_camera_intrinsic_matrix(
                sim=self.env.sim,
                camera_name=cam_name,
                camera_height=self.img_height,
                camera_width=self.img_width,
            )
        return self.virtual_intrinsic_fn(cam_name, self.img_width, self.img_height)

    def _get_cam_intrinsic(self, cam_name: str):
        cam_mat = self._get_cam_intrinsic_matrix(cam_name)
        cx = cam_mat[0, 2]
        fx = cam_mat[0, 0]
        cy = cam_mat[1, 2]
        fy = cam_mat[1, 1]
        return o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, fx, fy, cx, cy)

    def _get_cam_extrinsic(self, cam_name: str):
        if cam_name in self.real_cam_names:
            return get_camera_extrinsic_matrix(self.env.sim, cam_name)
        return self.virtual_extrinsic_fn(cam_name)

    def get_point_map_and_ray_map(self, obs_dict):
        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}

        n_pts = self.img_height * self.img_width
        o3d_point_cloud = o3d.geometry.PointCloud()
        for cam in self.cam_names:
            valid_key = f"{cam}_valid"
            if valid_key in obs_dict and int(np.asarray(obs_dict[valid_key]).item()) == 0:
                # Invalid camera: insert zero-filled dummy to preserve per-camera
                # slice positions in the concatenated point cloud for rearrange.
                dummy = o3d.geometry.PointCloud()
                dummy.points = o3d.utility.Vector3dVector(
                    np.zeros((n_pts, 3), dtype=np.float64)
                )
                o3d_point_cloud += dummy
                continue

            cam_intrinsics = self._get_cam_intrinsic(cam)
            # Replace zero-depth pixels with a tiny value so that
            # create_from_depth_image does not silently drop them.
            # Dropping pixels causes the point count to differ from H*W
            # and breaks the subsequent einops rearrange.
            depth_arr = np.asarray(depths[cam]).copy()
            depth_arr[depth_arr == 0] = 1e-9
            o3d_depth = o3d.geometry.Image(depth_arr)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, cam_intrinsics
            )

            cam_pose = self._get_cam_extrinsic(cam)
            transformed_cloud = o3d_cloud.transform(cam_pose)

            if not self.global_frame:
                base_pos = self.env.sim.data.get_site_xpos("mobilebase0_center")
                base_rot = self.env.sim.data.get_site_xmat("mobilebase0_center")
                base_pose = T.pose_inv(T.make_pose(base_pos, base_rot))
                transformed_cloud = transformed_cloud.transform(base_pose)

            o3d_point_cloud += transformed_cloud

        pc_points = np.asarray(o3d_point_cloud.points)
        pc = (
            einops.rearrange(
                pc_points[:, :3],
                "(num_cam h w) c -> num_cam h w c",
                num_cam=len(self.cam_names),
                h=self.img_height,
                w=self.img_width,
            )
        ).astype(np.float32)

        point_map = {}
        ray_map = {}
        for i, cam in enumerate(self.cam_names):
            point_map[cam] = pc[i]
            valid_key = f"{cam}_valid"
            if valid_key in obs_dict and int(np.asarray(obs_dict[valid_key]).item()) == 0:
                # Invalid camera: zero-filled ray map, consistent with zero point map.
                ray_map[cam] = np.zeros((self.img_height, self.img_width, 6), dtype=np.float32)
                continue
            if not self.global_frame:
                ray_map[cam] = self.get_ray_map_based_on_mobilebase(cam)
            else:
                ray_map[cam] = self.get_ray_map(cam)

        return point_map, ray_map

    def get_ray_map(self, cam_name: str):
        K = self._get_cam_intrinsic_matrix(cam_name)
        R = self._get_cam_extrinsic(cam_name)  # camera->world

        u = np.arange(self.img_width) + 0.5  # pixel centers
        v = np.arange(self.img_height) + 0.5
        uu, vv = np.meshgrid(u, v)

        ones = np.ones_like(uu)
        if macros.IMAGE_CONVENTION != "opengl":
            # OpenCV convention: z = 1
            pix = np.stack([uu, vv, ones], axis=-1).astype(np.float32)  # HxW x3, x y z
        else:
            # OpenGL convention: z = -1, flip Y axis  # robocasa default is OpenGL convention, so this is the default behavior
            pix = np.stack([uu, -vv, -ones], axis=-1).astype(np.float32)

        Kinv = np.linalg.inv(K)
        d_cam = pix @ Kinv.T
        d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8

        Rwc = R[:3, :3]
        C = R[:3, 3]

        d_world = d_cam @ Rwc.T
        m = np.cross(C[None, None, :], d_world)

        ray_map = np.concatenate([d_world, m], axis=-1).astype(np.float32)
        return ray_map

    def get_ray_map_based_on_mobilebase(self, cam_name: str):
        """
        Returns the ray map expressed in the mobilebase0_center frame,
        consistent with the point map produced when global_frame=False.

        Each pixel is encoded as a 6D Plücker-like vector [d_base, m_base]:
          - d_base : unit ray direction in the mobilebase frame
          - m_base : C_base × d_base, where C_base is the camera origin in the mobilebase frame
        """
        K = self._get_cam_intrinsic_matrix(cam_name)
        R = self._get_cam_extrinsic(cam_name)  # 4x4 camera->world

        u = np.arange(self.img_width) + 0.5
        v = np.arange(self.img_height) + 0.5
        uu, vv = np.meshgrid(u, v)

        ones = np.ones_like(uu)
        if macros.IMAGE_CONVENTION != "opengl":
            pix = np.stack([uu, vv, ones], axis=-1).astype(np.float32)
        else:
            pix = np.stack([uu, -vv, -ones], axis=-1).astype(np.float32)

        Kinv = np.linalg.inv(K)
        d_cam = pix @ Kinv.T
        d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8

        Rwc = R[:3, :3]   # camera->world rotation
        C_world = R[:3, 3]  # camera origin in world frame

        d_world = d_cam @ Rwc.T  # HxWx3

        # mobilebase0_center pose (world frame)
        base_pos = self.env.sim.data.get_site_xpos("mobilebase0_center")        # (3,)
        base_rot = self.env.sim.data.get_site_xmat("mobilebase0_center").reshape(3, 3)  # R_wb

        # world->base transform: R_bw = R_wb.T, t_bw = -R_bw @ base_pos
        R_bw = base_rot.T
        t_bw = -R_bw @ base_pos

        # transform ray directions (rotation only)
        d_base = d_world @ R_bw.T  # HxWx3

        # transform camera origin
        C_base = R_bw @ C_world + t_bw  # (3,)

        # moment in the base frame
        m_base = np.cross(C_base[None, None, :], d_base)

        return np.concatenate([d_base, m_base], axis=-1).astype(np.float32)

    def _check_success(self):
        return self.env._check_success()

if __name__ == "__main__":
    import numpy as np
    import cv2
    import os
    import sys
    from robocasa.utils.env_utils import create_env
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from env_wrappers.robocasa_quarter_sphere_camera_wrapper import RobocasaQuarterSphereCameraWrapper

    env = create_env(env_name="TurnSinkSpout")
    env = RobocasaQuarterSphereCameraWrapper(env, num_cameras=8, include_depth=True)
    env = VirtualPointRayMapWrapper(
        env,
        virtual_cam_names=env.get_virtual_camera_names(),
        virtual_intrinsic_fn=env.virtual_intrinsic_fn,
        virtual_extrinsic_fn=env.virtual_extrinsic_fn,
        include_real_cams=True,
    )

    obs = env.reset()
    # Check that virtual images/depths exist
    for name in env.get_virtual_camera_names():
        assert f"{name}_image" in obs, name
        assert f"{name}_depth" in obs, name
        assert f"{name}_point" in obs, name
        assert f"{name}_ray" in obs, name

    # Step once and ensure outputs are still present
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)

    images = {}
    for name in env.cam_names:
        if f"{name}_valid" in obs and obs[f"{name}_valid"][0] == 1:
            print(f"Virtual camera {name} is valid.")
        # images[name] = obs[f"{name}_image"]

    # for key, img in images.items():
    #     img = img[::-1]
    #     cv2.imshow(f"{key}", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print("OK: virtual cameras + point/ray maps present")
