import open3d as o3d
import numpy as np
import einops
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import robosuite.utils.transform_utils as T


class PointCloudGenerator:
    def __init__(
        self,
        sim,
        cam_names: list[str],
        img_width: int,
        img_height: int,
        global_frame: bool,
    ):
        self.sim = sim
        self.cam_names = cam_names
        self.img_width = img_width
        self.img_height = img_height
        self.global_frame = global_frame

    def get_point_cloud(
        self, imgs: dict[str, np.ndarray], depths: dict[str, np.ndarray]
    ) -> np.ndarray:
        o3d_point_cloud = o3d.geometry.PointCloud()
        colors = []

        for cam in self.cam_names:
            colors.append(imgs[cam])

            cam_intrinsics = self._get_cam_intrinsic(
                cam, self.img_width, self.img_height
            )

            o3d_depth = o3d.geometry.Image(depths[cam])
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, cam_intrinsics
            )

            cam_pose = get_camera_extrinsic_matrix(self.sim, cam)
            transformed_cloud = o3d_cloud.transform(cam_pose)

            if not self.global_frame:
                # base_pos = self.sim.data.get_site_xpos(f"mobilebase0_center")
                # base_rot = self.sim.data.get_site_xmat(f"mobilebase0_center")
                #
                # base_pose = T.pose_inv(T.make_pose(base_pos, base_rot))
                # transformed_cloud = transformed_cloud.transform(base_pose)
                if cam == "robot0_agentview_left":
                    reference_pose = cam_pose

                transformed_cloud = transformed_cloud.transform(reference_pose)

            o3d_point_cloud += transformed_cloud

        pc_points = np.asarray(o3d_point_cloud.points)
        pc = (
            einops.rearrange(
                pc_points[:, :3],
                "(num_cam h w) c -> num_cam h w c",
                num_cam=len(self.cam_names),
                h=224,
                w=224,
            )
        ).astype(np.float32)
        # pc_colors = np.array(colors).reshape(-1, 3)
        #
        # pc = np.concatenate([pc_points, pc_colors], axis=1)
        return pc

    def get_segmented_point_cloud(
        self,
        imgs: dict[str, np.ndarray],
        depths: dict[str, np.ndarray],
        segmentation: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        o3d_clouds = {}
        colors = {}

        for cam in self.cam_names:
            for class_name in segmentation[cam]:
                if class_name not in o3d_clouds:
                    o3d_clouds[class_name] = []
                    colors[class_name] = []

                colors[class_name].append(imgs[cam][segmentation[cam][class_name]])

                depth = depths[cam].copy()
                depth[~segmentation[cam][class_name]] = (
                    -10
                )  # any invalid depth value is fine

                cam_intrinsics = self._get_cam_intrinsic(
                    cam, self.img_width, self.img_height
                )

                od_depth = o3d.geometry.Image(depth)
                o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                    od_depth, cam_intrinsics
                )

                cam_pose = get_camera_extrinsic_matrix(self.sim, cam)
                transformed_cloud = o3d_cloud.transform(cam_pose)

                if not self.global_frame:
                    base_pos = self.sim.data.get_site_xpos(f"mobilebase0_center")
                    base_rot = self.sim.data.get_site_xmat(f"mobilebase0_center")

                    base_pose = T.make_pose(base_pos, base_rot)
                    transformed_cloud = transformed_cloud.transform(
                        T.pose_inv(base_pose)
                    )

                o3d_clouds[class_name].append(transformed_cloud)

        class_to_point_cloud = {}
        for class_name, clouds in o3d_clouds.items():
            class_cloud = o3d.geometry.PointCloud()
            for cloud in clouds:
                class_cloud += cloud

            colors[class_name] = np.concatenate(colors[class_name])

            class_to_point_cloud[class_name] = np.concatenate(
                [np.asarray(class_cloud.points), colors[class_name]], axis=1
            )

        return class_to_point_cloud

    def _get_cam_intrinsic(self, cam_name: str, img_width: int, img_height: int):
        cam_mat = get_camera_intrinsic_matrix(
            sim=self.sim,
            camera_name=cam_name,
            camera_height=img_height,
            camera_width=img_width,
        )

        cx = cam_mat[0, 2]
        fx = cam_mat[0, 0]
        cy = cam_mat[1, 2]
        fy = cam_mat[1, 1]

        return o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)
