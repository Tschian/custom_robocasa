import gym
from custom_robocasa.utils.point_cloud.pc_generator import PointCloudGenerator

class PointRayMapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(env.sim, self.cam_names, env.camera_widths[0], env.camera_heights[0], global_frame=False)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]
        
        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        self.pc_generator.sim = self.env.sim
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]

        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        self.pc_generator.sim = self.env.sim
        point_map, ray_map = self.get_point_map_and_ray_map(obs_dict)

        for cam in self.cam_names:
            obs_dict[f"{cam}_point"] = point_map[cam]
            obs_dict[f"{cam}_ray"] = ray_map[cam]

        return obs_dict
    
    def _check_success(self):
        return self.env._check_success()

    def get_point_map_and_ray_map(self, obs_dict):
        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}

        point_map, ray_map = self.pc_generator.get_point_map_and_ray_map(imgs, depths)
        return point_map, ray_map

