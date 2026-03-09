"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Single-process version of dataset_states_to_obs.py — no multiprocessing, simpler and more robust.
"""

import os
import json
from typing import OrderedDict
import h5py
import argparse
import numpy as np
from copy import deepcopy
import time
import traceback

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from robocasa.utils.env_utils import create_env
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
from tqdm import tqdm

from custom_robocasa.env_wrappers.robosuite_wrapper import RobosuiteWrapper
from custom_robocasa.env_wrappers.point_ray_map_wrapper import PointRayMapWrapper
from custom_robocasa.env_wrappers.virtual_point_ray_map_wrapper import VirtualPointRayMapWrapper
from custom_robocasa.env_wrappers.robocasa_quarter_sphere_camera_wrapper import RobocasaQuarterSphereCameraWrapper


def extract_trajectory(env, initial_state, states, actions, done_mode, args, demo_th=None):
    assert states.shape[0] == actions.shape[0]

    env.reset()
    env.reset_to(initial_state)

    if env.reference_pivot is not None and demo_th:
        print(f"{demo_th} with reference pivot: {env.reference_pivot}")

    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )

    traj_len = states.shape[0]
    for t in tqdm(range(traj_len)):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # drop invalid virtual camera observations
        invalid_suffixes = ("_image", "_point", "_ray", "_depth")
        for k in list(obs.keys()):
            if k.endswith("_valid"):
                try:
                    is_valid = int(np.asarray(obs[k]).item())
                except Exception:
                    continue
                if is_valid == 0:
                    prefix = k[: -len("_valid")]
                    for suf in invalid_suffixes:
                        key = f"{prefix}{suf}"
                        if key in obs:
                            del obs[key]

        if args.point_ray_dtype in ("float16", "float32"):
            dtype = np.float16 if args.point_ray_dtype == "float16" else np.float32
            for k in list(obs.keys()):
                if k.endswith("_point") or k.endswith("_ray") or k.endswith("_depth"):
                    obs[k] = obs[k].astype(dtype, copy=False)

        r = env.reward()

        done = False
        if (done_mode == 1) or (done_mode == 2):
            done = done or (t == traj_len - 1)
        if (done_mode == 0) or (done_mode == 2):
            done = done or env._check_success()
        done = int(done)

        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append({})

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["datagen_info"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                if isinstance(traj[k][kp][0], dict):
                    traj[k][kp] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj[k][kp])
                    for kpp in traj[k][kp]:
                        traj[k][kp][kpp] = np.array(traj[k][kp][kpp])
                else:
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def write_traj_to_dataset(ep_data_grp, traj, args, f_src, ep):
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

    for k in traj["obs"]:
        if isinstance(traj["obs"][k], OrderedDict):
            for kp in traj["obs"][k]:
                if args.no_compress:
                    ep_data_grp.create_dataset(
                        "obs/{}/{}".format(k, kp), data=np.array(traj["obs"][k][kp])
                    )
                else:
                    ep_data_grp.create_dataset(
                        "obs/{}/{}".format(k, kp),
                        data=np.array(traj["obs"][k][kp]),
                        compression="gzip",
                    )
            continue
        if args.no_compress:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
        else:
            ep_data_grp.create_dataset(
                "obs/{}".format(k),
                data=np.array(traj["obs"][k]),
                compression="gzip",
            )

    if "datagen_info" in traj:
        for k in traj["datagen_info"]:
            ep_data_grp.create_dataset(
                "datagen_info/{}".format(k), data=np.array(traj["datagen_info"][k])
            )

    # copy action dict if present in source
    if "data/{}/action_dict".format(ep) in f_src:
        action_dict = f_src["data/{}/action_dict".format(ep)]
        for k in action_dict:
            ep_data_grp.create_dataset(
                "action_dict/{}".format(k), data=np.array(action_dict[k][()])
            )

    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"]["ep_meta"]
    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]


def create_env_with_wrappers(env_name, args):
    base_env = create_env(
        env_name=env_name,
        camera_names=args.camera_names,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )

    env = RobosuiteWrapper(base_env)
    if args.use_virtual_cameras:
        env = RobocasaQuarterSphereCameraWrapper(
            env,
            num_cameras=args.num_virtual_cameras,
            include_depth=True,
            seed=42,
        )
        env = VirtualPointRayMapWrapper(
            env,
            virtual_cam_names=env.get_virtual_camera_names(),
            virtual_intrinsic_fn=env.virtual_intrinsic_fn,
            virtual_extrinsic_fn=env.virtual_extrinsic_fn,
            include_real_cams=True,
        )
    else:
        env = PointRayMapWrapper(env)

    return env


def dataset_states_to_obs(args):
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[:-5] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(image_suffix)

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)
    print("input file:  {}".format(args.dataset))
    print("output file: {}".format(output_path))

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")

    env = create_env_with_wrappers(env_meta["env_name"], args)

    total_samples = 0
    num_processed = 0
    start_time = time.time()

    for ind, ep in enumerate(demos):
        print("\n[{}/{}] Processing {}".format(ind + 1, len(demos), ep))
        try:
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
            actions = f["data/{}/actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                args=args,
                demo_th=ep,
            )

            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_data_grp = data_grp.create_group(ep)
            write_traj_to_dataset(ep_data_grp, traj, args, f, ep)

            total_samples += traj["actions"].shape[0]
            num_processed += 1
            elapsed = time.time() - start_time
            print("  wrote {} transitions | total demos done: {} | rate: {:.2f} sec/demo".format(
                ep_data_grp.attrs["num_samples"], num_processed, elapsed / num_processed
            ))

        except Exception as e:
            print("=" * 60)
            print("Error on {}: {}".format(ep, e))
            print(traceback.format_exc())
            print("=" * 60)
            print("Recreating environment and continuing...")
            try:
                del env
            except Exception:
                pass
            env = create_env_with_wrappers(env_meta["env_name"], args)

    if "mask" in f:
        f.copy("mask", f_out)

    data_grp.attrs["total"] = total_samples
    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    for num_demos in [
        10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100,
        125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
        1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000,
    ]:
        DatasetUtils.filter_dataset_size(output_path, num_demos=num_demos)

    elapsed = time.time() - start_time
    print("\nWrote {} total samples ({} demos) to {}".format(total_samples, num_processed, output_path))
    print("Total time: {:.2f} seconds".format(elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="", help="path to input hdf5 dataset")
    parser.add_argument("--output_name", type=str, default="multi_view_mobilebase.hdf5", help="name of output hdf5 dataset")
    parser.add_argument("--filter_key", type=str, help="filter key for input dataset")
    parser.add_argument("--n", type=int, default=None, help="(optional) stop after n trajectories are processed")
    parser.add_argument("--shaped", action="store_true", help="(optional) use shaped rewards")
    parser.add_argument(
        "--camera_names", type=str, nargs="+",
        default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        help="camera name(s) to use for image observations",
    )
    parser.add_argument("--camera_height", type=int, default=128)
    parser.add_argument("--camera_width", type=int, default=128)
    parser.add_argument("--done_mode", type=int, default=0,
        help="0: done at success state; 1: done at end of traj; 2: both")
    parser.add_argument("--copy_rewards", action="store_true", default=False)
    parser.add_argument("--copy_dones", action="store_true", default=False)
    parser.add_argument("--no_compress", action="store_true", help="disable gzip compression")
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--use_virtual_cameras", default=True, help="use virtual multiview cameras")
    parser.add_argument("--num_virtual_cameras", type=int, default=8, help="number of virtual cameras per side")
    parser.add_argument(
        "--point_ray_dtype", type=str, default="float16", choices=["float16", "float32"],
        help="dtype to store point/ray maps",
    )

    args = parser.parse_args()

    # data_directory = "/home/qian-wang/depth_vla/robocasa/datasets/v0.1/single_stage"
    data_directory = "/hkfs/work/workspace/scratch/mn4777-qian_space/robocasa/datasets/v0.1/single_stage"
    env_name = [
        # "PnPCounterToMicrowave",
        "PnPCounterToSink",
        "PnPMicrowaveToCounter",
        "PnPSinkToCounter",
        "TurnOffStove",
        "TurnOnStove",
        "TurnOffSinkFaucet",
        "TurnOnSinkFaucet",
        "TurnOffMicrowave",
        "TurnOnMicrowave",
        "CloseDrawer",
        "OpenDrawer",
        "CoffeePressButton",
        "CoffeeServeMug",
        "CoffeeSetupMug",
    ]

    for env in env_name:
        if "PnP" in env:
            data_dir = os.path.join(data_directory, "kitchen_pnp", env)
        elif "Door" in env:
            data_dir = os.path.join(data_directory, "kitchen_doors", env)
        elif "Drawer" in env:
            data_dir = os.path.join(data_directory, "kitchen_drawer", env)
        elif "Coffee" in env:
            data_dir = os.path.join(data_directory, "kitchen_coffee", env)
        elif "Stove" in env:
            data_dir = os.path.join(data_directory, "kitchen_stove", env)
        elif "Sink" in env:
            data_dir = os.path.join(data_directory, "kitchen_sink", env)
        elif "Microwave" in env:
            data_dir = os.path.join(data_directory, "kitchen_microwave", env)
        else:
            raise ValueError(f"Unknown environment: {env}")

        subdirs = sorted(os.listdir(data_dir))
        data_dir = os.path.join(data_dir, subdirs[0], "demo_gentex_im128_randcams.hdf5")
        args.dataset = data_dir

        print("\n" + "=" * 60)
        print("Processing environment: {}".format(env))
        print("=" * 60)
        try:
            dataset_states_to_obs(args)
        except Exception as e:
            print("Error processing env {}: {}".format(env, e))
            print(traceback.format_exc())