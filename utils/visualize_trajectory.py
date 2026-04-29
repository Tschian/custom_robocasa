import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
import glob

def visualize_trajectory(env_name: str, data_dir: str = "./data"):
    # Find dataset file for the given env
    pattern = os.path.join(data_dir, f"*{env_name}*.hdf5")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(data_dir, f"*{env_name}*.h5")
        files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No dataset found for env '{env_name}' in {data_dir}")

    filepath = files[0]
    print(f"Loading: {filepath}")

    with h5py.File(filepath, "r") as f:
        # List available trajectories
        traj_keys = list(f.keys())
        print(f"Found {len(traj_keys)} trajectories: {traj_keys[:5]}...")

        # Randomly select one trajectory
        traj_key = random.choice(traj_keys)
        print(f"Selected trajectory: {traj_key}")
        traj = f[traj_key]

        # Discover all keys in this trajectory
        def collect_keys(obj, prefix=""):
            keys = []
            for k in obj.keys():
                full_key = f"{prefix}/{k}" if prefix else k
                if isinstance(obj[k], h5py.Dataset):
                    keys.append(full_key)
                else:
                    keys.extend(collect_keys(obj[k], full_key))
            return keys

        all_keys = collect_keys(traj)
        print(f"Available keys: {all_keys}")

        # Find image keys (contain 'image' or 'rgb' in name)
        image_keys = [k for k in all_keys if any(
            kw in k.lower() for kw in ["image", "rgb", "color"]
        )]
        # Find depth keys
        depth_keys = [k for k in all_keys if "depth" in k.lower()]

        print(f"Image keys: {image_keys}")
        print(f"Depth keys: {depth_keys}")

        # Load data
        def load_key(traj, key):
            parts = key.split("/")
            obj = traj
            for p in parts:
                obj = obj[p]
            return obj[:]

        image_data = {k: load_key(traj, k) for k in image_keys}
        depth_data = {k: load_key(traj, k) for k in depth_keys}

        # Determine number of timesteps
        all_data = {**image_data, **depth_data}
        if not all_data:
            print("No image or depth data found in this trajectory.")
            return

        T = next(iter(all_data.values())).shape[0]
        timesteps = np.arange(T)

        # --- Plot ---
        n_image_cols = len(image_keys)
        n_depth_cols = len(depth_keys)
        total_cols = max(n_image_cols + n_depth_cols, 1)

        # Show a subset of frames if trajectory is long
        frame_indices = np.linspace(0, T - 1, min(8, T), dtype=int)

        n_rows = len(frame_indices)
        n_cols = total_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 3 * n_rows))
        fig.suptitle(f"Env: {env_name} | Trajectory: {traj_key} | "
                     f"{T} timesteps", fontsize=13)

        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        col_labels = image_keys + depth_keys

        for row_i, t in enumerate(frame_indices):
            for col_i, key in enumerate(col_labels):
                ax = axes[row_i][col_i]
                data = all_data[key][t]

                # Squeeze single-channel dims
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                elif data.ndim == 3 and data.shape[-1] == 1:
                    data = data[..., 0]

                is_depth = "depth" in key.lower()
                if is_depth:
                    im = ax.imshow(data, cmap="plasma")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # Normalize uint8 or float images
                    if data.dtype != np.uint8:
                        data = np.clip(data, 0, 1)
                    ax.imshow(data)

                if row_i == 0:
                    ax.set_title(key, fontsize=8)
                ax.set_ylabel(f"t={t}", fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{env_name}_{traj_key}_visualization.png",
                    dpi=120, bbox_inches="tight")
        plt.show()
        print(f"Saved: {env_name}_{traj_key}_visualization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True,
                        help="Environment name (used to find dataset file)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing .hdf5/.h5 dataset files")
    args = parser.parse_args()

    visualize_trajectory(args.env_name, args.data_dir)