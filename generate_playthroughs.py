# collect_trajectories.py

import os
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import cv2



CHECKPOINT_STEPS = [5_000, 10_000, 20_000, 50_000]
SAVE_DIR = "ppo_checkpoints"              # where checkpoints are stored
DATASET_DIR = "ppo_datasets"              # where datasets will be saved
os.makedirs(DATASET_DIR, exist_ok=True)


def preprocess_cartpole_obs(env):
    """Render CartPole frame and convert to 84x84 grayscale."""
    frame = env.render()  # gymnasium already returns RGB array if render_mode="rgb_array"

    # grayscale conversion
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # add channel dimension
    return resized[:, :, None].astype(np.uint8)

def compute_rtg(rewards):
    """Return-to-go: reverse cumulative sum."""
    return np.flip(np.cumsum(np.flip(rewards)))


def collect_episodes(model, n_episodes=500):
    """
    Roll out episodes using a PPO model.
    Return a list of trajectory dictionaries.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    dataset = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        obs_list = []
        act_list = []
        rew_list = []

        while not done:
            # preprocess the rendered image
            img_obs = preprocess_cartpole_obs(env)
            obs_list.append(img_obs)

            # policy expects vector obs, not images
            action, _ = model.predict(obs, deterministic=True)

            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            act_list.append(int(action))
            rew_list.append(reward)

            obs = next_obs

        # compute return-to-go
        rtg = compute_rtg(np.array(rew_list))

        dataset.append({
            "observations": np.array(obs_list),       # [T, 84, 84, 1]
            "actions": np.array(act_list),            # [T]
            "rewards": np.array(rew_list),            # [T]
            "returns_to_go": rtg,                     # [T]
        })

        print(f"Collected episode {ep+1}/{n_episodes}")

    env.close()
    return dataset



if __name__ == "__main__":
    print("\n==============================")
    print("📦 Generating PPO Trajectory Datasets")
    print("==============================\n")

    for ckpt in CHECKPOINT_STEPS:

        ckpt_path = os.path.join(SAVE_DIR, f"ppo_cartpole_{ckpt}.zip")

        if not os.path.exists(ckpt_path):
            print(f"⚠️  Checkpoint {ckpt} is missing, skipping...")
            continue

        print(f"\n📌 Loading checkpoint {ckpt}...")
        model = PPO.load(ckpt_path)

        print(f"🎬 Collecting trajectories for {ckpt}...")
        dataset = collect_episodes(model, n_episodes=500)

        dataset_path = os.path.join(DATASET_DIR, f"ppo_cartpole_{ckpt}_dataset.pkl")
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"💾 Saved dataset to {dataset_path}")