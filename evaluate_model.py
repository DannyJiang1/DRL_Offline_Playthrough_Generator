# evaluate_ppo.py
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

CHECKPOINT_STEPS = [5_000, 10_000, 20_000, 50_000]
SAVE_DIR = "ppo_checkpoints"

def evaluate(model, n_episodes=10):
    env = gym.make("CartPole-v1")
    scores = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_r = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward

        scores.append(total_r)

    return np.mean(scores)

if __name__ == "__main__":
    print("\n==============================")
    print("📊 Evaluating PPO Checkpoints")
    print("==============================\n")

    for ckpt in CHECKPOINT_STEPS:
        ckpt_path = os.path.join(SAVE_DIR, f"ppo_cartpole_{ckpt}.zip")

        if not os.path.exists(ckpt_path):
            print(f"⚠️  Missing checkpoint: {ckpt_path}")
            continue

        model = PPO.load(ckpt_path)
        mean_r = evaluate(model)

        print(f"Checkpoint {ckpt:>7d} steps → mean return: {mean_r:.2f}")
