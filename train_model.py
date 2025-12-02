# train_ppo.py
import os
import gymnasium as gym
from stable_baselines3 import PPO

# Where to save checkpoints
SAVE_DIR = "ppo_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# At what timesteps to save the model
CHECKPOINT_STEPS = [5_000, 10_000, 20_000, 50_000]

def train_with_checkpoints():
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
    )

    total_steps = 0
    for ckpt in CHECKPOINT_STEPS:
        steps_to_train = ckpt - total_steps
        print(f"\n📌 Training from {total_steps} → {ckpt} total steps...\n")

        model.learn(total_timesteps=steps_to_train)
        total_steps = ckpt

        ckpt_path = os.path.join(SAVE_DIR, f"ppo_cartpole_{ckpt}.zip")
        model.save(ckpt_path)
        print(f"💾 Saved checkpoint to {ckpt_path}")

    print("\n🎉 Finished generating all PPO checkpoints.")

if __name__ == "__main__":
    train_with_checkpoints()
