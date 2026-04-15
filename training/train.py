import numpy as np
import torch
from utils.logger import Logger

from env.kuka_env import KukaEnv
from agent.sac import SAC
from agent.replay_buffer import ReplayBuffer

logger = Logger()
def train():

    # -------- Setup --------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = KukaEnv(render=True)

    state_dim = 20
    action_dim = 7

    agent = SAC(state_dim, action_dim, device=device)
    buffer = ReplayBuffer(state_dim, action_dim)

    episodes = 500
    max_steps = 200
    batch_size = 256
    start_steps = 2000  # random steps before training

    total_steps = 0
    success_history = []
    # -------- Training Loop --------
    for episode in range(episodes):

        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):

            # -------- Action --------
            if total_steps < start_steps:
                action = np.random.uniform(-1, 1, size=action_dim)
            else:
                action = agent.select_action(state)

            # -------- Environment Step --------
            next_state, reward, done, info = env.step(action)

            # -------- Store in Buffer --------
            buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # -------- Train Agent --------
            if buffer.size > batch_size:
                losses = agent.update(buffer, batch_size)

            if done:
                break

        success = info["distance"] < 0.07
        success_history.append(success)  
        if len(success_history) >= 20:
            recent_success = success_history[-20:]
            success_rate = sum(recent_success) / 20

            env.update_difficulty(success_rate)

        if len(success_history) >= 20:
            success_rate = sum(success_history[-20:]) / 20
            print(f"Success rate (last 20): {success_rate:.2f}")
            env.update_difficulty(success_rate)
        else:
            success_rate = 0.0

        logger.log(
            episode,
            info["distance"],
            success,
            success_rate,
            env.difficulty
        )

        print(f"Ep {episode} | Dist: {info['distance']:.4f} | Success: {success} | Diff: {env.difficulty}") 
        

    env.close()
    logger.save()


if __name__ == "__main__":
    train()