import os
import copy
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque

import torch
from torch.optim import Adam
import numpy as np

from src.agents.Agent import Agent
from src.agents.training.TrainingEnums import LearningCheckpoint
from src.neural_networks.FeedForwardNN import FeedForwardNN
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.RandomAgent import RandomAgent
from src.env.AgentBattler import AgentBattler
from src.env.MetricsTracker import MetricsTracker
from src.game.FluxxEnums import GameConfig

# TODO: update this
"""
Metrics recorded:
- average game turn count (per 16,000 timesteps)
- trainee WR against HeuristicAgentMKII and RandomAgent (100 games each per 16,000 timesteps)

"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_default_device() -> torch.device:
    """
    Pick the best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OpponentPool:
    def __init__(self, game_config: GameConfig, player_number: int, pool_size: int = 20,
                 device: torch.device = torch.device("cpu")):
        self.pool: Deque[Agent] = deque()
        self.game_config: GameConfig = game_config
        self.player_number: int = player_number
        self.pool_size: int = pool_size
        self.device: torch.device = device

    def add_agent(self, agent: Agent):
        # Best-effort move any nn.Module attributes on the agent to the pool's device.
        policy = getattr(agent, "policy_network", None)
        if isinstance(policy, torch.nn.Module):
            policy.to(self.device)
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
        self.pool.append(agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def add_ppo(self, policy_network: FeedForwardNN):
        new_agent = PPOAgent(self.game_config, self.player_number)
        cpu_copy = copy.deepcopy(policy_network).to("cpu")
        new_agent.policy_network = cpu_copy.to(self.device)
        new_agent.policy_network.eval()
        for p in new_agent.policy_network.parameters():
            p.requires_grad_(False)

        self.pool.append(new_agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def sample(self):
        return np.random.choice(self.pool)

    def get_oldest(self):
        return self.pool[0]


class PPO:
    def __init__(self, env, agent_names: list[str], device: torch.device = None, from_checkpoint: LearningCheckpoint = None):
        super().__init__()
        self._init_hyperparameters()

        self.device: torch.device = device if device is not None else get_default_device()
        print(f"PPO using device: {self.device}")

        self.agent_names = agent_names
        self.env = env

        self.actor = PPOAgent(env.game.game_config, 0)
        self.actor.policy_network.to(self.device)
        self.actor_optim = Adam(self.actor.policy_network.parameters(), lr=self.lr)

        self.critic = FeedForwardNN(self.actor.observation_space["observation"].shape[0], 1).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.agents = {
            "player_0": self.actor,
            "player_1": None
        }
        self.opponent_pool = OpponentPool(env.game.game_config, 1, device=self.device)
        base_opponent = PPOAgent(env.game.game_config, 1)
        self.opponent_pool.add_agent(base_opponent)

        # TODO: tensorboard integration
        self.run_name = f"ppo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.tracker = MetricsTracker(f"{self.run_name}", True, 100, {
            "max_timesteps_per_episode": self.max_timesteps_per_episode,
            "games_per_batch": self.games_per_batch,
            "gamma": self.gamma,
            "epoch_count": self.epoch_count,
            "minibatch_size": self.minibatch_size,
            "kl_limit": self.kl_limit,
            "clip": self.clip,
            "lr": self.lr,
            "gae_lambda": self.gae_lambda,
            "device": str(self.device),
        })
        self.tracker.register_flat_statistic("games_vs_heuristicagentmkii/winrate")
        self.tracker.register_flat_statistic("games_vs_heuristicagentmkii/average_game_length")
        self.tracker.register_flat_statistic("games_vs_randomagent/winrate")
        self.tracker.register_flat_statistic("games_vs_randomagent/average_game_length")
        self.tracker.register_flat_statistic("games_vs_past_version/winrate")
        self.tracker.register_flat_statistic("games_vs_past_version/average_game_length")

        self.agent_battler = AgentBattler(env)

        self.global_timestep = 0
        self.model_checkpoints_taken = 0

        # save_current_model needs the file to already exist
        if from_checkpoint is not None:
            self.run_name = from_checkpoint.run_name
            self.global_timestep = from_checkpoint.global_timestep
            self.model_checkpoints_taken = from_checkpoint.model_checkpoints_taken
            state_dict = torch.load(f"{PROJECT_ROOT}/experiments/{self.run_name}/models/model_{self.global_timestep}.pt", map_location="cpu", weights_only=True)
            self.actor.policy_network.load_state_dict(state_dict)
            print(f"Loaded model from checkpoint {self.run_name}")
        else:
            os.makedirs(f"{PROJECT_ROOT}/experiments/{self.run_name}/models")


    def _init_hyperparameters(self):
        # base hyperparameters
        self.max_timesteps_per_episode = 1600
        self.games_per_batch = 128
        self.gamma = 0.99
        self.epoch_count = 4
        self.minibatch_size = 64
        self.kl_limit = 0.02
        self.clip = 0.2
        self.lr = 1e-4

        # extra hyperparameters
        self.gae_lambda = 0.95
        self.model_eval_count = 200

    def learn(self, total_timesteps):
        eval_every = total_timesteps // self.model_eval_count
        evals_performed = 0

        # timestep..?
        while self.global_timestep < total_timesteps:
            print(f"current timestep: {self.global_timestep}")

            current_opponent = self.opponent_pool.sample()
            self.agents["player_1"] = current_opponent

            batch_obs, batch_acts, batch_action_masks, batch_log_probs, batch_advantages, batch_returns = self.rollout()
            print("rollout complete")

            V, _, entropy = self.evaluate(batch_obs, batch_acts, batch_action_masks)
            self.tracker.record("actor/entropy", entropy.mean().item())

            # explained variance
            with torch.no_grad():
                explained_var = 1 - (batch_returns - V).var() / (batch_returns.var() + 1e-8)
            self.tracker.record("actor/explained_variance", explained_var.item())
            self.tracker.flush(self.global_timestep)

            # advantage normalization
            advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)

            # Per-update metrics
            kl_divergences = []
            clip_fractions = []
            actor_losses, critic_losses = [], []

            # setup minibatching
            batch_size = batch_obs.shape[0]
            indices = np.arange(batch_size)

            early_stopped = False
            for epoch in range(self.epoch_count):
                if early_stopped:
                    break

                np.random.shuffle(indices)

                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = indices[start:end]
                    # Use a torch long tensor on device for indexing to avoid host<->device syncs
                    idx_tensor = torch.as_tensor(minibatch_indices, dtype=torch.long, device=self.device)

                    mb_obs = batch_obs.index_select(0, idx_tensor)
                    mb_acts = batch_acts.index_select(0, idx_tensor)
                    mb_masks = batch_action_masks.index_select(0, idx_tensor)
                    mb_old_log_probs = batch_log_probs.index_select(0, idx_tensor)
                    mb_advantages = advantages.index_select(0, idx_tensor)
                    mb_returns = batch_returns.index_select(0, idx_tensor)

                    # Forward pass on the minibatch
                    V_mb, current_log_probs, _ = self.evaluate(mb_obs, mb_acts, mb_masks)

                    # PPO actor loss
                    ratios = torch.exp(current_log_probs - mb_old_log_probs)
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mb_advantages
                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    # Critic loss
                    critic_loss = torch.nn.MSELoss()(V_mb, mb_returns)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    # Diagnostics
                    with torch.no_grad():
                        log_ratio = current_log_probs - mb_old_log_probs
                        approx_kl = ((ratios - 1) - log_ratio).mean()
                        clip_frac = ((ratios - 1).abs() > self.clip).float().mean()

                    kl_divergences.append(approx_kl.item())
                    clip_fractions.append(clip_frac.item())
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

                # if the kl of this branch exceeds the limt, stop training off the minibatches
                if self.kl_limit is not None:
                    # KLs from this epoch only
                    epoch_kls = kl_divergences[-(batch_size // self.minibatch_size):]
                    if np.mean(epoch_kls) > self.kl_limit:
                        print(f"Early stopping at epoch {epoch + 1}: KL {np.mean(epoch_kls):.4f} > {self.kl_limit}")
                        early_stopped = True

            self.tracker.record("kl_divergence", np.mean(kl_divergences))
            self.tracker.record("clip_fraction", np.mean(clip_fractions))

            # check wr every batch
            if self.global_timestep // eval_every >= evals_performed + 1:
                evals_performed += 1
                vs_heuristicagent = self.agent_battler.run_games([self.actor, HeuristicAgentMKII(self.env.game.game_config, 1)], 50, 10000)
                vs_randomagent = self.agent_battler.run_games([self.actor, RandomAgent(self.env.game.game_config, 1)], 50, 10000)
                vs_past_version = self.agent_battler.run_games([self.actor, self.opponent_pool.get_oldest()], 50, 10000)
                self.tracker.record("games_vs_heuristicagentmkii/winrate", vs_heuristicagent["player_wins"]["player_0"]/50)
                self.tracker.record("games_vs_heuristicagentmkii/average_game_length", vs_heuristicagent["average_game_length"])
                self.tracker.record("games_vs_randomagent/winrate", vs_randomagent["player_wins"]["player_0"]/50)
                self.tracker.record("games_vs_randomagent/average_game_length", vs_randomagent["average_game_length"])
                self.tracker.record("games_vs_past_version/winrate", vs_past_version["player_wins"]["player_0"]/50)
                self.tracker.record("games_vs_past_version/average_game_length", vs_past_version["average_game_length"])
                self.tracker.flush(self.global_timestep)

            # Add a new enemy to the pool every batch
            self.opponent_pool.add_ppo(self.actor.policy_network)

            # save the policy every time a threshold is crossed
            if self.global_timestep // 500000 >= self.model_checkpoints_taken + 1:
                self.model_checkpoints_taken += 1
                self.save_current_model(f"model_{self.global_timestep}")

        self.save_current_model("final_model")

        # final evaluation
        vs_heuristicagent = self.agent_battler.run_games([self.actor, HeuristicAgentMKII(self.env.game.game_config, 1)],100, 10000)
        vs_randomagent = self.agent_battler.run_games([self.actor, RandomAgent(self.env.game.game_config, 1)], 100,10000)
        self.tracker.close({
            "final_wr_vs_heuristicagent": vs_heuristicagent["player_wins"]["player_0"]/100,
            "final_wr_vs_randomagent": vs_randomagent["player_wins"]["player_0"]/100,
        })

    def save_current_model(self, filename):
        path = f"{PROJECT_ROOT}/experiments/{self.run_name}/models/{filename}.pt"
        critic_path = f"{PROJECT_ROOT}/experiments/{self.run_name}/models/{filename}_critic.pt"
        # Save a CPU copy of the state_dict so checkpoints are portable across devices.
        state_dict = {k: v.detach().cpu() for k, v in self.actor.policy_network.state_dict().items()}
        critic_state_dict = {k: v.detach().cpu() for k, v in self.critic.policy_network.state_dict().items()}
        torch.save(state_dict, path)
        torch.save(critic_state_dict, critic_path)
        print(f"Model saved to {path}")

    def evaluate(self, batch_obs, batch_acts, batch_action_masks):
        # Inputs are expected to be on self.device already (set up in rollout / minibatching).
        V = self.critic(batch_obs).squeeze()

        logits = self.actor.policy_network(batch_obs)
        logits = logits.masked_fill(~batch_action_masks, float("-inf"))

        distribution = torch.distributions.Categorical(logits=logits)

        log_probs = distribution.log_prob(batch_acts)
        entropy = distribution.entropy()

        return V, log_probs, entropy

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_action_masks = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_dones = []

        games_played = 0
        game_lengths = []
        while games_played < self.games_per_batch:
            obs = self.env.reset()
            episode_rewards = []
            episode_values = []
            episode_dones = []

            current_timestep = 0
            for agent in self.env.agent_iter():
                # Ideally only complete games, but terminate games early in case some pathological loop occurs
                if current_timestep > self.max_timesteps_per_episode:
                    break

                observation, reward, termination, truncation, info = self.env.last()

                if agent == "player_0":
                    # only the timesteps of the agent being trained matter
                    current_timestep += 1
                    self.global_timestep += 1

                    # .last() gets rewards from the previous action, whose entry in ep_rewards will have been appended in the previous iteration of the loop (see below)
                    if len(episode_rewards) > 0:
                        episode_rewards[-1] += reward

                if termination or truncation:
                    action = None
                    if len(episode_dones) > 0:
                        episode_dones[-1] = True
                else:
                    action, log_probs, observation = self.agents[agent].act(observation)

                    # (For now) only collect training data from the actor agent
                    # TODO: also collect training data from the opponent agent
                    if agent == "player_0":
                        # Normalize `action` to a plain python int for env.step and storage.
                        if isinstance(action, torch.Tensor):
                            action_for_env = action.detach().cpu().item()
                        else:
                            action_for_env = int(action)

                        # Normalize log_probs to a python float for later stacking.
                        if isinstance(log_probs, torch.Tensor):
                            log_prob_val = log_probs.detach().cpu().item()
                        else:
                            log_prob_val = float(log_probs)

                        batch_acts.append(action_for_env)  # consider whether to decode before this
                        batch_action_masks.append(np.asarray(observation["action_mask"], dtype=bool))
                        batch_obs.append(np.asarray(observation["observation"]))
                        episode_rewards.append(reward)
                        batch_log_probs.append(log_prob_val)

                        # Compute value for this state (on device)
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(
                                np.asarray(observation["observation"])
                            ).float().unsqueeze(0).to(self.device)
                            value = self.critic(obs_tensor).squeeze().item()
                        episode_values.append(value)
                        episode_dones.append(False)  # will be flipped to True on terminal turn

                        action = action_for_env
                    else:
                        # Opponent action: still needs to be a plain int for env.step.
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().item()

                    action = self.env.decode_action(action)

                self.env.step(action)

            games_played += 1
            game_lengths.append(self.env.game.turn_count)

            batch_rewards.append(episode_rewards)
            batch_values.append(episode_values)
            batch_dones.append(episode_dones)

        # Reshape data as tensors on self.device before returning.
        batch_obs = torch.from_numpy(np.stack(batch_obs)).float().to(self.device)
        batch_acts = torch.from_numpy(np.stack(batch_acts)).long().to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_action_masks = torch.from_numpy(np.stack(batch_action_masks)).bool().to(self.device)

        # compute advantages and returns (done in numpy, then moved to device)
        batch_advantages, batch_returns = self.compute_gae(batch_rewards, batch_values, batch_dones)

        # end of rollout metric tracking
        self.tracker.record("rollout/average_game_length", np.mean(game_lengths))
        wins = sum(1 for ep in batch_rewards if sum(ep) > 0)
        self.tracker.record("rollout/win_rate_vs_pool", wins / games_played)
        self.tracker.flush(self.global_timestep)

        # Return the batch data
        return batch_obs, batch_acts, batch_action_masks, batch_log_probs, batch_advantages, batch_returns

    def compute_gae(self, rewards, values, dones):
        """
        Compute GAE for a batch of experiences. Kept on CPU/numpy (small, sequential work)
        and moved to self.device at the end.
        """
        all_advantages = []
        all_returns = []

        for ep_rewards, ep_values, ep_dones in zip(rewards, values, dones):
            T = len(ep_rewards)
            advantages = np.zeros(T)
            last_advantage = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                    next_non_terminal = 0.0
                else:
                    next_value = ep_values[t + 1]
                    next_non_terminal = 1.0 - float(ep_dones[t + 1])

                delta = ep_rewards[t] + self.gamma * next_value * next_non_terminal - ep_values[t]
                advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

            returns = advantages + np.array(ep_values, dtype=np.float32)
            all_advantages.extend(advantages.tolist())
            all_returns.extend(returns.tolist())

        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float, device=self.device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float, device=self.device)

        return advantages_tensor, returns_tensor