from collections import deque
from typing import Deque

import torch
from torch import distributions, Tensor
from torch.optim import Adam
import numpy as np

from agents.Agent import Agent
from agents.FeedForwardNN import FeedForwardNN
from agents.HeuristicAgentMKI import HeuristicAgentMKI
from agents.HeuristicAgentMKII import HeuristicAgentMKII
from agents.PPOAgent import PPOAgent
from agents.RandomAgent import RandomAgent
from fluxx.AgentBattler import AgentBattler
from fluxx.MetricsTracker import MetricsTracker
from fluxx.game.FluxxEnums import GameConfig

"""
Metrics recorded:
- average game turn count (per 16,000 timesteps)
- trainee WR against HeuristicAgentMKII and RandomAgent (100 games each per 16,000 timesteps)

"""

class OpponentPool:
    def __init__(self, game_config: GameConfig, player_number: int, pool_size: int = 20):
        self.pool: Deque[Agent] = deque()
        self.game_config: GameConfig = game_config
        self.player_number: int = player_number
        self.pool_size: int = pool_size

    def add_agent(self, agent: Agent):
        self.pool.append(agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def add_ppo(self, policy_network: FeedForwardNN):
        new_agent = PPOAgent(self.game_config, self.player_number)
        new_agent.policy_network = policy_network

        self.pool.append(new_agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def sample(self):
        return np.random.choice(self.pool)


class PPO:
    def __init__(self, env, agent_names: list[str]):
        super().__init__()
        self._init_hyperparameters()

        self.agent_names = agent_names
        self.env = env

        self.actor = PPOAgent(env.game.game_config, 0)
        self.actor_optim = Adam(self.actor.policy_network.parameters(), lr=self.lr)

        self.critic = FeedForwardNN(self.actor.observation_space["observation"].shape[0], 1)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.agents = {
            "player_0": self.actor,
            "player_1": None
        }
        self.opponent_pool = OpponentPool(env.game.game_config, 1)
        base_opponent = PPOAgent(env.game.game_config, 1)
        self.opponent_pool.add_agent(base_opponent)

        # TODO: tensorboard integration
        self.tracker = MetricsTracker(None, False)
        self.tracker.register_flat_statistic("games_vs_heuristicagentmkii/wins_out_of_100")
        self.tracker.register_flat_statistic("games_vs_heuristicagentmkii/average_game_length")
        self.tracker.register_flat_statistic("games_vs_randomagent/wins_out_of_100")
        self.tracker.register_flat_statistic("games_vs_randomagent/average_game_length")

        self.agent_battler = AgentBattler(env)


    def _init_hyperparameters(self):
        # base hyperparameters
        self.max_timesteps_per_episode = 1600
        self.games_per_batch = 128
        self.gamma = 0.999
        self.updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 3e-4

        # extra hyperparameters


    def learn(self, total_timesteps):
        current_timestep = 0

        # timestep..?
        while current_timestep < total_timesteps:
            print(current_timestep)
            print(self.env.game.player_turn)

            current_opponent = self.opponent_pool.sample()
            #self.agents["player_1"] = current_opponent
            #testing something
            self.agents["player_1"] = HeuristicAgentMKII(self.env.game.game_config, 1)

            batch_obs, batch_acts, batch_action_masks, batch_log_probs, batch_rewards_to_go, batch_lens = self.rollout()
            print("rollout complete")

            V, _= self.evaluate(batch_obs, batch_acts, batch_action_masks)
            advantages = batch_rewards_to_go - V.detach()

            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for _ in range(self.updates_per_iteration):
                V, current_log_probs = self.evaluate(batch_obs, batch_acts, batch_action_masks)
                ratios = torch.exp(current_log_probs - batch_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = torch.nn.MSELoss()(V, batch_rewards_to_go)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

            current_timestep += np.sum(batch_lens)

            # check wr every batch
            vs_heuristicagent = self.agent_battler.run_games([self.actor, HeuristicAgentMKII(self.env.game.game_config, 1)], 100, 10000)
            vs_randomagent = self.agent_battler.run_games([self.actor, RandomAgent(self.env.game.game_config, 1)], 100, 10000)
            self.tracker.record("games_vs_heuristicagentmkii/wins_out_of_100", vs_heuristicagent["player_wins"]["player_0"])
            self.tracker.record("games_vs_heuristicagentmkii/average_game_length", vs_heuristicagent["average_game_length"])
            self.tracker.record("games_vs_randomagent/wins_out_of_100", vs_randomagent["player_wins"]["player_0"])
            self.tracker.record("games_vs_randomagent/average_game_length", vs_randomagent["average_game_length"])
            self.tracker.flush(current_timestep)

            # Add a new enemy to the pool every batch
            self.opponent_pool.add_ppo(self.actor.policy_network)


        torch.save(self.actor.policy_network.state_dict(), "actor.pt")


    def evaluate(self, batch_obs, batch_acts, batch_action_masks):
        V = self.critic(batch_obs).squeeze()

        logits = self.actor.policy_network(batch_obs)
        logits[~batch_action_masks] = -float("inf")

        distribution = torch.distributions.Categorical(logits=logits)

        log_probs = distribution.log_prob(batch_acts)

        return V, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_action_masks = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = []

        games_played = 0
        while games_played < self.games_per_batch:
            obs = self.env.reset()
            episode_rewards = []

            current_timestep = 0
            for agent in self.env.agent_iter():
                current_timestep += 1
                # Ideally only complete games, but terminate games early in case some pathological loop occurs
                if current_timestep > self.max_timesteps_per_episode:
                    break

                observation, reward, termination, truncation, info = self.env.last()

                # .last() gets rewards from the previous action, whose entry in ep_rewards will have been appended in the previous iteration of the loop (see below)
                if agent == "player_0" and len(episode_rewards) > 0:
                    episode_rewards[-1] += reward

                if termination or truncation:
                    action = None
                else:
                    action, log_probs, observation = self.agents[agent].act(observation)

                    # (For now) only collect training data from the actor agent
                    # TODO: also collect training data from the opponent agent
                    if agent == "player_0":
                        batch_acts.append(action) # consider whether to decode before this
                        batch_action_masks.append(observation["action_mask"])
                        batch_obs.append(observation["observation"])
                        episode_rewards.append(reward)
                        batch_log_probs.append(log_probs)

                    action = self.env.decode_action(action)

                self.env.step(action)

            games_played += 1

            batch_lens.append(current_timestep + 1)
            batch_rewards.append(episode_rewards)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.from_numpy(np.stack(batch_obs)).float()
        batch_acts = torch.from_numpy(np.stack(batch_acts)).long()
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_action_masks = torch.tensor(batch_action_masks, dtype=torch.bool)
        # ALG STEP #4
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # Return the batch data
        return batch_obs, batch_acts, batch_action_masks, batch_log_probs, batch_rewards_to_go, batch_lens

    def compute_rewards_to_go(self, batch_rewards):
        """
        Compute rewards-to-go for each episode in the batch.
        For each episode, computes the rtg at each timestep
        """
        batch_rewards_to_go = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rewards in batch_rewards:
            discounted_reward = 0  # The discounted reward so far
            rewards_to_go = []
            for rew in reversed(ep_rewards):
                discounted_reward = rew + discounted_reward * self.gamma
                rewards_to_go.append(discounted_reward)
            rewards_to_go.reverse()
            batch_rewards_to_go.extend(rewards_to_go)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rewards_to_go, dtype=torch.float)
        return batch_rtgs