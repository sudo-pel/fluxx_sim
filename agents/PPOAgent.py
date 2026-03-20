import torch
from torch import distributions
from torch.optim import Adam
import numpy as np

from agents.Agent import Agent
from agents.FeedForwardNN import FeedForwardNN

class PPO:
    def __init__(self, env, agent_names: list[str]):
        super().__init__()
        self._init_hyperparameters()

        self.agent_names = agent_names # needed to access the correct obs/action spaces from the environment
        self.env = env
        self.obs_dim = env.observation_spaces[agent_names[0]]["observation"].shape[0]
        self.act_dim = env.observation_spaces[agent_names[0]]["action_mask"].shape[0]
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # TODO: dynamic opponent selection
        self.opponent = FeedForwardNN(self.obs_dim, self.act_dim)
        self.agents = {
            "player_0": self.actor,
            "player_1": self.opponent
        }

        # I think that these are unneeded
        #self.timesteps_per_batch = 4800

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 1600
        self.games_per_batch = 16
        self.gamma = 0.95
        self.updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def learn(self, total_timesteps):
        current_timestep = 0

        # timestep..?
        while current_timestep < total_timesteps:
            print(current_timestep)
            batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens = self.rollout()

            V, _= self.evaluate(batch_obs)
            advantages = batch_rewards_to_go - V.detach()

            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for _ in range(self.updates_per_iteration):
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
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


    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        batch_observations = batch_obs["observation"]
        batch_action_masks = batch_obs["action_mask"]

        logits = self.actor(batch_observations)
        logits[~batch_action_masks] = -float("inf")

        distribution = torch.distributions.Categorical(logits=logits)

        action = distribution.sample()
        log_probs = distribution.log_prob(action)

        return V, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
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

                if termination or truncation:
                    action = None
                    break
                else:
                    # this is where you would insert your policy
                    action, log_probs = self.agents[agent].act(observation)

                    # (For now) only collect training data from the actor agent
                    # TODO: also collect training data from the opponent agent
                    if agent == "player_0":
                        batch_acts.append(action) # consider whether to decode before this
                        batch_obs.append(obs)
                        batch_rewards.append(reward)
                        batch_log_probs.append(log_probs)

                    action = self.env.decode_action(action)

                self.env.step(action)

            batch_lens.append(current_timestep + 1)
            batch_rewards.append(episode_rewards)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens

    def compute_rewards_to_go(self, batch_rewards):
        """
        Compute rewards-to-go for each episode in the batch.
        For each episode, computes the rtg at each timestep
        """
        batch_rewards_to_go = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rewards in batch_rewards_to_go:
            discounted_reward = 0  # The discounted reward so far
            rewards_to_go = []
            for rew in reversed(ep_rewards):
                discounted_reward = rew + discounted_reward * self.gamma
                rewards_to_go.append(discounted_reward)
            rewards_to_go.reverse()
            batch_rewards_to_go.extend(rewards_to_go)
        batch_rewards_to_go.reverse()

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rewards_to_go, dtype=torch.float)
        return batch_rtgs