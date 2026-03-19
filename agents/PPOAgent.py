from agents.Agent import Agent
from agents.FeedForwardNN import FeedForwardNN

class PPO(Agent):
    def __init__(self, env, name):
        super().__init__()
        self.name = name # needed to access the correct obs/action spaces from the environment
        self.env = env
        self.obs_dim = env.observation_spaces[name]["observation"].shape[0]
        self.act_dim = env.action_spaces[name].shape[0]
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # TODO: dynamic opponent selection
        self.opponent = FeedForwardNN(self.obs_dim, self.act_dim)
        self.agents = {
            "player_0": self.actor,
            "player_1": self.opponent
        }

        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600

    def learn(self, total_timesteps):
        current_timestep = 0

        while current_timestep < total_timesteps:
            pass

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lens = []

        obs = self.env.reset()

        current_timestep = 0
        for agent in self.env.agent_iter():
            current_timestep += 1
            if current_timestep > self.max_timesteps_per_episode:
                break

            observation, reward, termination, truncation, info = self.env.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = agents[agent].act(observation)

                # (For now) only collect training data from the actor agent
                # TODO: also collect training data from the opponent agent
                if agent == "player_0":
                    batch_obs.append(obs)
                    batch_rewards.append(reward)
                    # TODO: batch_log_probs?

                action = env.decode_action(action)

            env.step(action)


