import copy
from collections import deque
from pathlib import Path
from typing import Deque

import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from src.agents.Agent import Agent
from src.neural_networks.FluxxActorNetwork import FluxxStateEncoder
from src.training import training_utils
from src.training.TrainingEnums import LearningCheckpoint
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.env.AgentBattler import AgentBattler
from src.env.MetricsTracker import MetricsTracker
from src.game.FluxxEnums import GameConfig, GameState
from src.training.TrainingEnums import BufferEntry
from src.training.training_utils import GameplanExtendedSortingOptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CRITIC_CHUNK_SIZE = 256

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class FluxxCriticNetwork(nn.Module):
    """
    FluxxStateEncoder here has separate parameters from the actor.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = FluxxStateEncoder()
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state_vec = self.encoder(obs)
        return self.head(state_vec)

class OpponentPool:
    def __init__(self, game_config: GameConfig, player_number: int, pool_size: int = 20,
                 device: torch.device = torch.device("cpu"), seed: np.random.SeedSequence = None):
        self.pool: Deque[Agent] = deque()
        self.game_config: GameConfig = game_config
        self.player_number: int = player_number
        self.pool_size: int = pool_size
        self.device: torch.device = device
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def add_agent(self, agent: Agent):
        policy = getattr(agent, "policy_network", None)
        if isinstance(policy, torch.nn.Module):
            policy.to(self.device)
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
        self.pool.append(agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def add_ppo(self, policy_network: torch.nn.Module):
        # consider changing type hint to FluxxActorNetwork
        new_agent = PPOAgentGeneralized(self.game_config, self.player_number)
        cpu_copy = copy.deepcopy(policy_network).to("cpu")
        new_agent.policy_network = cpu_copy.to(self.device)
        new_agent.policy_network.eval()
        for p in new_agent.policy_network.parameters():
            p.requires_grad_(False)

        self.pool.append(new_agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def sample(self):
        return self.pool[self.rng.integers(len(self.pool))]

    def get_oldest(self):
        return self.pool[0]

class PPOGeneralizedRewardShaped:
    def __init__(self, env, agent_names: list[str], run_name,
                 device: torch.device = None,
                 from_checkpoint: LearningCheckpoint = None,
                 seed: np.random.SeedSequence = None):
        super().__init__()
        self.init_hyperparameters()

        self.device: torch.device = device if device is not None else get_default_device()
        print(f"PPO using device: {self.device}")

        self.agent_names = agent_names
        self.env = env

        ss_opponent_pool, ss_ppo = seed.spawn(2)
        self.rng = np.random.default_rng(ss_ppo)

        self.actor = PPOAgentGeneralized(env.game.game_config, 0)
        self.actor.policy_network.to(self.device)
        self.actor_optim = Adam(self.actor.policy_network.parameters(), lr=self.lr)

        self.critic = FluxxCriticNetwork().to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.agents = {
            "player_0": self.actor,
            "player_1": None
        }
        self.opponent_pool = OpponentPool(env.game.game_config, 1, device=self.device, seed=ss_opponent_pool)
        base_opponent = PPOAgentGeneralized(env.game.game_config, 1)
        self.opponent_pool.add_agent(base_opponent)

        self.run_name = run_name
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

        if from_checkpoint is not None:
            self.run_name = from_checkpoint.run_name
            self.global_timestep = from_checkpoint.global_timestep
            self.model_checkpoints_taken = from_checkpoint.model_checkpoints_taken
            state_dict = torch.load(
                f"{PROJECT_ROOT}/experiments/{self.run_name}/models/model_{self.global_timestep}.pt",
                map_location="cpu", weights_only=True,
            )
            self.actor.policy_network.load_state_dict(state_dict)
            print(f"Loaded model from checkpoint {self.run_name}")
            # TODO: critic checkpoint loading

        # Reward shaping
        self.phi_previous: float = 0.0

    def init_hyperparameters(self):
        self.max_timesteps_per_episode = 3200
        self.games_per_batch = 128
        self.gamma = 0.99
        self.epoch_count = 4
        self.minibatch_size = 64
        self.kl_limit = 0.02
        self.clip = 0.2
        self.lr = 1e-4
        self.entropy_coefficient = 0.01

        self.gae_lambda = 0.95
        self.model_eval_count = 200

    def learn(self, total_timesteps):
        eval_every = total_timesteps // self.model_eval_count
        evals_performed = 0

        while self.global_timestep < total_timesteps:
            print(f"current timestep: {self.global_timestep}")

            current_opponent = self.opponent_pool.sample()
            self.agents["player_1"] = current_opponent

            batch_obs, batch_acts, batch_log_probs, batch_advantages, batch_returns = self.rollout()
            print("rollout complete")

            V, entropy = self.batched_evaluate_diagnostics(batch_obs, batch_acts)
            self.tracker.record("actor/entropy", entropy.mean().item())

            with torch.no_grad():
                explained_var = 1 - (batch_returns - V).var() / (batch_returns.var() + 1e-8)
            self.tracker.record("actor/explained_variance", explained_var.item())
            self.tracker.flush(self.global_timestep)

            advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)

            kl_divergences = []
            clip_fractions = []
            actor_losses, critic_losses = [], []

            batch_size = len(batch_obs)
            indices = np.arange(batch_size)

            early_stopped = False
            for epoch in range(self.epoch_count):
                if early_stopped:
                    break

                self.rng.shuffle(indices)

                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = indices[start:end]
                    idx_tensor = torch.as_tensor(minibatch_indices, dtype=torch.long, device=self.device)

                    mb_entries = [batch_obs[i] for i in minibatch_indices]
                    mb_obs = self.actor.collate(mb_entries, self.device)

                    mb_acts = batch_acts.index_select(0, idx_tensor)
                    mb_old_log_probs = batch_log_probs.index_select(0, idx_tensor)
                    mb_advantages = advantages.index_select(0, idx_tensor)
                    mb_returns = batch_returns.index_select(0, idx_tensor)

                    V_mb, current_log_probs, mb_entropy = self.evaluate(mb_obs, mb_acts)

                    ratios = torch.exp(current_log_probs - mb_old_log_probs)
                    surr1 = ratios * mb_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mb_advantages
                    policy_loss = (-torch.min(surr1, surr2)).mean()
                    entropy_bonus = mb_entropy.mean()
                    actor_loss = policy_loss - self.entropy_coefficient * entropy_bonus

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=False)
                    self.actor_optim.step()

                    critic_loss = torch.nn.MSELoss()(V_mb, mb_returns)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                    with torch.no_grad():
                        log_ratio = current_log_probs - mb_old_log_probs
                        approx_kl = ((ratios - 1) - log_ratio).mean()
                        clip_frac = ((ratios - 1).abs() > self.clip).float().mean()

                    kl_divergences.append(approx_kl.item())
                    clip_fractions.append(clip_frac.item())
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

                if self.kl_limit is not None:
                    epoch_kls = kl_divergences[-(batch_size // self.minibatch_size):]
                    if np.mean(epoch_kls) > self.kl_limit:
                        print(f"Early stopping at epoch {epoch + 1}: KL {np.mean(epoch_kls):.4f} > {self.kl_limit}")
                        early_stopped = True

            self.tracker.record("kl_divergence", np.mean(kl_divergences))
            self.tracker.record("clip_fraction", np.mean(clip_fractions))
            self.tracker.record("actor/loss", np.mean(actor_losses))
            self.tracker.record("critic/loss", np.mean(critic_losses))

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

            self.opponent_pool.add_ppo(self.actor.policy_network)

            if self.global_timestep // 500000 >= self.model_checkpoints_taken + 1:
                self.model_checkpoints_taken += 1
                self.save_current_model(f"model_{self.global_timestep}")

        self.save_current_model(f"final_model_{self.global_timestep}", final=True)

        vs_heuristicagent = self.agent_battler.run_games([self.actor, HeuristicAgentMKII(self.env.game.game_config, 1)], 100, 10000)
        vs_randomagent = self.agent_battler.run_games([self.actor, RandomAgent(self.env.game.game_config, 1)], 100, 10000)
        self.tracker.close({
            "final_wr_vs_heuristicagent": vs_heuristicagent["player_wins"]["player_0"]/100,
            "final_wr_vs_randomagent": vs_randomagent["player_wins"]["player_0"]/100,
        })

    def save_current_model(self, filename, final: bool = False):
        if not final:
            path = f"{PROJECT_ROOT}/experiments/{self.run_name}/models/{filename}.pt"
            critic_path = f"{PROJECT_ROOT}/experiments/{self.run_name}/models/{filename}_critic.pt"
        else:
            path = f"{PROJECT_ROOT}/experiments/{self.run_name}/final/{filename}.pt"
            critic_path = f"{PROJECT_ROOT}/experiments/{self.run_name}/final/{filename}_critic.pt"
        state_dict = {k: v.detach().cpu() for k, v in self.actor.policy_network.state_dict().items()}
        critic_state_dict = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}
        torch.save(state_dict, path)
        torch.save(critic_state_dict, critic_path)
        print(f"Model saved to {path}")


    def evaluate(self, batch_obs: dict, batch_acts):
        """
        Run actor + critic on a collated minibatch dict.
        """
        V = self.critic(batch_obs).squeeze()

        logits = self.actor.policy_network(batch_obs)
        logits = logits.masked_fill(~batch_obs["action_mask"], float("-inf"))

        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(batch_acts)
        entropy = distribution.entropy()

        return V, log_probs, entropy

    def batched_evaluate_diagnostics(
        self, entries: list[BufferEntry], acts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        V_chunks = []
        entropy_chunks = []
        N = len(entries)

        with torch.no_grad():
            for start in range(0, N, CRITIC_CHUNK_SIZE):
                end = min(start + CRITIC_CHUNK_SIZE, N)
                chunk_entries = entries[start:end]
                chunk_acts = acts[start:end]
                chunk_obs = self.actor.collate(chunk_entries, self.device)

                V_chunk = self.critic(chunk_obs).squeeze(-1)
                logits = self.actor.policy_network(chunk_obs)
                logits = logits.masked_fill(~chunk_obs["action_mask"], float("-inf"))
                dist = torch.distributions.Categorical(logits=logits)
                entropy_chunk = dist.entropy()

                V_chunks.append(V_chunk)
                entropy_chunks.append(entropy_chunk)

        return torch.cat(V_chunks, dim=0), torch.cat(entropy_chunks, dim=0)

    def score_game_state(self, game_state: GameState, player_number: int) -> float:
        """
        Custom reward shaping

        Base: reward calculation by getting all agent gameplans, and summing their scores w/
        DISCOUNT_RATE (to avoid over-rewarding "sparse sets of game states")

        Maximum individual gameplan score is 10.
        """
        DISCOUNT_RATE = 0.75
        relevant_cards = game_state.hands[player_number] + game_state.keepers[player_number] + game_state.goals
        gameplans, _ = training_utils.get_gameplans_from_cards(relevant_cards, game_state, player_number, sort_by=GameplanExtendedSortingOptions.GAMEPLAN_SCORE, reverse=True)
        total_reward = 0.0

        for i, gameplan in enumerate(gameplans):
            total_reward += gameplan.score * (DISCOUNT_RATE ** i)

        return total_reward

    def rollout(self):
        batch_obs: list[BufferEntry] = []
        batch_acts: list[int] = []
        batch_log_probs: list[float] = []
        batch_rewards: list[list[float]] = []
        batch_dones: list[list[bool]] = []

        # PBRS statistics
        episode_shaping_totals: list[float] = []
        episode_phi_initial: list[float] = []
        episode_env_reward_totals: list[float] = []

        batch_env_rewards: list[list[float]] = []
        batch_shaping_rewards: list[list[float]] = []

        # required for deferred GAE calculation
        per_episode_lengths: list[int] = []

        games_played = 0
        game_lengths = []

        while games_played < self.games_per_batch:
            self.env.reset()
            episode_rewards = []
            episode_dones = []
            episode_entry_count = 0 # for per_episode_lengths

            # PBRS stats
            current_episode_shaping_sum = 0.0
            current_episode_phi_initial: float | None = None
            current_episode_env_reward_sum = 0.0

            # PBRS (advantage decomposition into terminal vs. intermediate)
            episode_env_rewards: list[float] = []
            episode_shaping_rewards: list[float] = []

            current_timestep = 0
            for agent in self.env.agent_iter():
                if current_timestep > self.max_timesteps_per_episode:
                    print(
                        f"[rollout] truncating game: timestep {current_timestep}, "
                        f"game_turn={self.env.game.turn_count}, last_agent={agent}"
                    )
                    break

                game_state, reward, termination, truncation, info = self.env.last()
                reward *= 100

                env_reward_component = 0.0
                shaping_term = 0.0
                if agent == "player_0":
                    env_reward_component = reward
                    phi = self.score_game_state(game_state, 0)
                    if termination:
                        phi = 0
                    # TODO: phi no longer needs to be global (was originally only going to reset on termination)
                    shaping_term = self.gamma * phi - self.phi_previous
                    reward += shaping_term

                    current_episode_shaping_sum += shaping_term
                    current_episode_env_reward_sum += env_reward_component
                    if current_episode_phi_initial is None:
                        current_episode_phi_initial = phi

                    self.phi_previous = phi

                current_timestep += 1

                if agent == "player_0":
                    self.global_timestep += 1
                    if len(episode_rewards) > 0:
                        episode_rewards[-1] += reward
                        episode_env_rewards[-1] += env_reward_component
                        episode_shaping_rewards[-1] += shaping_term

                if termination or truncation:
                    action = None
                    if len(episode_dones) > 0:
                        episode_dones[-1] = True
                else:
                    action, log_probs, entry = self.agents[agent].act(game_state)

                    if agent == "player_0":
                        action_for_env = int(action)

                        if isinstance(log_probs, torch.Tensor):
                            log_prob_val = log_probs.detach().cpu().item()
                        else:
                            log_prob_val = float(log_probs)

                        batch_obs.append(entry)
                        batch_acts.append(action_for_env)
                        batch_log_probs.append(log_prob_val)

                        episode_rewards.append(reward)
                        episode_env_rewards.append(0.0)
                        episode_shaping_rewards.append(0.0)
                        episode_dones.append(False) # will be flipped to True on terminal turn
                        episode_entry_count += 1

                        action = action_for_env
                    else:
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().item()

                    action = self.env.decode_action(action)

                self.env.step(action)

            games_played += 1
            game_lengths.append(self.env.game.turn_count)

            # PBRS
            episode_shaping_totals.append(current_episode_shaping_sum)
            if current_episode_phi_initial is not None:
                episode_phi_initial.append(current_episode_phi_initial)
            episode_env_reward_totals.append(current_episode_env_reward_sum)

            batch_env_rewards.append(episode_env_rewards)
            batch_shaping_rewards.append(episode_shaping_rewards)

            batch_rewards.append(episode_rewards)
            batch_dones.append(episode_dones)
            per_episode_lengths.append(episode_entry_count)

        batch_values = self.compute_values_batched(batch_obs)
        per_episode_values = []
        cursor = 0
        for L in per_episode_lengths:
            per_episode_values.append(batch_values[cursor:cursor + L])
            cursor += L
        assert cursor == len(batch_values)

        batch_advantages, batch_returns = self.compute_gae(
            batch_rewards, per_episode_values, batch_dones,
        )

        # Decomposed advantages for diagnostics (uses zero values to isolate reward contribution)
        zero_values = [[0.0] * len(ep) for ep in batch_env_rewards]
        env_advantages, _ = self.compute_gae(
            batch_env_rewards, zero_values, batch_dones,
        )
        shaping_advantages, _ = self.compute_gae(
            batch_shaping_rewards, zero_values, batch_dones,
        )

        batch_acts_t = torch.tensor(batch_acts, dtype=torch.long, device=self.device)
        batch_log_probs_t = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)

        # metrics calculation
        self.tracker.record("rollout/average_game_length", np.mean(game_lengths))
        wins = sum(1 for ep in batch_rewards if sum(ep) > 0)
        self.tracker.record("rollout/win_rate_vs_pool", wins / games_played)
        self.tracker.flush(self.global_timestep)

        # PBRS metric 1: per-episode total shaping
        self.tracker.record("pbrs/episode_shaping_total_mean", np.mean(episode_shaping_totals))
        self.tracker.record("pbrs/episode_shaping_total_std", np.std(episode_shaping_totals))

        # Telescoping residual: should be ~0 if PBRS invariant holds (sum_F + phi_0 = phi_T = 0)
        if len(episode_phi_initial) == len(episode_shaping_totals):
            telescoping_residual = [
                s + p for s, p in zip(episode_shaping_totals, episode_phi_initial)
            ]
            self.tracker.record("pbrs/telescoping_residual_mean", np.mean(telescoping_residual))
            self.tracker.record("pbrs/telescoping_residual_max_abs", np.max(np.abs(telescoping_residual)))

        # PBRS metric 2: shaping vs env reward ratio
        env_abs = np.abs(episode_env_reward_totals)
        shaping_abs = np.abs(episode_shaping_totals)
        ratios = shaping_abs / (env_abs + 1e-8)
        self.tracker.record("pbrs/shaping_to_env_ratio_mean", np.mean(ratios))
        self.tracker.record("pbrs/shaping_to_env_ratio_median", np.median(ratios))

        # PBRS metric 3: advantage decomposition
        env_adv_mag = env_advantages.abs().mean().item()
        shaping_adv_mag = shaping_advantages.abs().mean().item()
        self.tracker.record("pbrs/advantage_env_magnitude", env_adv_mag)
        self.tracker.record("pbrs/advantage_shaping_magnitude", shaping_adv_mag)
        self.tracker.record(
            "pbrs/advantage_shaping_fraction",
            shaping_adv_mag / (env_adv_mag + shaping_adv_mag + 1e-8)
        )

        return batch_obs, batch_acts_t, batch_log_probs_t, batch_advantages, batch_returns

    # "degree of chunking" is limited to accommodate GPU sizes
    # TODO: consider changing this to be greedier
    def compute_values_batched(self, entries: list[BufferEntry]) -> list[float]:
        values: list[float] = []
        N = len(entries)
        with torch.no_grad():
            for start in range(0, N, CRITIC_CHUNK_SIZE):
                end = min(start + CRITIC_CHUNK_SIZE, N)
                chunk = entries[start:end]
                obs = self.actor.collate(chunk, self.device)
                v_chunk = self.critic(obs).squeeze(-1)
                values.extend(v_chunk.detach().cpu().tolist())
        return values

    def compute_gae(self, rewards, values, dones):
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