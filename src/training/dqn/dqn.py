import copy
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import SeedSequence
from torch.optim import Adam

from src.agents.Agent import Agent
from src.agents.DQNAgent import DQNAgent
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.RandomAgent import RandomAgent
from src.env.AgentBattler import AgentBattler
from src.env.MetricsTracker import MetricsTracker
from src.game.FluxxEnums import GameConfig
from src.neural_networks.DuelingFeedForwardNN import DuelingFeedForwardNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def freeze_eval(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


class DQNOpponentPool:
    """
    Agents store q-values networks and not policy networks, making this distinct from PPOs OpponentPool
    """
    def __init__(self, game_config: GameConfig, player_number: int, pool_size: int = 20,
                 device: torch.device = torch.device("cpu"), seed: np.random.SeedSequence = None):
        self.pool: Deque[Agent] = deque()
        self.game_config = game_config
        self.player_number = player_number
        self.pool_size = pool_size
        self.device = device
        if seed is None:
            self.ss_models = SeedSequence()
            self.rng = np.random.default_rng()
        else:
            self.ss_models, self.ss_rng = seed.spawn(2)
            self.rng = np.random.default_rng(ss_rng)

    def add_agent(self, agent: Agent):
        q_net = getattr(agent, "q_network", None)
        if isinstance(q_net, torch.nn.Module):
            q_net.to(self.device)
            freeze_eval(q_net)
        self.pool.append(agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def add_dqn(self, q_network: DuelingFeedForwardNN):
        # Synchronize so any in-flight CUDA work on q_network is finished - attempt at curbing segfault issues
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        new_agent_seed = self.ss_models.spawn(1)[0]
        new_agent = DQNAgent(self.game_config, self.player_number, seed=new_agent_seed)
        snapshot_sd = {k: v.detach().to("cpu", copy=True) for k, v in q_network.state_dict().items()}
        new_agent.q_network.load_state_dict(snapshot_sd)
        new_agent.q_network.to(self.device)
        freeze_eval(new_agent.q_network)
        self.pool.append(new_agent)
        if len(self.pool) > self.pool_size:
            self.pool.popleft()

    def sample(self):
        return self.pool[self.rng.integers(len(self.pool))]

    def get_oldest(self):
        return self.pool[0]


class NStepReplayBuffer:
    def __init__(self, capacity: int, n_step: int, gamma: float, seed: np.random.SeedSequence = None):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buf: list = []
        self.pos = 0
        self.pending = deque()
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def flush(self, trans_queue, final_s, final_mask, done):
        R, g = 0.0, 1.0
        for (_, _, _, r, _, _) in trans_queue:
            R += g * r
            g *= self.gamma
        s0, m0, a0, _, _, _ = trans_queue[0]
        self.add((s0, m0, a0, R, final_s, final_mask, done, g))

    def add(self, item):
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def push(self, s, mask, a, r, s_next, mask_next, done):
        self.pending.append((s, mask, a, r, s_next, done))
        if len(self.pending) >= self.n_step:
            self.flush(list(self.pending), s_next, mask_next, done)
            self.pending.popleft()
        if done:
            while self.pending:
                self.flush(list(self.pending), s_next, mask_next, True)
                self.pending.popleft()

    def sample(self, batch_size):
        idx = self.rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s, m, a, R, s2, m2, done, g = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.stack(m).astype(bool),
            np.array(a, dtype=np.int64),
            np.array(R, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.stack(m2).astype(bool),
            np.array(done, dtype=np.float32),
            np.array(g, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQN:
    def __init__(self, env, agent_names: list[str], run_name, device: torch.device = None, seed: np.random.SeedSequence = None):
        super().__init__()
        self.init_hyperparameters()

        self.device: torch.device = device if device is not None else get_default_device()
        print(f"DQN using device: {self.device}", flush=True)
        print(f"torch threads: {torch.get_num_threads()} / interop: {torch.get_num_interop_threads()}", flush=True)

        self.agent_names = agent_names
        self.env = env

        # seed handling
        ss_buffer, ss_opponent_pool, ss_actor, ss_opponents = seed.spawn(4)

        self.actor = DQNAgent(env.game.game_config, 0, ss_actor)
        self.actor.q_network.to(self.device)
        self.q_optim = Adam(self.actor.q_network.parameters(), lr=self.lr)

        self.target_network = copy.deepcopy(self.actor.q_network).to(self.device)
        # Eval on target not needed
        freeze_eval(self.target_network)

        self.agents = {
            "player_0": self.actor,
            "player_1": None,
        }
        self.opponent_pool = DQNOpponentPool(env.game.game_config, 1, device=self.device, seed=ss_opponent_pool)
        self.opponent_pool.add_agent(DQNAgent(env.game.game_config, 1))

        self.buffer = NStepReplayBuffer(self.buffer_capacity, self.n_step, self.gamma, seed=ss_buffer)

        self.run_name = run_name
        self.tracker = MetricsTracker(f"{self.run_name}", True, 100, {
            "buffer_capacity": self.buffer_capacity,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "n_step": self.n_step,
            "lr": self.lr,
            "target_sync_every": self.target_sync_every,
            "learn_every": self.learn_every,
            "warmup_steps": self.warmup_steps,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
            "games_per_eval": self.games_per_eval,
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
        self.games_played = 0

    def init_hyperparameters(self):
        # replay / learning
        self.buffer_capacity = 350000
        self.batch_size = 128          # was 256; halves per-update gradient cost on CPU
        self.gamma = 0.99
        self.n_step = 3                # was 5; reduces pending-queue bookkeeping + bias
        self.lr = 1e-4

        # target network + update cadence
        self.target_sync_every = 2_000
        self.learn_every = 8           # was 4; halves number of gradient updates per env step
        self.warmup_steps = 10_000

        # exploration
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_steps = 1500000
        self.eps_decay_delay = self.warmup_steps

        # evaluation / pool / logging
        self.games_per_eval = 50
        self.log_every_steps = 30000
        self.eval_every_steps = 30000
        self.pool_push_every_steps = 16000
        self.run_games_every_steps = 50000    # expensive: plays full games vs fixed opponents
        self.max_steps_per_episode = 3200

    def current_epsilon(self) -> float:
        steps_into_decay = max(0, self.global_timestep - self.eps_decay_delay)
        frac = min(1.0, steps_into_decay / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def learn(self, total_timesteps: int):
        self.global_timestep = 0
        last_log_step = 0
        last_pool_push_step = 0
        last_run_games_step = 0
        episode_game_lengths = []
        recent_losses = []
        recent_q_values = []

        print(f"[start] total_timesteps={total_timesteps} warmup_steps={self.warmup_steps} "
              f"learn_every={self.learn_every} batch_size={self.batch_size}", flush=True)

        while self.global_timestep < total_timesteps:
            current_opponent = self.opponent_pool.sample()
            self.agents["player_1"] = current_opponent

            episode_return, ep_len, ep_loss, ep_q = self.play_one_episode()
            self.games_played += 1
            episode_game_lengths.append(ep_len)
            recent_losses.extend(ep_loss)
            recent_q_values.extend(ep_q)

            if self.global_timestep - last_log_step >= self.log_every_steps:
                last_log_step = self.global_timestep
                if episode_game_lengths:
                    self.tracker.record("rollout/average_game_length", float(np.mean(episode_game_lengths)))
                self.tracker.record("rollout/epsilon", self.current_epsilon())
                if recent_losses:
                    self.tracker.record("q/loss", float(np.mean(recent_losses)))
                if recent_q_values:
                    self.tracker.record("q/mean_q_value", float(np.mean(recent_q_values)))
                self.tracker.record("buffer/size", len(self.buffer))
                self.tracker.flush(self.global_timestep)

                episode_game_lengths.clear()
                recent_losses.clear()
                recent_q_values.clear()

            # Run evaluations against fixed opponents
            if self.global_timestep - last_run_games_step >= self.run_games_every_steps:
                last_run_games_step = self.global_timestep
                print(f"[step {self.global_timestep}] running evaluations...", flush=True)
                self.run_evaluations()
                print(f"[step {self.global_timestep}] evaluations done", flush=True)

            # Add current agent to self-play pool
            if self.global_timestep - last_pool_push_step >= self.pool_push_every_steps:
                last_pool_push_step = self.global_timestep
                self.opponent_pool.add_dqn(self.actor.q_network)

            # Take a checkpoint of the current model
            if self.global_timestep // 500_000 >= self.model_checkpoints_taken + 1:
                self.model_checkpoints_taken += 1
                self.save_current_model(f"model_{self.global_timestep}")

        self.run_evaluations(final=True)
        self.save_current_model(f"final_model_{self.global_timestep}", final=True)

    def run_evaluations(self, final: bool = False):
        vs_heuristicagent = self.agent_battler.run_games(
            [self.actor, HeuristicAgentMKII(self.env.game.game_config, 1)], 15, 10000)
        vs_randomagent = self.agent_battler.run_games(
            [self.actor, RandomAgent(self.env.game.game_config, 1)], 15, 10000)
        vs_past_version = self.agent_battler.run_games(
            [self.actor, self.opponent_pool.get_oldest()], 15, 10000)

        self.tracker.record("games_vs_heuristicagentmkii/winrate", vs_heuristicagent["player_wins"]["player_0"] / 15)
        self.tracker.record("games_vs_heuristicagentmkii/average_game_length", vs_heuristicagent["average_game_length"])
        self.tracker.record("games_vs_randomagent/winrate", vs_randomagent["player_wins"]["player_0"] / 15)
        self.tracker.record("games_vs_randomagent/average_game_length", vs_randomagent["average_game_length"])
        self.tracker.record("games_vs_past_version/winrate", vs_past_version["player_wins"]["player_0"] / 15)
        self.tracker.record("games_vs_past_version/average_game_length", vs_past_version["average_game_length"])
        self.tracker.flush(self.global_timestep)

        if final:
            self.tracker.close({
                "final_wr_vs_heuristicagent": vs_heuristicagent["player_wins"]["player_0"] / 100,
                "final_wr_vs_randomagent": vs_randomagent["player_wins"]["player_0"] / 100,
            })

    def play_one_episode(self):
        self.env.reset()
        pending = None  # (s_vec, a, mask) for player_0
        accumulated_reward = 0.0
        episode_return = 0.0
        ep_losses = []
        ep_q_values = []
        ep_steps = 0

        for agent_id in self.env.agent_iter():
            if ep_steps > self.max_steps_per_episode:
                print(f"[rollout] truncating game: timestep {ep_steps}, "f"game_turn={self.env.game.turn_count}, last_agent={agent_id}")
                break
            ep_steps += 1

            observation, reward, termination, truncation, info = self.env.last()
            done = termination or truncation

            if agent_id == "player_0":
                accumulated_reward += reward
                self.global_timestep += 1

                # Close out the previous trainee transition if one is pending.
                if pending is not None:
                    encoded = self.actor.encode(observation)
                    s_next = np.asarray(encoded["observation"], dtype=np.float32)
                    m_next = np.asarray(encoded["action_mask"], dtype=bool)
                    s_prev, m_prev, a_prev = pending
                    self.buffer.push(
                        s_prev, m_prev, a_prev,
                        accumulated_reward,
                        s_next, m_next, done,
                    )
                    episode_return += accumulated_reward
                    accumulated_reward = 0.0
                    pending = None

                if done:
                    self.env.step(None)
                    continue

                # Select next action
                with torch.no_grad():
                    action, q_val, obs_dict = self.actor.act(observation, epsilon=self.current_epsilon())
                ep_q_values.append(float(q_val.item()) if hasattr(q_val, "item") else float(q_val))

                pending = (
                    np.asarray(obs_dict["observation"], dtype=np.float32),
                    np.asarray(obs_dict["action_mask"], dtype=bool),
                    int(action),
                )

                # Gradient step.
                if len(self.buffer) >= self.warmup_steps and self.global_timestep % self.learn_every == 0:
                    loss, mean_q = self.q_update()
                    ep_losses.append(loss)

                # Target sync.
                if self.global_timestep % self.target_sync_every == 0:
                    self.target_network.load_state_dict(self.actor.q_network.state_dict())

                self.env.step(self.env.decode_action(action))

            else:
                if done:
                    self.env.step(None)
                    continue
                with torch.no_grad():
                    action, _unused, _obs = current_opponent_act(self.agents[agent_id], observation)
                self.env.step(self.env.decode_action(action))

        return episode_return, self.env.game.turn_count, ep_losses, ep_q_values

    def q_update(self):
        s, m, a, R, s2, m2, done, g = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(s).to(self.device)
        m = torch.from_numpy(m).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        m2 = torch.from_numpy(m2).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        R = torch.from_numpy(R).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        g = torch.from_numpy(g).to(self.device)

        with torch.no_grad():
            next_q_online = self.actor.q_network(s2, action_mask=m2).masked_fill(~m2, float("-inf"))
            next_actions = next_q_online.argmax(dim=-1, keepdim=True)

            next_q_target = self.target_network(s2, action_mask=m2).masked_fill(~m2, float("-inf"))
            next_q = next_q_target.gather(1, next_actions).squeeze(-1)

            next_q = torch.where(
                done.bool() | torch.isinf(next_q),
                torch.zeros_like(next_q),
                next_q,
            )
            td_target = R + (1.0 - done) * g * next_q

        q_sa = self.actor.q_network(s, action_mask=m).gather(1, a.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q_sa, td_target)

        self.q_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.q_network.parameters(), 10.0)
        self.q_optim.step()

        return loss.item(), q_sa.mean().item()

    def save_current_model(self, filename, final: bool = False):
        if not final:
            path = f"{PROJECT_ROOT}/experiments/{self.run_name}/models/{filename}.pt"
        else:
            path = f"{PROJECT_ROOT}/experiments/{self.run_name}/final/{filename}.pt"
        state_dict = {k: v.detach().cpu() for k, v in self.actor.q_network.state_dict().items()}
        torch.save(state_dict, path)
        print(f"Model saved to {path}", flush=True)


def current_opponent_act(agent, observation):
    try:
        return agent.act(observation, epsilon=0.05)
    except TypeError:
        out = agent.act(observation)
        # Normalize to (action, None, obs) shape.
        if isinstance(out, tuple):
            return (out[0], None, out[-1] if len(out) >= 3 else None)
        return (out, None, None)