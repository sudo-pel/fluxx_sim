from pathlib import Path

import torch

from src.agents.DQNAgent import DQNAgent
from src.agents.DQNAgentGeneralized import DQNAgentGeneralized
from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.agents.card_embeddings import generate_embedding_table
from src.training.TrainingEnums import GameLogConfig
from src.env.AgentBattler import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists

def format_matchup_grid(results: dict[tuple[str, str], dict[str, float]]) -> str:
    """
    Format pairwise agent matchup results as a readable grid.

    Each cell shows the row agent's win rate against the column agent,
    plus the average game length in parentheses. Diagonal cells (self-play)
    are filled with "—" if not present in the results.

    Args:
        results: Mapping from (agent_a, agent_b) tuples to a dict of stat (piped straight from agent battler output).
            Each stats dict must contain:
                - "player_wins": int, wins for agent_a in this matchup.
                - "total_games": int, total games played.
                - "average_game_length": float, mean game length.
            The dict is treated as directional: results[(a, b)] describes
            agent_a's performance against agent_b. Both directions can be
            present and will be rendered in their respective cells.

    Returns:
        A multi-line string containing the formatted grid.

    Example:
        Matchup win rates (row vs column) — avg game length in parens
        ════════════════════════════════════════════════════════════════
                       │ DQN          │ Heuristic    │ Random       │
        ───────────────┼──────────────┼──────────────┼──────────────┤
        DQN            │ —            │ 64.0% (24.7) │ 94.0% (18.4) │
        Heuristic      │ 36.0% (24.7) │ —            │ 82.0% (21.2) │
        Random         │  6.0% (18.4) │ 18.0% (21.2) │ —            │
        ───────────────┴──────────────┴──────────────┴──────────────┘
    """
    if not results:
        return "No matchup data."

    # Collect the set of agent names from both sides of each pairing.
    agents = sorted({name for pair in results for name in pair})

    # Decide column width. Each cell is "XX.X% (YY.Y)" plus padding.
    name_col_width = max(len("Agent"), max(len(a) for a in agents))
    cell_width = max(12, max(len(a) for a in agents))

    def fmt_cell(stats: dict[str, float] | None) -> str:
        if stats is None:
            return "—".center(cell_width)
        wins = stats["player_wins"]
        total = stats["total_games"]
        winrate = (wins / total) * 100 if total else 0.0
        avg_len = stats["average_game_length"]
        text = f"{winrate:5.1f}% ({avg_len:.1f})"
        return text.ljust(cell_width)

    lines: list[str] = []

    # Header
    title = "Matchup win rates (row vs column) — avg game length in parens"
    lines.append(title)
    lines.append("═" * len(title))

    # Column header row
    header = " " * name_col_width + " │ " + " │ ".join(a.ljust(cell_width) for a in agents) + " │"
    lines.append(header)

    # Separator under header
    sep = "─" * name_col_width + "─┼─" + "─┼─".join("─" * cell_width for _ in agents) + "─┤"
    lines.append(sep)

    # Body rows
    for row_agent in agents:
        cells: list[str] = []
        for col_agent in agents:
            if row_agent == col_agent:
                cells.append("—".ljust(cell_width))
            else:
                cells.append(fmt_cell(results.get((row_agent, col_agent))))
        line = row_agent.ljust(name_col_width) + " │ " + " │ ".join(cells) + " │"
        lines.append(line)

    # Bottom border
    bottom = "─" * name_col_width + "─┴─" + "─┴─".join("─" * cell_width for _ in agents) + "─┘"
    lines.append(bottom)

    return "\n".join(lines)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

generate_embedding_table(card_lists.base_deck)

two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, logger=None)
env = FluxxEnv(two_player_fluxx, 2, render_mode="human")
agent_battler = AgentBattler(env)

# Prepare the agents.
agents = {
    "ppo": PPOAgent(env.game.game_config, 0),
    "ppo_general": PPOAgentGeneralized(env.game.game_config, 0),
    "dqn": DQNAgent(env.game.game_config, 0),
    "dqn_general": None,
    "ppo_general_with_reward_shaping": None,
    "random": RandomAgent(env.game.game_config, 0),
    "heuristic_agent_mki": HeuristicAgentMKI(env.game.game_config, 0),
    "heuristic_agent_mkii": HeuristicAgentMKII(env.game.game_config, 0),
}

# Load the trained state dicts.
agents["ppo"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_2026-04-28_17-11-02/final/final_model_50004751.pt"))
agents["ppo"].policy_network.eval()

agents["ppo_general"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_general_2026-05-02_09-11-14/models/model_42008245.pt"))
agents["ppo_general"].policy_network.eval()

agents["dqn"].q_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/dqn_2026-04-29_08-06-30/final/final_model_50000050.pt"))
agents["dqn"].q_network.eval()

# TODO: dqn_general and ppo_general_with_reward_shaping

# (Temporary) filter out uninitialized agents
for agent_name, agent in agents.items():
    if agent is None:
        del agents[agent_name]

results: dict[tuple[str, str], dict[str, float]] = {}

for agent_name, agent in agents.items():
    for other_agent_name, other_agent in agents.items():
        if agent_name == other_agent_name: continue
        print(f"RUNNING {agent_name} vs {other_agent_name}")
        results[(agent_name, other_agent_name)] = agent_battler.run_games([agent, other_agent], 10, 10000, log_games=False)
        print(f"RESULTS: {results[(agent_name, other_agent_name)]}")

print(format_matchup_grid(results))