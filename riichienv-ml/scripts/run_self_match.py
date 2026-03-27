"""Run config-driven self-match games and save MJAI logs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

load_dotenv()

from riichienv_ml.config import SelfMatchAgentConfig, load_config
from riichienv_ml.self_match import SelfMatchRunner
from riichienv_ml.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-match games and save MJAI logs")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--num_games", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--seed_stride", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress_logs", action="store_true")
    parser.add_argument("--validate_saved_logs", action="store_true")
    parser.add_argument("--agent_model_path", type=str, default=None)
    parser.add_argument("--agent_device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).self_match

    overrides = {}
    for field in ["num_games", "output_dir", "base_seed", "seed_stride", "progress_interval"]:
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if args.overwrite:
        overrides["overwrite"] = True
    if args.compress_logs:
        overrides["compress_logs"] = True
    if args.validate_saved_logs:
        overrides["validate_saved_logs"] = True

    if args.agent_model_path is not None or args.agent_device is not None:
        updated_agents = []
        for agent_cfg in cfg.agents:
            agent_updates = {}
            if args.agent_model_path is not None:
                agent_updates["model_path"] = args.agent_model_path
            if args.agent_device is not None:
                agent_updates["device"] = args.agent_device
            updated_agents.append(agent_cfg.model_copy(update=agent_updates))
        overrides["agents"] = updated_agents

    if overrides:
        cfg = cfg.model_copy(update=overrides)

    output_dir = Path(cfg.output_dir)
    setup_logging(str(output_dir), "self_match")

    if len(cfg.agents) == 1:
        agent_cfg = cfg.agents[0]
        cfg = cfg.model_copy(
            update={
                "agents": [
                    SelfMatchAgentConfig(
                        config_path=agent_cfg.config_path,
                        model_path=agent_cfg.model_path,
                        device=agent_cfg.device,
                        name=agent_cfg.name,
                    )
                ]
            }
        )

    runner = SelfMatchRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
