"""Config-driven self-match runner that saves MJAI logs."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from loguru import logger

from riichienv import GameRule, MjaiReplay, RiichiEnv

from riichienv_ml.agents import Agent
from riichienv_ml.config import SelfMatchConfig


def _make_rule(rule_name: str) -> GameRule:
    if rule_name == "tenhou":
        return GameRule.default_tenhou()
    if rule_name == "mjsoul":
        return GameRule.default_mjsoul()
    raise ValueError(f"Unsupported rule for self-match: {rule_name}")


class SelfMatchRunner:
    def __init__(self, cfg: SelfMatchConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.summary_path = Path(cfg.summary_path) if cfg.summary_path else self.output_dir / "summary.json"
        self.rule = _make_rule(cfg.game.replay_rule)
        self.game_mode = cfg.game.game_mode
        self.starting_scores = list(cfg.game.starting_scores)
        self._shared_agents, self.seat_agents = self._build_agents()

    def _build_agents(self) -> tuple[list[Agent], dict[int, Agent]]:
        if len(self.cfg.agents) == 1:
            agent_cfg = self.cfg.agents[0]
            shared = Agent(
                config_path=agent_cfg.config_path,
                model_path=agent_cfg.model_path,
                device=agent_cfg.device,
            )
            return [shared], {seat: shared for seat in range(self.cfg.game.n_players)}

        built = []
        for agent_cfg in self.cfg.agents:
            built.append(
                Agent(
                    config_path=agent_cfg.config_path,
                    model_path=agent_cfg.model_path,
                    device=agent_cfg.device,
                )
            )
        return built, {seat: built[seat] for seat in range(self.cfg.game.n_players)}

    def _log_path(self, game_idx: int) -> Path:
        suffix = ".jsonl.gz" if self.cfg.compress_logs else ".jsonl"
        return self.output_dir / f"game_{game_idx:06d}{suffix}"

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pattern = "game_*.jsonl.gz" if self.cfg.compress_logs else "game_*.jsonl"
        if self.cfg.overwrite:
            for path in self.output_dir.glob(pattern):
                path.unlink()
            if self.summary_path.exists():
                self.summary_path.unlink()
            return

        existing = list(self.output_dir.glob(pattern))
        if existing:
            raise FileExistsError(
                f"Output directory already contains self-match logs: {self.output_dir}. "
                "Set self_match.overwrite=true or choose a different output_dir."
            )

    def _write_log(self, path: Path, mjai_log: list[dict]) -> None:
        if self.cfg.compress_logs:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for event in mjai_log:
                    f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                    f.write("\n")
            return

        with path.open("w", encoding="utf-8") as f:
            for event in mjai_log:
                f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")

    def _validate_log(self, path: Path) -> None:
        MjaiReplay.from_jsonl(str(path), rule=self.cfg.game.replay_rule)

    def _play_one_game(self, game_idx: int) -> dict:
        seed = self.cfg.base_seed + game_idx * self.cfg.seed_stride
        env = RiichiEnv(
            game_mode=self.game_mode,
            skip_mjai_logging=self.cfg.skip_mjai_logging,
            seed=seed,
            rule=self.rule,
        )
        for agent in self._shared_agents:
            agent.reset()
        obs_dict = env.reset(scores=list(self.starting_scores))

        while not env.done():
            actions = {
                pid: self.seat_agents[pid].act(obs)
                for pid, obs in obs_dict.items()
            }
            obs_dict = env.step(actions)

        log_path = self._log_path(game_idx)
        self._write_log(log_path, env.mjai_log)
        if self.cfg.validate_saved_logs:
            self._validate_log(log_path)

        return {
            "game_index": game_idx,
            "seed": seed,
            "log_path": str(log_path),
            "ranks": env.ranks(),
            "scores": env.scores(),
            "num_events": len(env.mjai_log),
        }

    def _write_summary(self, summary: dict) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def run(self) -> dict:
        self._ensure_output_dir()

        started_at = time.time()
        games = []
        rank_totals = [0 for _ in range(self.cfg.game.n_players)]
        score_totals = [0 for _ in range(self.cfg.game.n_players)]

        for game_idx in range(self.cfg.num_games):
            result = self._play_one_game(game_idx)
            games.append(result)

            for seat, rank in enumerate(result["ranks"]):
                rank_totals[seat] += rank
            for seat, score in enumerate(result["scores"]):
                score_totals[seat] += score

            completed = game_idx + 1
            if completed % self.cfg.progress_interval == 0 or completed == self.cfg.num_games:
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = self.cfg.num_games - completed
                eta_seconds = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Self-match {}/{} games complete, {:.2f} games/s, ETA {:.1f} min",
                    completed,
                    self.cfg.num_games,
                    rate,
                    eta_seconds / 60.0,
                )

        elapsed = time.time() - started_at
        summary = {
            "num_games": self.cfg.num_games,
            "elapsed_seconds": elapsed,
            "games_per_second": self.cfg.num_games / elapsed if elapsed > 0 else 0.0,
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "game_mode": self.game_mode,
            "replay_rule": self.cfg.game.replay_rule,
            "base_seed": self.cfg.base_seed,
            "seed_stride": self.cfg.seed_stride,
            "average_rank_by_seat": [
                rank_total / self.cfg.num_games for rank_total in rank_totals
            ],
            "average_score_by_seat": [
                score_total / self.cfg.num_games for score_total in score_totals
            ],
            "games": games,
        }
        self._write_summary(summary)
        logger.info("Saved self-match summary to {}", self.summary_path)
        return summary
