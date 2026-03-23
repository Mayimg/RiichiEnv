import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from riichienv_ml.features.grp_agari_features import (
    TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT,
    encode_agari_rank_gains,
)

LEGACY_GRP_INPUT_FORMAT = "legacy_round_transition"
KYOKU_START_GRP_INPUT_FORMAT = "kyoku_start_with_player_rank"


def _compute_stable_ranks(scores: list[float] | np.ndarray) -> list[int]:
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (-item[1], item[0]))
    ranks = [0] * len(indexed)
    for rank, (seat, _) in enumerate(indexed):
        ranks[seat] = rank
    return ranks


class RankPredictor(nn.Module):
    def __init__(self, input_dim: int = 20, n_players: int = 4):
        super().__init__()
        self.n_players = n_players
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_players)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class RewardPredictor:
    def __init__(self, model_path: str, pts_weight: list[float],
                 n_players: int = 4, input_dim: int | None = None,
                 device: str = "cuda"):
        self.device: str = device
        self.n_players: int = n_players
        self.pts_weight: list[float] = pts_weight
        self._score_norm = 35000.0 if n_players == 3 else 25000.0

        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            self.input_format = checkpoint.get("grp_input_format", LEGACY_GRP_INPUT_FORMAT)
        else:
            state_dict = checkpoint
            self.input_format = LEGACY_GRP_INPUT_FORMAT
        if input_dim is None:
            input_dim = state_dict["fc1.weight"].shape[1]
        self.input_dim = input_dim

        self.model: RankPredictor = RankPredictor(input_dim, n_players)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(torch.device(device))
        self.model = self.model.eval()

    def _calc_pts(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch.softmax(self.model(x), dim=1) @ torch.tensor(self.pts_weight, device=self.device).float()

    def _encode_base_features(self, row: dict) -> np.ndarray:
        n = self.n_players
        round_meta = np.array([
            row["chang"] / 3.0, row["ju"] / 3.0, row["ben"] / 4.0, row["liqibang"] / 4.0
        ], dtype=np.float32)

        has_start_scores = all(f"p{i}_start_score" in row for i in range(n))
        has_old_scores = (
            all(f"p{i}_init_score" in row for i in range(n))
            and all(f"p{i}_end_score" in row for i in range(n))
        )

        if self.input_format in (KYOKU_START_GRP_INPUT_FORMAT, TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT):
            if not has_start_scores:
                raise KeyError(
                    "This GRP checkpoint expects kyoku-start features with p{i}_start_score / p{i}_delta_score keys"
                )
            scores = np.array([row[f"p{i}_start_score"] for i in range(n)], dtype=np.float32) / self._score_norm
            delta_scores = np.array([row[f"p{i}_delta_score"] for i in range(n)], dtype=np.float32) / 12000.0
            base = np.concatenate([scores, delta_scores, round_meta], dtype=np.float32)
        elif has_old_scores:
            scores = np.array(
                [row[f"p{i}_init_score"] for i in range(n)] + [row[f"p{i}_end_score"] for i in range(n)],
                dtype=np.float32,
            ) / self._score_norm
            delta_scores = np.array([row[f"p{i}_delta_score"] for i in range(n)], dtype=np.float32) / 12000.0
            base = np.concatenate([scores, delta_scores, round_meta], dtype=np.float32)
        else:
            raise KeyError("Unsupported GRP feature keys")
        return base

    def _encode_player_rank(self, row: dict, player_idx: int) -> np.ndarray:
        n = self.n_players
        if self.input_format not in (KYOKU_START_GRP_INPUT_FORMAT, TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT):
            return np.zeros(0, dtype=np.float32)

        rank_one_hot = np.zeros(n, dtype=np.float32)
        if all(f"p{i}_current_rank" in row for i in range(n)):
            current_rank = int(row[f"p{player_idx}_current_rank"])
        else:
            current_scores = [row[f"p{i}_start_score"] for i in range(n)]
            current_rank = _compute_stable_ranks(current_scores)[player_idx]
        rank_one_hot[current_rank] = 1.0
        return rank_one_hot

    def _encode_agari_rank_gains(self, row: dict, player_idx: int) -> np.ndarray:
        if self.input_format != TENHOU_4P_AGARI_RANK_GAINS_INPUT_FORMAT:
            return np.zeros(0, dtype=np.float32)

        current_rank = None
        if all(f"p{i}_current_rank" in row for i in range(self.n_players)):
            current_rank = int(row[f"p{player_idx}_current_rank"])

        return encode_agari_rank_gains(
            [row[f"p{i}_start_score"] for i in range(self.n_players)],
            oya=int(row["ju"]),
            honba=int(row["ben"]),
            liqibang=int(row["liqibang"]),
            player_idx=player_idx,
            current_rank=current_rank,
            n_players=self.n_players,
            replay_rule="tenhou",
        )

    def calc_pts_reward(self, row: dict, player_idx: int) -> np.ndarray:
        n = self.n_players
        base = self._encode_base_features(row)
        player = np.zeros(n, dtype=np.float32)
        player[player_idx] = 1.0
        current_rank = self._encode_player_rank(row, player_idx)
        agari_rank_gains = self._encode_agari_rank_gains(row, player_idx)

        x = np.concatenate([base, player, current_rank, agari_rank_gains], dtype=np.float32)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"GRP model expects input_dim={self.input_dim}, but encoded features have {x.shape[0]}")
        return x

    def calc_all_player_rewards(self, grp_features: dict) -> list[float]:
        """Compute final rewards for all players in one batched forward pass."""
        n = self.n_players
        xs = np.stack([self.calc_pts_reward(grp_features, pid) for pid in range(n)], dtype=np.float32)

        xs_t = torch.from_numpy(xs).to(self.device)
        mean_pts = float(np.mean(self.pts_weight))
        with torch.inference_mode():
            pts_w = torch.tensor(self.pts_weight, device=self.device).float()
            pts = torch.softmax(self.model(xs_t), dim=1) @ pts_w
        rewards = (pts - mean_pts).cpu().tolist()
        return rewards

    def calc_pts_rewards(self, kyoku_features: list[dict], player_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        for row in kyoku_features:
            x = self.calc_pts_reward(row, player_idx)
            xs.append(x)

        xs = torch.from_numpy(np.array(xs)).float().to(self.device)
        pts = torch.concat([
            torch.tensor([np.mean(self.pts_weight)], device=self.device).float(),
            self._calc_pts(xs)
        ], dim=0)

        rewards = pts[1:] - pts[:-1]
        return pts[1:], rewards
