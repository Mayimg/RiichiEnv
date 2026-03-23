import glob
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjaiReplay

from riichienv_ml.datasets.mjai_logs import _compute_rank


class GrpReplayDataset(IterableDataset):
    """GRP dataset over kyoku start states.

    For each kyoku in each replay, encodes the state at the start of the kyoku:
    current start scores, score deltas from the previous kyoku boundary, and
    round metadata. The target remains the final hanchan rank for each player.

    The replay's first kyoku start state is excluded because it has no preceding
    kyoku boundary and is therefore outside the intended GRP inference
    distribution.
    """

    def __init__(
        self,
        data_glob: str,
        n_players: int = 4,
        replay_rule: str = "mjsoul",
        is_train: bool = True,
    ):
        self.data_glob = data_glob
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.is_train = is_train

    def _get_files(self) -> list[str]:
        files = sorted(glob.glob(self.data_glob, recursive=True))
        return files

    def _encode_features(self, grp_features: dict, player_idx: int) -> torch.Tensor:
        n = self.n_players
        score_norm = 35000.0 if n == 3 else 25000.0

        scores = np.array(
            [grp_features[f"p{i}_start_score"] / score_norm for i in range(n)]
            + [grp_features[f"p{i}_delta_score"] / 12000.0 for i in range(n)],
            dtype=np.float32,
        )
        round_meta = np.array([
            grp_features["chang"] / 3.0,
            grp_features["ju"] / 3.0,
            grp_features["ben"] / 4.0,
            grp_features["liqibang"] / 4.0,
        ], dtype=np.float32)
        player = np.zeros(n, dtype=np.float32)
        player[player_idx] = 1.0

        x = np.concatenate([scores, round_meta, player])
        return torch.from_numpy(x)

    def _encode_label(self, final_scores: list, player_idx: int) -> torch.Tensor:
        n = self.n_players
        rank = _compute_rank(final_scores, player_idx, n)
        y = np.zeros(n, dtype=np.float32)
        y[rank] = 1.0
        return torch.from_numpy(y)

    def _extract_kyoku_start_features(self, kyoku_features: list[dict]) -> list[dict]:
        state_features = []
        prev_start_scores: list[int] | None = None

        for feat in kyoku_features:
            start_scores = list(feat["round_initial_scores"][:self.n_players])
            if prev_start_scores is None:
                delta_scores = [0] * self.n_players
            else:
                delta_scores = [start_scores[i] - prev_start_scores[i] for i in range(self.n_players)]

            row = {
                "chang": feat["chang"],
                "ju": feat["ju"],
                "ben": feat["ben"],
                "liqibang": feat["liqibang"],
            }
            for i in range(self.n_players):
                row[f"p{i}_start_score"] = start_scores[i]
                row[f"p{i}_delta_score"] = delta_scores[i]

            state_features.append(row)
            prev_start_scores = start_scores

        return state_features

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        buffer = []
        for file_path in files:
            try:
                replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
                raw_kyoku_features = []
                for kyoku in replay.take_kyokus():
                    raw_kyoku_features.append(kyoku.take_grp_features())

                if not raw_kyoku_features:
                    continue

                kyoku_start_features = self._extract_kyoku_start_features(raw_kyoku_features)
                if len(kyoku_start_features) <= 1:
                    continue
                kyoku_start_features = kyoku_start_features[1:]

                # Final hanchan ranking from the last kyoku's end scores
                final_scores = list(raw_kyoku_features[-1]["round_end_scores"][:self.n_players])

                for grp_features in kyoku_start_features:
                    for player_idx in range(self.n_players):
                        x = self._encode_features(grp_features, player_idx)
                        y = self._encode_label(final_scores, player_idx)
                        buffer.append((x, y))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

            # Flush buffer periodically to limit memory usage
            if len(buffer) >= 10000:
                if self.is_train:
                    random.shuffle(buffer)
                yield from buffer
                buffer.clear()

        # Flush remaining
        if buffer:
            if self.is_train:
                random.shuffle(buffer)
            yield from buffer
