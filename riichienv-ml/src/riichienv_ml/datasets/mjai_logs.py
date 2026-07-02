import glob
import logging
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from riichienv import MjaiReplay

logger = logging.getLogger(__name__)


def _compute_rank(end_scores: list, player_id: int, n_players: int) -> int:
    """Compute rank (0=1st, n-1=last) from end-of-kyoku scores."""
    scores = np.array(end_scores[:n_players], dtype=np.float64)
    return int((-scores).argsort(kind='stable').argsort(kind='stable')[player_id])


class GrpFeatureEncoder:
    """Extracts GRP features from a kyoku, parameterized by n_players."""

    def __init__(self, kyoku, n_players: int = 4):
        self.kyoku = kyoku
        self.n_players = n_players

    def encode(self) -> dict:
        feat = self.kyoku.take_grp_features()
        n = self.n_players
        row = {}
        for i in range(n):
            row[f"p{i}_init_score"] = feat["round_initial_scores"][i]
            row[f"p{i}_end_score"] = feat["round_end_scores"][i]
            row[f"p{i}_delta_score"] = feat["round_delta_scores"][i]
        row["chang"] = feat["chang"]
        row["ju"] = feat["ju"]
        row["ben"] = feat["ben"]
        row["liqibang"] = feat["liqibang"]
        return row


class BaseDataset(IterableDataset):
    def __init__(self, data_sources, reward_predictor=None, gamma=0.99,
                 is_train=True, n_players=4, replay_rule="mjsoul", encoder=None):
        self.data_sources = data_sources
        self.reward_predictor = reward_predictor
        self.gamma = gamma
        self.is_train = is_train
        self.n_players = n_players
        self.replay_rule = replay_rule
        self.encoder = encoder

    def _get_files(self):
        if isinstance(self.data_sources, list):
            return self.data_sources
        elif isinstance(self.data_sources, str):
            return glob.glob(self.data_sources, recursive=True)
        return []


class MCDataset(BaseDataset):
    """Yields (features, action_id, return, mask, rank).

    Target: Monte-Carlo Return (G_t), decayed.
    Uses MjaiReplay.from_jsonl() for replay parsing.
    """

    def __iter__(self):
        files = self._get_files()
        if self.is_train:
            random.shuffle(files)

        # Shard files across DataLoader workers to avoid duplicated work
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        skipped = 0
        total = len(files)

        for file_path in files:
            try:
                replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
            except (RuntimeError, ValueError) as e:
                logger.warning("Skipping unparseable replay: %s: %s", file_path, e)
                skipped += 1
                continue

            buffer = []

            try:
                for kyoku in replay.take_kyokus():
                    grp_features = GrpFeatureEncoder(kyoku, self.n_players).encode()

                    assert self.reward_predictor is not None
                    all_rewards = self.reward_predictor.calc_all_player_rewards(grp_features)

                    end_scores = [grp_features[f"p{i}_end_score"] for i in range(self.n_players)]

                    for player_id in range(self.n_players):
                        trajectory = []
                        final_reward = all_rewards[player_id]
                        rank = _compute_rank(end_scores, player_id, self.n_players)

                        for obs, action in kyoku.steps(player_id):
                            features = self.encoder.encode(obs)
                            action_id = action.encode()

                            mask_bytes = obs.mask()
                            mask = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
                            assert 0 <= action_id < mask.shape[0], f"action_id should be in [0, {mask.shape[0]})"
                            assert mask[action_id] == 1, f"action_id {action_id} should be legal"
                            trajectory.append((features, action_id, mask))

                        trajectory_len = len(trajectory)
                        for t, (feat, act, mask) in enumerate(trajectory):
                            decayed = final_reward * (self.gamma ** (trajectory_len - t - 1))
                            buffer.append((feat, act, decayed, mask, rank))
            except (RuntimeError, ValueError) as e:
                logger.warning("Skipping replay due to error: %s: %s", file_path, e)
                skipped += 1
                continue

            if self.is_train:
                random.shuffle(buffer)

            yield from buffer

        if skipped > 0:
            logger.warning("Skipped %d / %d replay files due to errors", skipped, total)


class BehaviorCloningDataset(BaseDataset):
    """Yields (features, candidate_index, candidate_mask) tuples for pure action cloning."""

    def __init__(self, *args, shuffle_buffer_files: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle_buffer_files = int(shuffle_buffer_files)
        if self.shuffle_buffer_files < 1:
            raise ValueError("shuffle_buffer_files must be >= 1")

    def _candidate_mask(self, obs) -> np.ndarray:
        return np.ones(len(obs.candidate_actions()), dtype=np.uint8)

    def _load_file_samples(self, file_path: str):
        try:
            replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
        except (RuntimeError, ValueError) as e:
            logger.warning("Skipping unparseable replay: %s: %s", file_path, e)
            return None

        buffer = []
        try:
            for kyoku in replay.take_kyokus():
                for player_id in range(self.n_players):
                    for obs, action in kyoku.steps(player_id):
                        features = self.encoder.encode(obs)
                        action_id = obs.find_candidate_index(action)
                        if action_id is None:
                            raise ValueError(f"action {action} is not in candidate actions")

                        mask = self._candidate_mask(obs)
                        if not 0 <= action_id < mask.shape[0]:
                            raise ValueError(f"candidate index {action_id} exceeds candidate_count={mask.shape[0]}")
                        if mask[action_id] != 1:
                            raise ValueError(f"candidate index {action_id} is not legal")
                        buffer.append((features, action_id, mask))
        except (RuntimeError, ValueError) as e:
            logger.warning("Skipping replay due to error: %s: %s", file_path, e)
            return None

        return buffer

    def _yield_shuffled(self, buffer: list):
        random.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        files = list(self._get_files())
        if self.is_train:
            random.shuffle(files)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        skipped = 0
        total = len(files)

        if self.is_train:
            shuffle_buffer = []
            buffered_files = 0
            for file_path in files:
                samples = self._load_file_samples(file_path)
                if samples is None:
                    skipped += 1
                    continue

                shuffle_buffer.extend(samples)
                buffered_files += 1
                if buffered_files >= self.shuffle_buffer_files:
                    yield from self._yield_shuffled(shuffle_buffer)
                    shuffle_buffer = []
                    buffered_files = 0

            if shuffle_buffer:
                yield from self._yield_shuffled(shuffle_buffer)

            if skipped > 0:
                logger.warning("Skipped %d / %d replay files due to errors", skipped, total)
            return

        for file_path in files:
            samples = self._load_file_samples(file_path)
            if samples is None:
                skipped += 1
                continue
            yield from samples

        if skipped > 0:
            logger.warning("Skipped %d / %d replay files due to errors", skipped, total)


class BehaviorCloningRankDataset(BehaviorCloningDataset):
    """Yields BC action samples plus kyoku-start rank-only samples.

    Samples are ``(features, candidate_index, candidate_mask, rank, has_policy)``.
    Action-decision samples set ``has_policy=True`` and use the final hanchan
    rank label. Kyoku-start samples set ``has_policy=False`` and contain no
    legal candidates; only their rank label contributes to training.
    """

    def _final_ranks(self, raw_kyoku_features: list[dict]) -> list[int]:
        if not raw_kyoku_features:
            raise ValueError("raw_kyoku_features must not be empty")
        last = raw_kyoku_features[-1]
        if "final_ranks" in last:
            return [int(rank) for rank in last["final_ranks"][: self.n_players]]
        final_scores = list(last["round_end_scores"][: self.n_players])
        return [_compute_rank(final_scores, player_id, self.n_players) for player_id in range(self.n_players)]

    def _load_file_samples(self, file_path: str):  # noqa: PLR0915
        try:
            replay = MjaiReplay.from_jsonl(file_path, rule=self.replay_rule)
        except (RuntimeError, ValueError) as e:
            logger.warning("Skipping unparseable replay: %s: %s", file_path, e)
            return None

        buffer = []
        try:
            kyokus = list(replay.take_kyokus())
            raw_kyoku_features = [kyoku.take_grp_features() for kyoku in kyokus]
            if not raw_kyoku_features:
                return buffer

            final_ranks = self._final_ranks(raw_kyoku_features)

            for kyoku_idx, kyoku in enumerate(kyokus):
                for player_id in range(self.n_players):
                    for obs, action in kyoku.steps(player_id):
                        features = self.encoder.encode(obs)
                        action_id = obs.find_candidate_index(action)
                        if action_id is None:
                            raise ValueError(f"action {action} is not in candidate actions")

                        mask = self._candidate_mask(obs)
                        if not 0 <= action_id < mask.shape[0]:
                            raise ValueError(f"candidate index {action_id} exceeds candidate_count={mask.shape[0]}")
                        if mask[action_id] != 1:
                            raise ValueError(f"candidate index {action_id} is not legal")
                        buffer.append((features, action_id, mask, final_ranks[player_id], True))

                if kyoku_idx == 0:
                    continue

                feat = raw_kyoku_features[kyoku_idx]
                scores = list(feat["round_initial_scores"][: self.n_players])
                for player_id in range(self.n_players):
                    features = self.encoder.encode_kyoku_start(
                        scores=scores,
                        chang=feat["chang"],
                        ju=feat["ju"],
                        ben=feat["ben"],
                        liqibang=feat["liqibang"],
                        player_id=player_id,
                    )
                    buffer.append(
                        (
                            features,
                            0,
                            np.zeros(0, dtype=np.uint8),
                            final_ranks[player_id],
                            False,
                        )
                    )
        except (RuntimeError, ValueError) as e:
            logger.warning("Skipping replay due to error: %s: %s", file_path, e)
            return None

        return buffer


class DiscardHistoryDataset(MCDataset):
    """MCDataset with discard history decay features (78 channels)."""
    pass


class DiscardHistoryShantenDataset(MCDataset):
    """MCDataset with discard history + shanten features (94 channels)."""
    pass


class ExtendedDataset(MCDataset):
    """MCDataset with extended features (215 channels)."""
    pass
