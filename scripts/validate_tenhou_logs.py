"""
Datasource: https://www.kaggle.com/datasets/shokanekolouis/tenhou-to-mjai
"""
import argparse
from pathlib import Path

from riichienv import MjaiReplay
from riichienv._riichienv import Observation, Observation3P
import numpy as np


def validate_obs_4p(obs: Observation) -> None:
    """Validate all encode_* methods on a 4-player Observation."""
    feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 34)
    assert np.isfinite(feat).all(), "Feature contains non-finite values"

    feat = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, 34)
    assert np.isfinite(feat).all(), "Discard history decay contains non-finite values"

    feat = np.frombuffer(obs.encode_yaku_possibility(), dtype=np.float32).reshape(4, 21, 2)
    assert np.isfinite(feat).all(), "Yaku possibility contains non-finite values"

    feat = np.frombuffer(obs.encode_furiten_ron_possibility(), dtype=np.float32).reshape(4, 21)
    assert np.isfinite(feat).all(), "Furiten ron possibility contains non-finite values"

    feat = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(4, 4)
    assert np.isfinite(feat).all(), "Shanten efficiency contains non-finite values"

    feat = np.frombuffer(obs.encode_kawa_overview(), dtype=np.float32).reshape(4, 7, 34)
    assert np.isfinite(feat).all(), "Kawa overview contains non-finite values"

    feat = np.frombuffer(obs.encode_fuuro_overview(), dtype=np.float32).reshape(4, 4, 5, 34)
    assert np.isfinite(feat).all(), "Fuuro overview contains non-finite values"

    feat = np.frombuffer(obs.encode_ankan_overview(), dtype=np.float32).reshape(4, 34)
    assert np.isfinite(feat).all(), "Ankan overview contains non-finite values"

    feat = np.frombuffer(obs.encode_action_availability(), dtype=np.float32).reshape((11,))
    assert np.isfinite(feat).all(), "Action availability contains non-finite values"

    feat = np.frombuffer(obs.encode_riichi_sutehais(), dtype=np.float32).reshape(3, 3)
    assert np.isfinite(feat).all(), "Riichi sutehais contains non-finite values"

    feat = np.frombuffer(obs.encode_last_tedashis(), dtype=np.float32).reshape(3, 3)
    assert np.isfinite(feat).all(), "Last tedashis contains non-finite values"

    feat = np.frombuffer(obs.encode_pass_context(), dtype=np.float32).reshape((3,))
    assert np.isfinite(feat).all(), "Pass context contains non-finite values"

    feat = np.frombuffer(obs.encode_discard_candidates(), dtype=np.float32).reshape((5,))
    assert np.isfinite(feat).all(), "Discard candidates contains non-finite values"

    feat = np.frombuffer(obs.encode_extended(), dtype=np.float32).reshape(215, 34)
    assert np.isfinite(feat).all(), "Extended features contains non-finite values"

    # Sequence encodings (variable-length)
    buf = obs.encode_seq_sparse()
    feat = np.frombuffer(buf, dtype=np.uint16)
    assert len(feat) <= 25, f"Seq sparse too long: {len(feat)}"

    feat = np.frombuffer(obs.encode_seq_numeric(), dtype=np.float32).reshape((12,))
    assert np.isfinite(feat).all(), "Seq numeric contains non-finite values"

    buf = obs.encode_seq_progression()
    feat = np.frombuffer(buf, dtype=np.uint16)
    assert feat.size % 5 == 0, f"Seq progression size {feat.size} not divisible by 5"
    if feat.size > 0:
        feat = feat.reshape(-1, 5)

    buf = obs.encode_seq_candidates()
    feat = np.frombuffer(buf, dtype=np.uint16)
    assert feat.size % 4 == 0, f"Seq candidates size {feat.size} not divisible by 4"
    if feat.size > 0:
        feat = feat.reshape(-1, 4)


def validate_obs_3p(obs: Observation3P) -> None:
    """Validate all encode_* methods on a 3-player Observation3P."""
    feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 27)
    assert np.isfinite(feat).all(), "Feature contains non-finite values"

    feat = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(3, 27)
    assert np.isfinite(feat).all(), "Discard history decay contains non-finite values"

    feat = np.frombuffer(obs.encode_yaku_possibility(), dtype=np.float32).reshape(3, 21, 2)
    assert np.isfinite(feat).all(), "Yaku possibility contains non-finite values"

    feat = np.frombuffer(obs.encode_furiten_ron_possibility(), dtype=np.float32).reshape(3, 21)
    assert np.isfinite(feat).all(), "Furiten ron possibility contains non-finite values"

    feat = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(3, 4)
    assert np.isfinite(feat).all(), "Shanten efficiency contains non-finite values"

    feat = np.frombuffer(obs.encode_kawa_overview(), dtype=np.float32).reshape(3, 7, 27)
    assert np.isfinite(feat).all(), "Kawa overview contains non-finite values"

    feat = np.frombuffer(obs.encode_fuuro_overview(), dtype=np.float32).reshape(3, 4, 5, 27)
    assert np.isfinite(feat).all(), "Fuuro overview contains non-finite values"

    feat = np.frombuffer(obs.encode_ankan_overview(), dtype=np.float32).reshape(3, 27)
    assert np.isfinite(feat).all(), "Ankan overview contains non-finite values"

    feat = np.frombuffer(obs.encode_action_availability(), dtype=np.float32).reshape((11,))
    assert np.isfinite(feat).all(), "Action availability contains non-finite values"

    feat = np.frombuffer(obs.encode_riichi_sutehais(), dtype=np.float32).reshape(2, 3)
    assert np.isfinite(feat).all(), "Riichi sutehais contains non-finite values"

    feat = np.frombuffer(obs.encode_last_tedashis(), dtype=np.float32).reshape(2, 3)
    assert np.isfinite(feat).all(), "Last tedashis contains non-finite values"

    feat = np.frombuffer(obs.encode_pass_context(), dtype=np.float32).reshape((3,))
    assert np.isfinite(feat).all(), "Pass context contains non-finite values"

    feat = np.frombuffer(obs.encode_discard_candidates(), dtype=np.float32).reshape((5,))
    assert np.isfinite(feat).all(), "Discard candidates contains non-finite values"

    feat = np.frombuffer(obs.encode_extended(), dtype=np.float32).reshape(215, 27)
    assert np.isfinite(feat).all(), "Extended features contains non-finite values"


def validate_tenhou_log(log_path: Path, *, rule: str | None = None) -> None:
    """Validates a Tenhou log by attempting to parse and encode all features.

    Raises ValueError or AssertionError if the log is invalid.
    """
    replay = MjaiReplay.from_jsonl(str(log_path), rule=rule)
    kyokus = list(replay.take_kyokus())

    for kyoku_idx, kyoku in enumerate(kyokus):
        is_3p = len(kyoku.scores) == 3
        mode = "3P" if is_3p else "4P"

        step_count = 0
        for _step_idx, obs, _action in kyoku.steps(skip_single_action=False):
            if is_3p:
                assert isinstance(obs, Observation3P), f"Expected Observation3P, got {type(obs)}"
                validate_obs_3p(obs)
            else:
                assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
                validate_obs_4p(obs)
            step_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Tenhou logs by encoding all observation features")
    parser.add_argument("paths", nargs="*", help="Log file paths or directories to validate")
    parser.add_argument("--rule", default="tenhou", choices=["tenhou", "mjsoul", "tenhou_sanma", "mjsoul_sanma"],
                        help="Game rule to use for parsing")
    parser.add_argument("--glob", default="*.mjson", help="Glob pattern for log files (default: *.mjson)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of files to validate (0=all)")
    args = parser.parse_args()

    if not args.paths:
        args.paths = ["/data/workspace/tenhou/2026"]

    log_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            log_files.append(path)
        elif path.is_dir():
            log_files.extend(sorted(path.glob(args.glob)))
        else:
            print(f"Warning: {p} not found, skipping")

    if not log_files:
        print("No log files found")
        return

    total = len(log_files)
    if args.limit > 0:
        log_files = log_files[:args.limit]

    print(f"Found {total} log files, validating {len(log_files)}")

    failed = 0
    for i, log_path in enumerate(log_files):
        print(f"[{i + 1}/{len(log_files)}] {log_path.name}")
        try:
            validate_tenhou_log(log_path, rule=args.rule)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nDone: {len(log_files) - failed}/{len(log_files)} passed")


if __name__ == "__main__":
    main()
