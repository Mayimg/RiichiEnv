import json
from pathlib import Path

import torch
import yaml
from riichienv_ml.agents import PolicyDecision, _action_to_policy_payload
from riichienv_ml.config import load_config
from riichienv_ml.models.transformer import TransformerPolicyNetwork
from riichienv_ml.self_match import SelfMatchRunner

from riichienv import Action, ActionType, MjaiReplay


def test_policy_payload_marks_tedashi_and_tsumogiri_discards():
    tedashi = _action_to_policy_payload(Action(ActionType.DISCARD, tile=53, actor=0), drawn_tile=52)
    tsumogiri = _action_to_policy_payload(Action(ActionType.DISCARD, tile=52, actor=0), drawn_tile=52)
    call = _action_to_policy_payload(Action(ActionType.PON, tile=52, consume_tiles=[53, 54], actor=1), drawn_tile=52)

    assert tedashi["tsumogiri"] is False
    assert tedashi["moqie"] == "tedashi"
    assert tedashi["moqie_id"] == 0
    assert tedashi["mjai"]["tsumogiri"] is False

    assert tsumogiri["tsumogiri"] is True
    assert tsumogiri["moqie"] == "tsumogiri"
    assert tsumogiri["moqie_id"] == 1
    assert tsumogiri["mjai"]["tsumogiri"] is True

    assert call["tsumogiri"] is None
    assert call["moqie"] == "na"
    assert call["moqie_id"] == 2
    assert "tsumogiri" not in call["mjai"]


def test_self_match_response_policy_meta_attaches_to_source_event():
    annotated_log = [{"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False}]
    pass_decision = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": True,
            "chosen_action": {"mjai": {"type": "none", "actor": 1}},
        },
    )

    SelfMatchRunner._attach_response_policies(annotated_log[0], [pass_decision])

    assert annotated_log[0]["meta"]["response_policies"] == [pass_decision.meta]


def test_self_match_policy_log_annotation_handles_pass_only_response():
    events = [
        {"type": "start_kyoku"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "1m"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": True},
    ]
    discard_decision = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": False,
            "chosen_action": {"mjai": {"type": "dahai", "actor": 0, "pai": "5p"}},
        },
    )
    pass_decision = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": True,
            "chosen_action": {"mjai": {"type": "none", "actor": 2}},
        },
    )
    next_discard_decision = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": False,
            "chosen_action": {"mjai": {"type": "dahai", "actor": 1, "pai": "1m"}},
        },
    )

    SelfMatchRunner._annotate_policy_log(
        events,
        [
            (1, 2, [discard_decision]),
            (2, 3, [pass_decision]),
            (3, 4, [next_discard_decision]),
        ],
    )

    assert events[1]["meta"]["policy"] == discard_decision.meta
    assert events[1]["meta"]["response_policies"] == [pass_decision.meta]
    assert events[3]["meta"]["policy"] == next_discard_decision.meta


def test_self_match_policy_log_annotation_does_not_match_rejected_claim_to_future_event():
    events = [
        {"type": "start_kyoku"},
        {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": False},
        {"type": "pon", "actor": 0, "pai": "5p", "target": 2, "consumed": ["5p", "5p"]},
        {"type": "dahai", "actor": 0, "pai": "2s", "tsumogiri": False},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {"type": "chi", "actor": 1, "pai": "1m", "target": 0, "consumed": ["2m", "3m"]},
    ]
    source_discard = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": False,
            "chosen_action": {"mjai": {"type": "dahai", "actor": 2, "pai": "5p"}},
        },
    )
    accepted_pon = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": True,
            "chosen_action": {"mjai": {"type": "pon", "actor": 0, "pai": "5p"}},
        },
    )
    rejected_chi = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": True,
            "chosen_action": {"mjai": {"type": "chi", "actor": 1, "pai": "1m"}},
        },
    )
    post_pon_discard = PolicyDecision(
        action=None,
        meta={
            "head": "pointer",
            "response_decision": False,
            "chosen_action": {"mjai": {"type": "dahai", "actor": 0, "pai": "2s"}},
        },
    )

    SelfMatchRunner._annotate_policy_log(
        events,
        [
            (1, 2, [source_discard]),
            (2, 3, [accepted_pon, rejected_chi]),
            (3, 4, [post_pon_discard]),
        ],
    )

    assert events[1]["meta"]["policy"] == source_discard.meta
    assert events[1]["meta"]["response_policies"] == [accepted_pon.meta, rejected_chi.meta]
    assert events[2]["meta"]["policy"] == accepted_pon.meta
    assert events[3]["meta"]["policy"] == post_pon_discard.meta
    assert "meta" not in events[5]


def test_self_match_runner_writes_parseable_mjai_logs(tmp_path):
    train_cfg_path = tmp_path / "bc_config.yml"
    model_path = tmp_path / "model.pth"
    output_dir = tmp_path / "self_match_logs"
    self_match_cfg_path = tmp_path / "self_match.yml"

    train_cfg = {
        "bc": {
            "game": {
                "n_players": 4,
                "replay_rule": "tenhou",
            },
            "model_class": "riichienv_ml.models.transformer.TransformerPolicyNetwork",
            "encoder_class": "riichienv_ml.features.sequence_features.SequenceFeatureEncoder",
            "model": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "num_actions": 82,
            },
        }
    }
    with train_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f)

    model = TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_actions=82,
    )
    torch.save(model.state_dict(), model_path)

    self_match_cfg = {
        "self_match": {
            "game": {
                "n_players": 4,
                "replay_rule": "tenhou",
            },
            "agents": [
                {
                    "config_path": str(train_cfg_path),
                    "model_path": str(model_path),
                    "device": "cpu",
                    "name": "test-agent",
                }
            ],
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "num_games": 1,
            "base_seed": 7,
            "seed_stride": 1,
            "progress_interval": 1,
            "overwrite": False,
            "compress_logs": False,
            "validate_saved_logs": True,
            "log_policy_meta": True,
        }
    }
    with self_match_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(self_match_cfg, f)

    cfg = load_config(str(self_match_cfg_path)).self_match
    summary = SelfMatchRunner(cfg).run()

    log_files = sorted(output_dir.glob("game_*.jsonl"))
    assert len(log_files) == 1

    replay = MjaiReplay.from_jsonl(str(log_files[0]), rule="tenhou")
    assert replay.num_rounds() > 0
    events = [
        json.loads(line)
        for line in log_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    policy_events = [
        event
        for event in events
        if isinstance(event.get("meta"), dict) and "policy" in event["meta"]
    ]
    assert policy_events
    policy = policy_events[0]["meta"]["policy"]
    assert policy["head"] == "pointer"
    assert policy["logit_space"] == "candidate"
    assert policy["candidate_count"] == len(policy["candidates"])
    assert policy["legal_action_count"] == len(policy["legal_actions"])
    assert policy["chosen_action"]["mjai"]["type"] == policy_events[0]["type"]
    discard_candidates = [
        candidate
        for event in policy_events
        for candidate in event["meta"]["policy"]["candidates"]
        if candidate["action"]["mjai"].get("type") == "dahai"
    ]
    assert discard_candidates
    assert all("tsumogiri" in candidate["action"]["mjai"] for candidate in discard_candidates)
    assert {candidate["action"]["moqie_id"] for candidate in discard_candidates} <= {0, 1}

    summary_data = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_games"] == 1
    assert summary_data["log_policy_meta"] is True
    assert summary_data["num_games"] == 1
    assert Path(summary_data["games"][0]["log_path"]) == log_files[0]
