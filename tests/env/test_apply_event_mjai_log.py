"""Tests for mjai_log recording in apply_event() and observe_event().

Verifies that events applied via apply_event() / observe_event() are
correctly recorded in mjai_log so that get_viewer() can display them.

Regression tests for https://github.com/smly/RiichiEnv/issues/184
"""

from riichienv import RiichiEnv


# ---------------------------------------------------------------------------
# Shared event builders
# ---------------------------------------------------------------------------

_4P_TEHAIS = [
    ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
    ["1m", "1m", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s"],
    ["1z", "2z", "3z", "4z", "5z", "6z", "7z", "1s", "2s", "3s", "7s", "8s", "9s"],
    ["4s", "5s", "6s", "7s", "8s", "9s", "4m", "5m", "6m", "7m", "8m", "9m", "1z"],
]

_3P_TEHAIS = [
    ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z"],
    ["1p", "1p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "3z", "4z"],
    ["5z", "6z", "7z", "7s", "8s", "9s", "7p", "8p", "9p", "1m", "9m", "1z", "2z"],
]


def _start_kyoku_4p(oya=0):
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "2p",
        "kyoku": 1,
        "honba": 0,
        "kyoutaku": 0,
        "oya": oya,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": _4P_TEHAIS,
    }


def _start_kyoku_3p(oya=0):
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "2p",
        "kyoku": 1,
        "honba": 0,
        "kyoutaku": 0,
        "oya": oya,
        "scores": [35000, 35000, 35000],
        "tehais": _3P_TEHAIS,
    }


# ===========================================================================
# apply_event: mjai_log recording (4P)
# ===========================================================================


class TestApplyEventMjaiLog4P:
    """Verify apply_event() records events in mjai_log for 4-player mode."""

    def test_start_game_clears_constructor_log(self):
        """start_game via apply_event should clear stale log from constructor."""
        env = RiichiEnv(game_mode="default")
        # Constructor creates initial events
        assert len(env.mjai_log) > 0

        env.apply_event({"type": "start_game"})
        assert len(env.mjai_log) == 1
        assert env.mjai_log[0]["type"] == "start_game"

    def test_events_accumulate_in_order(self):
        """Events applied via apply_event should accumulate in mjai_log."""
        env = RiichiEnv(game_mode="default")
        events = [
            {"type": "start_game"},
            _start_kyoku_4p(),
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        ]
        for ev in events:
            env.apply_event(ev)

        log_types = [e["type"] for e in env.mjai_log]
        assert log_types == ["start_game", "start_kyoku", "tsumo", "dahai"]

    def test_start_game_preserves_custom_fields(self):
        """start_game event with extra fields (e.g. names) should be logged as-is."""
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game", "names": ["A", "B", "C", "D"]})
        assert env.mjai_log[0]["type"] == "start_game"

    def test_pon_event_logged(self):
        """Pon event should be recorded in mjai_log."""
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        env.apply_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
        env.apply_event(
            {
                "type": "pon",
                "actor": 1,
                "target": 0,
                "pai": "1m",
                "consumed": ["1m", "1m"],
            }
        )

        log_types = [e["type"] for e in env.mjai_log]
        assert "pon" in log_types

    def test_chi_event_logged(self):
        """Chi event should be recorded in mjai_log."""
        tehais = [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["4m", "5m", "6m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1z"],
            ["1z", "2z", "3z", "4z", "5z", "6z", "7z", "1s", "2s", "3s", "7s", "8s", "9s"],
            ["4s", "5s", "6s", "7s", "8s", "9s", "4m", "5m", "6m", "7m", "8m", "9m", "1z"],
        ]
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event({**_start_kyoku_4p(), "tehais": tehais})
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        env.apply_event({"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False})
        env.apply_event(
            {
                "type": "chi",
                "actor": 1,
                "target": 0,
                "pai": "3m",
                "consumed": ["4m", "5m"],
            }
        )

        log_types = [e["type"] for e in env.mjai_log]
        assert "chi" in log_types

    def test_reach_events_logged(self):
        """Reach and reach_accepted should be recorded in mjai_log."""
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        env.apply_event({"type": "reach", "actor": 0})
        env.apply_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
        env.apply_event({"type": "reach_accepted", "actor": 0})

        log_types = [e["type"] for e in env.mjai_log]
        assert "reach" in log_types
        assert "reach_accepted" in log_types

    def test_full_round_event_count(self):
        """A minimal round should log all events."""
        env = RiichiEnv(game_mode="default")
        events = [
            {"type": "start_game"},
            _start_kyoku_4p(),
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "6s"},
            {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": True},
            {"type": "tsumo", "actor": 2, "pai": "4z"},
            {"type": "dahai", "actor": 2, "pai": "4z", "tsumogiri": True},
            {"type": "tsumo", "actor": 3, "pai": "2z"},
            {"type": "dahai", "actor": 3, "pai": "2z", "tsumogiri": True},
        ]
        for ev in events:
            env.apply_event(ev)

        assert len(env.mjai_log) == len(events)

    def test_end_game_logged(self):
        """end_kyoku and end_game events should be logged."""
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        env.apply_event({"type": "hora", "actor": 0, "target": 0})
        env.apply_event({"type": "end_kyoku"})
        env.apply_event({"type": "end_game"})

        log_types = [e["type"] for e in env.mjai_log]
        assert "end_kyoku" in log_types
        assert "end_game" in log_types


# ===========================================================================
# apply_event: mjai_log recording (3P)
# ===========================================================================


class TestApplyEventMjaiLog3P:
    """Verify apply_event() records events in mjai_log for 3-player mode."""

    def test_start_game_clears_constructor_log(self):
        env = RiichiEnv(game_mode="3p-red-half")
        assert len(env.mjai_log) > 0

        env.apply_event({"type": "start_game"})
        assert len(env.mjai_log) == 1
        assert env.mjai_log[0]["type"] == "start_game"

    def test_events_accumulate_in_order(self):
        env = RiichiEnv(game_mode="3p-red-half")
        events = [
            {"type": "start_game"},
            _start_kyoku_3p(),
            {"type": "tsumo", "actor": 0, "pai": "3z"},
            {"type": "dahai", "actor": 0, "pai": "3z", "tsumogiri": True},
        ]
        for ev in events:
            env.apply_event(ev)

        log_types = [e["type"] for e in env.mjai_log]
        assert log_types == ["start_game", "start_kyoku", "tsumo", "dahai"]

    def test_pon_event_logged(self):
        env = RiichiEnv(game_mode="3p-red-half")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_3p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "3z"})
        env.apply_event({"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False})
        env.apply_event(
            {
                "type": "pon",
                "actor": 1,
                "target": 0,
                "pai": "1p",
                "consumed": ["1p", "1p"],
            }
        )

        log_types = [e["type"] for e in env.mjai_log]
        assert "pon" in log_types


# ===========================================================================
# observe_event: mjai_log recording
# ===========================================================================


class TestObserveEventMjaiLog:
    """Verify observe_event() also records events in mjai_log."""

    def test_observe_event_logs_events_4p(self):
        env = RiichiEnv(game_mode="default")
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(
            {
                **_start_kyoku_4p(),
                "tehais": [
                    _4P_TEHAIS[0],
                    ["?"] * 13,
                    ["?"] * 13,
                    ["?"] * 13,
                ],
            },
            0,
        )
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "5p"}, 0)
        env.observe_event(
            {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
            0,
        )

        log_types = [e["type"] for e in env.mjai_log]
        assert log_types == ["start_game", "start_kyoku", "tsumo", "dahai"]

    def test_observe_event_logs_events_3p(self):
        env = RiichiEnv(game_mode="3p-red-half")
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(
            {
                **_start_kyoku_3p(),
                "tehais": [
                    _3P_TEHAIS[0],
                    ["?"] * 13,
                    ["?"] * 13,
                ],
            },
            0,
        )
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "3z"}, 0)
        env.observe_event(
            {"type": "dahai", "actor": 0, "pai": "3z", "tsumogiri": True},
            0,
        )

        log_types = [e["type"] for e in env.mjai_log]
        assert log_types == ["start_game", "start_kyoku", "tsumo", "dahai"]

    def test_observe_event_clears_on_start_game(self):
        """observe_event should also clear stale log on start_game."""
        env = RiichiEnv(game_mode="default")
        assert len(env.mjai_log) > 0  # constructor log

        env.observe_event({"type": "start_game"}, 0)
        assert len(env.mjai_log) == 1
        assert env.mjai_log[0]["type"] == "start_game"


# ===========================================================================
# Consistency: apply_event vs step log
# ===========================================================================


class TestApplyEventLogConsistency:
    """Verify that apply_event log content is consistent with the input."""

    def test_tsumo_log_contains_pai_field(self):
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})

        tsumo_event = env.mjai_log[-1]
        assert tsumo_event["type"] == "tsumo"
        assert tsumo_event["actor"] == 0
        assert tsumo_event["pai"] == "5p"

    def test_dahai_log_contains_tsumogiri_field(self):
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        env.apply_event({"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True})

        dahai_event = env.mjai_log[-1]
        assert dahai_event["type"] == "dahai"
        assert dahai_event["actor"] == 0
        assert dahai_event["pai"] == "5p"
        assert dahai_event["tsumogiri"] is True

    def test_start_kyoku_log_contains_tehais(self):
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())

        kyoku_event = env.mjai_log[-1]
        assert kyoku_event["type"] == "start_kyoku"
        assert "tehais" in kyoku_event
        assert len(kyoku_event["tehais"]) == 4

    def test_multiple_start_game_resets_log(self):
        """Calling start_game multiple times should reset the log each time."""
        env = RiichiEnv(game_mode="default")
        env.apply_event({"type": "start_game"})
        env.apply_event(_start_kyoku_4p())
        env.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
        assert len(env.mjai_log) == 3

        # Second start_game should reset
        env.apply_event({"type": "start_game"})
        assert len(env.mjai_log) == 1
        assert env.mjai_log[0]["type"] == "start_game"
