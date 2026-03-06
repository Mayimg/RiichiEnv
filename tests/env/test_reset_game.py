"""Regression tests: env.reset() must start a fresh game."""

import pytest

from riichienv import RiichiEnv


class TestResetGame:
    def test_scores_reset_to_defaults_4p(self):
        env = RiichiEnv(seed=42)
        env.reset(scores=[30000, 20000, 40000, 10000])
        assert env.scores() == [30000, 20000, 40000, 10000]

        # reset() with no args should restore starting scores
        env.reset()
        assert env.scores() == [25000, 25000, 25000, 25000]

    def test_scores_reset_to_defaults_3p(self):
        env = RiichiEnv(seed=42, game_mode="3p-red-east")
        env.reset(scores=[40000, 30000, 35000])
        assert env.scores() == [40000, 30000, 35000]

        env.reset()
        assert env.scores() == [35000, 35000, 35000]

    def test_round_state_reset_to_defaults(self):
        env = RiichiEnv(seed=42)
        env.reset(round_wind=1, oya=2, honba=3, kyotaku=5)
        assert env.round_wind == 1
        assert env.oya == 2
        assert env.honba == 3
        assert env.riichi_sticks == 5

        env.reset()
        assert env.round_wind == 0
        assert env.oya == 0
        assert env.honba == 0
        assert env.riichi_sticks == 0

    def test_scores_length_validation_4p(self):
        env = RiichiEnv(seed=42)
        with pytest.raises(ValueError, match="does not match"):
            env.reset(scores=[25000, 25000, 25000])

    def test_scores_length_validation_3p(self):
        env = RiichiEnv(seed=42, game_mode="3p-red-east")
        with pytest.raises(ValueError, match="does not match"):
            env.reset(scores=[35000, 35000, 35000, 35000])

    def test_scores_empty_list_rejected(self):
        env = RiichiEnv(seed=42)
        with pytest.raises(ValueError, match="does not match"):
            env.reset(scores=[])
