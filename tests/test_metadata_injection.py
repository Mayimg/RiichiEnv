import json
import os

import pytest

from riichienv.visualizer.viewer import MetadataInjector


# Load the log once for all tests
@pytest.fixture(scope="module")
def enriched_log():
    # Use the .bak file as requested (now renamed)
    path = os.path.join(os.path.dirname(__file__), "../riichienv-ui/example_before_injection.jsonl")
    if not os.path.exists(path):
        # Fallback if running from root relative path issues
        path = "riichienv-ui/example_before_injection.jsonl"

    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    injector = MetadataInjector(events)
    return injector.process()


def find_kyoku_events(events, bakaze_char, kyoku_num, honba_num):
    """Finds all events for a specific kyoku."""
    kyoku_events = []
    in_kyoku = False

    for ev in events:
        if ev["type"] == "start_kyoku":
            if ev.get("bakaze") == bakaze_char and ev.get("kyoku") == kyoku_num and ev.get("honba") == honba_num:
                in_kyoku = True
                kyoku_events = [ev]
            else:
                in_kyoku = False
        elif in_kyoku:
            kyoku_events.append(ev)
            if ev["type"] == "end_kyoku":
                in_kyoku = False
                yield kyoku_events


def get_hora_score(kyoku_events):
    """Extracts score from the first hora event in the kyoku."""
    for ev in kyoku_events:
        if ev["type"] == "hora":
            return ev.get("meta", {}).get("score")
    return None


def test_e2_1_fu_correctness(enriched_log):
    """
    E2-1 (East 2, 1 Honba):
    Expectation: 3 Han 40 Fu.
    Issue Fixed: Was calculated as 30 Fu due to ID overflow and sorting bugs.
    """
    kyokus = list(find_kyoku_events(enriched_log, "E", 2, 1))
    assert len(kyokus) > 0, "E2-1 not found in log"

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in E2-1"
    assert score["fu"] == 40, f"Expected 40 Fu for E2-1, got {score['fu']}"
    assert score["han"] == 3


def test_e4_0_ippatsu_correctness(enriched_log):
    """
    E4-0 (East 4, 0 Honba):
    Expectation: 4 Han 30 Fu (No Ippatsu).
    Issue Fixed: Ippatsu was wrongly awarded (5 Han) because it wasn't cleared on subsequent discard.
    """
    kyokus = list(find_kyoku_events(enriched_log, "E", 4, 0))
    assert len(kyokus) > 0, "E4-0 not found in log"

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in E4-0"

    # Check Ippatsu (ID 30) is NOT present
    yaku_ids = score["yaku"]
    assert 30 not in yaku_ids, f"Ippatsu (30) should not be present in E4-0. Yaku: {yaku_ids}"

    # Check Han/Fu
    assert score["han"] == 4, f"Expected 4 Han for E4-0, got {score['han']}"
    assert score["fu"] == 30


def test_s4_0_fu_correctness(enriched_log):
    """
    S4-0 (South 4, 0 Honba):
    Expectation: 40 Fu.
    Issue Fixed: Previous fix for S4-0 regarding 136-tile offsets.
    """
    kyokus = list(find_kyoku_events(enriched_log, "S", 4, 0))
    if not kyokus:
        pytest.skip("S4-0 not found in log")
        return

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in S4-0"
    assert score["fu"] == 40, f"Expected 40 Fu for S4-0, got {score['fu']}"


def test_waits_calculation(enriched_log):
    """
    Verify that waits are calculated and injected into Dahai events.
    """
    found_waits = False
    for ev in enriched_log:
        if ev["type"] == "dahai":
            if "waits" in ev.get("meta", {}):
                found_waits = True
                waits = ev["meta"]["waits"]
                assert isinstance(waits, list)
                break

    assert found_waits, "No waits calculated in any Dahai event"
