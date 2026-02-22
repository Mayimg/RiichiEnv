import riichienv


def test_shanten_3p_differs_from_4p():
    """1111m111122233z: 4P=1, 3P=2.

    In 4P, the only path to tenpai is discard 1z + draw 2m or 3m
    (forming taatsu with the leftover 1m). In 3P, 2m/3m don't exist,
    so this path is unavailable, making shanten one higher.
    """
    tiles, _ = riichienv.parse_hand("1111m111122233z")
    assert riichienv.calculate_shanten(tiles) == 1
    assert riichienv.calculate_shanten_3p(tiles) == 2


def test_shanten_3p_complete_koutsu():
    """Complete hand: 4 koutsu + 1 pair (valid 3P tiles only)."""
    tiles, _ = riichienv.parse_hand("111m111z222z333z44z")
    assert riichienv.calculate_shanten(tiles) == -1
    assert riichienv.calculate_shanten_3p(tiles) == -1


def test_shanten_3p_complete_mixed():
    """Complete hand with pinzu shuntsu + honor koutsu + pair."""
    tiles, _ = riichienv.parse_hand("123456789p11222z")
    assert riichienv.calculate_shanten(tiles) == -1
    assert riichienv.calculate_shanten_3p(tiles) == -1


def test_shanten_3p_complete_souzu_shuntsu():
    """Complete hand with souzu shuntsu (sequences valid in 3P)."""
    tiles, _ = riichienv.parse_hand("111m123456789s11z")
    assert riichienv.calculate_shanten(tiles) == -1
    assert riichienv.calculate_shanten_3p(tiles) == -1


def test_shanten_3p_tenpai_kokushi():
    """Kokushi tenpai: valid in both 3P and 4P."""
    tiles, _ = riichienv.parse_hand("19m19p19s1234567z")
    assert riichienv.calculate_shanten(tiles) == 0
    assert riichienv.calculate_shanten_3p(tiles) == 0


def test_shanten_3p_tenpai_normal():
    """Normal tenpai: 3 koutsu + 1 shuntsu + tanki wait."""
    # 111m999m123p789s + tanki 1z
    tiles, _ = riichienv.parse_hand("111m999m123p789s1z")
    assert riichienv.calculate_shanten(tiles) == 0
    assert riichienv.calculate_shanten_3p(tiles) == 0


def test_shanten_3p_tenpai_chiitoitsu():
    """Chiitoitsu tenpai with valid 3P tiles."""
    tiles, _ = riichienv.parse_hand("1199m1199p1199s1z")
    assert riichienv.calculate_shanten(tiles) == 0
    assert riichienv.calculate_shanten_3p(tiles) == 0


def test_shanten_3p_manzu_terminals():
    """1m and 9m can only form koutsu/pair, not sequences, in both 3P and 4P."""
    # 11m99m + 123p + 456s + 111z = 2 pairs + 2 shuntsu + 1 koutsu = tenpai
    tiles, _ = riichienv.parse_hand("11m99m123p456s111z")
    assert riichienv.calculate_shanten(tiles) == 0
    assert riichienv.calculate_shanten_3p(tiles) == 0


def test_shanten_3p_iishanten():
    """Iishanten (shanten=1) with valid 3P tiles."""
    # 111m 999m 123p 1s 3s 7z: 3 mentsu + kanchan 13s, no pair
    tiles, _ = riichienv.parse_hand("111m999m123p13s7z")
    assert riichienv.calculate_shanten(tiles) == 1
    assert riichienv.calculate_shanten_3p(tiles) == 1


def test_shanten_3p_both_manzu_maxed():
    """When both 1m and 9m are at 4 copies, 3P shanten can differ from 4P.

    The 4P lookup gives adjacency credit for both leftover 1m and 9m tiles,
    but in 3P neither can form sequences.
    """
    tiles, _ = riichienv.parse_hand("11119999m22345s")
    assert riichienv.calculate_shanten(tiles) == 1
    assert riichienv.calculate_shanten_3p(tiles) == 2


def test_shanten_3p_overflow_honors():
    """Overflow case: all 7 honor slots occupied + both 1m and 9m present.

    In this case relocation to zipai slots is partially limited, but the
    result is still correct because kokushi dominates for such scattered hands.
    """
    tiles, _ = riichienv.parse_hand("1111m9m1234567z")
    assert riichienv.calculate_shanten(tiles) == 3
    assert riichienv.calculate_shanten_3p(tiles) == 3


def test_shanten_3p_consistency_with_4p():
    """For valid 3P hands without heavy 1m/9m, 3P and 4P shanten should match."""
    test_hands = [
        ("19m19p19s1234567z", 0),  # Kokushi tenpai
        ("1199m1199p1199s1z", 0),  # Chiitoitsu tenpai
        ("111m999m111p11z", -1),  # Complete: 3 koutsu + koutsu + pair
        ("111m123456789p1z", 0),  # Tenpai: koutsu + 3 shuntsu + tanki
        ("999m111222333z1p", 0),  # Tenpai: 3 koutsu + koutsu + tanki
        ("11m99m11p99p11s99s1z", 0),  # Chiitoitsu tenpai (terminals)
        ("111999m111999p1z", 0),  # Tenpai: 4 koutsu + tanki
        ("19m147p258s12345z", 5),  # Scattered hand
    ]
    for hand_str, expected in test_hands:
        tiles, _ = riichienv.parse_hand(hand_str)
        shanten_4p = riichienv.calculate_shanten(tiles)
        shanten_3p = riichienv.calculate_shanten_3p(tiles)
        assert shanten_4p == shanten_3p, f"{hand_str}: 4P={shanten_4p} != 3P={shanten_3p}"
        assert shanten_4p == expected, f"{hand_str}: expected {expected}, got {shanten_4p}"
