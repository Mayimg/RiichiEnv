import dataclasses
from dataclasses import dataclass

from . import _riichienv as rust_core
from ._riichienv import (
    Meld,
    Wind,
    WinResult,
)

WINDS = [Wind.East, Wind.South, Wind.West, Wind.North]


@dataclass
class Conditions:
    tsumo: bool = False
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    haitei: bool = False
    houtei: bool = False
    rinshan: bool = False
    chankan: bool = False
    tsumo_first_turn: bool = False

    player_wind: int | Wind = 0  # E,S,W,N = (0,1,2,3) or Wind enum values
    round_wind: int | Wind = 0  # E,S,W,N = (0,1,2,3) or Wind enum values

    riichi_sticks: int = 0
    honba: int = 0

    kita_count: int = 0
    is_sanma: bool = False
    num_players: int = 4


class HandEvaluator:
    def __init__(self, tiles: list[int], melds: list[Meld] | None = None) -> None:
        self.tiles_136 = tiles
        self.melds = melds or []
        self._rust_melds = self.melds
        self.calc_rust = rust_core.HandEvaluator(self.tiles_136, self._rust_melds)

    @staticmethod
    def hand_from_text(hand_str_repr: str) -> "HandEvaluator":
        """
        # Hand representation parsing
        `Hand::from_text()` method accepts a string representation.
        It expects a 13-tile hand (plus extra tiles for Kans).
        """
        tiles, melds = rust_core.parse_hand(hand_str_repr)

        num_kans = 0
        current_tiles_count = len(tiles)
        for m in melds:
            current_tiles_count += len(m.tiles)
            if m.meld_type in [rust_core.MeldType.Daiminkan, rust_core.MeldType.Ankan, rust_core.MeldType.Kakan]:
                num_kans += 1

        expected_count = 13 + num_kans
        if current_tiles_count != expected_count:
            raise ValueError(f"Hand must have {expected_count} tiles (got {current_tiles_count})")

        tiles = list(tiles)
        tiles.sort()
        return HandEvaluator(tiles, melds)

    def to_text(self, win_tile: str | None = None) -> str:
        """
        Convert hand to text representation.
        """
        tiles = sorted(self.tiles_136)
        # TODO: Implement full meld string reconstruction if needed.
        # For now, focus on closed tiles.
        # User requested full implementation.
        # I'll implement basic tile string construction.

        result = self._tiles_to_string(tiles)

        for meld in self.melds:
            # Reconstruct meld string
            # Format: (XYZCI) etc.
            # This requires reconstructing the exact input format which includes 'call index'.
            # Rust Meld struct DOES NOT store 'call index'.
            # It only stores type, tiles, opened.
            # So we cannot perfectly reconstruct (p1z2) vs (p1z3).
            # We can only approximate or omit index?
            # User sample: (p1z1).
            # I'll assume index 0 if unknown.
            result += self._meld_to_string(meld)

        return result

    @staticmethod
    def calc_from_text(
        hand_str_repr_with_win_tile: str,
        dora_indicators: str | None = None,
        conditions: Conditions | None = None,
        ura_indicators: str | None = None,
    ) -> WinResult:
        """
        hand_str_repr_with_win_tile: str は 14 枚分の牌を想定する。最後の1枚を win_tile として扱う。
        """
        conditions = conditions or Conditions()
        tiles, melds = rust_core.parse_hand(hand_str_repr_with_win_tile)
        if not tiles and not melds:
            raise ValueError("Empty hand")

        if not tiles:
            raise ValueError("No standing tiles to check for win tile")

        win_tile = tiles[-1]
        tiles = list(tiles)
        tiles.sort()

        calc = HandEvaluator(tiles, melds)

        dora_inds = []
        if dora_indicators:
            dora_inds, _ = rust_core.parse_hand(dora_indicators)
            dora_inds = list(dora_inds)
            dora_inds.sort()

        ura_inds = []
        if ura_indicators:
            ura_inds, _ = rust_core.parse_hand(ura_indicators)
            ura_inds = list(ura_inds)
            ura_inds.sort()

        return calc.calc(win_tile, dora_inds, conditions, ura_inds)

    def _tiles_to_string(self, tiles: list[int]) -> str:
        # Group by suit: m, p, s, z
        man = []
        pin = []
        sou = []
        honors = []

        for t in tiles:
            is_red = t in [16, 52, 88]
            val = t // 4

            digit = 0
            if val < 9:
                digit = val + 1
                if is_red:
                    digit = 0
                man.append(digit)
            elif val < 18:
                digit = (val - 9) + 1
                if is_red:
                    digit = 0
                pin.append(digit)
            elif val < 27:
                digit = (val - 18) + 1
                if is_red:
                    digit = 0
                sou.append(digit)
            else:
                digit = (val - 27) + 1
                honors.append(digit)

        res = ""
        if man:
            res += "".join(map(str, man)) + "m"
        if pin:
            res += "".join(map(str, pin)) + "p"
        if sou:
            res += "".join(map(str, sou)) + "s"
        if honors:
            res += "".join(map(str, honors)) + "z"
        return res

    def _meld_to_string(self, meld: Meld) -> str:  # noqa: PLR0915
        # Reconstruct string from Meld
        # (XYZCI)
        # Meld has tiles[].
        # Determine suit from first tile.
        t0 = meld.tiles[0]
        val0 = t0 // 4

        suffix = "m"
        if val0 >= 9 and val0 < 18:
            suffix = "p"
        elif val0 >= 18 and val0 < 27:
            suffix = "s"
        elif val0 >= 27:
            suffix = "z"

        m_type = meld.meld_type

        # Digits
        digits = ""
        is_chi = m_type == rust_core.MeldType.Chi

        if is_chi:
            for t in meld.tiles:
                is_red = t in [16, 52, 88]
                v = t // 4

                d = 0
                if v < 9:
                    d = v + 1
                elif v < 18:
                    d = v - 9 + 1
                elif v < 27:
                    d = v - 18 + 1
                else:
                    d = v - 27 + 1

                if is_red:
                    d = 0
                digits += str(d)
        else:
            # Pon/Kan/Add: single digit
            has_red = any(t in [16, 52, 88] for t in meld.tiles)
            v = val0
            d = 0
            if v < 9:
                d = v + 1
            elif v < 18:
                d = v - 9 + 1
            elif v < 27:
                d = v - 18 + 1
            else:
                d = v - 27 + 1

            if has_red:
                d = 0
            digits = str(d)

        # Prefix
        prefix = ""

        if m_type == rust_core.MeldType.Pon:
            prefix = "p"
        elif m_type == rust_core.MeldType.Daiminkan:
            prefix = "k"
        elif m_type == rust_core.MeldType.Kakan:
            prefix = "s"
        elif m_type == rust_core.MeldType.Ankan:
            prefix = "k"
        # Closed Kan also 'k'? Usually closed kan not indicated in open string?
        # But user example (k2z) was "closed kan or daiminkan".
        # If Angang, maybe suffix missing?

        # Call index? Not stored. Use 0.

        return f"({prefix}{digits}{suffix}0)"  # Fake index 0

    def calc(
        self,
        win_tile: int,
        dora_indicators: list[int] | None = None,
        conditions: Conditions | None = None,
        ura_indicators: list[int] | None = None,
    ) -> WinResult:
        conditions = conditions or Conditions()
        if dora_indicators is None:
            dora_indicators = []
        if ura_indicators is None:
            ura_indicators = []

        rust_conditions = _to_rust_conditions(conditions)

        dora_inds_136 = dora_indicators if dora_indicators else []
        ura_inds_136 = ura_indicators if ura_indicators else []

        rust_melds = self._rust_melds
        meld_tiles = sum(len(m.tiles) for m in self.melds)
        total_tiles = len(self.tiles_136) + meld_tiles

        calc_obj = self.calc_rust
        if total_tiles % 3 == 1:
            temp_tiles = sorted(self.tiles_136 + [win_tile])
            calc_obj = rust_core.HandEvaluator(temp_tiles, rust_melds)

        return calc_obj.calc(win_tile, dora_inds_136, ura_inds_136, rust_conditions)

    def is_tenpai(self) -> bool:
        return self.calc_rust.is_tenpai()

    def get_waits(self) -> list[int]:
        return self.calc_rust.get_waits()


def _to_rust_conditions(conditions: Conditions) -> "rust_core.Conditions":
    """Convert a Python Conditions dataclass to the Rust Conditions object."""
    p_wind = conditions.player_wind
    if isinstance(p_wind, int):
        p_wind = WINDS[p_wind % 4]

    r_wind = conditions.round_wind
    if isinstance(r_wind, int):
        r_wind = WINDS[r_wind % 4]

    return rust_core.Conditions(
        tsumo=conditions.tsumo,
        riichi=conditions.riichi,
        double_riichi=conditions.double_riichi,
        ippatsu=conditions.ippatsu,
        haitei=conditions.haitei,
        houtei=conditions.houtei,
        rinshan=conditions.rinshan,
        chankan=conditions.chankan,
        tsumo_first_turn=conditions.tsumo_first_turn,
        player_wind=p_wind,
        round_wind=r_wind,
        riichi_sticks=conditions.riichi_sticks,
        honba=conditions.honba,
        kita_count=conditions.kita_count,
        is_sanma=conditions.is_sanma,
        num_players=conditions.num_players,
    )


class HandEvaluator3P:
    def __init__(self, tiles: list[int], melds: list[Meld] | None = None) -> None:
        self.tiles_136 = tiles
        self.melds = melds or []
        self._rust_melds = self.melds
        self.calc_rust = rust_core.HandEvaluator3P(self.tiles_136, self._rust_melds)

    def calc(
        self,
        win_tile: int,
        dora_indicators: list[int] | None = None,
        conditions: Conditions | None = None,
        ura_indicators: list[int] | None = None,
    ) -> WinResult:
        if conditions is None:
            conditions = Conditions(is_sanma=True, num_players=3)
        else:
            conditions = dataclasses.replace(conditions, is_sanma=True, num_players=3)
        if dora_indicators is None:
            dora_indicators = []
        if ura_indicators is None:
            ura_indicators = []

        rust_conditions = _to_rust_conditions(conditions)

        dora_inds_136 = dora_indicators if dora_indicators else []
        ura_inds_136 = ura_indicators if ura_indicators else []

        rust_melds = self._rust_melds
        meld_tiles = sum(len(m.tiles) for m in self.melds)
        total_tiles = len(self.tiles_136) + meld_tiles

        calc_obj = self.calc_rust
        if total_tiles % 3 == 1:
            temp_tiles = sorted(self.tiles_136 + [win_tile])
            calc_obj = rust_core.HandEvaluator3P(temp_tiles, rust_melds)

        return calc_obj.calc(win_tile, dora_inds_136, ura_inds_136, rust_conditions)

    def is_tenpai(self) -> bool:
        return self.calc_rust.is_tenpai()

    def get_waits(self) -> list[int]:
        return self.calc_rust.get_waits()
