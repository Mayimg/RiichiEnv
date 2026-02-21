import lzma
import time
import glob
from typing import Iterator

import tqdm

from riichienv import HandEvaluator, MjSoulReplay, Kyoku
from mjsoul_parser import MjsoulPaifuParser, Paifu

YAKUMAN_IDS = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50]
# Yaku IDs excluded from comparison:
# 31=Dora, 32=Aka, 33=Ura
EXCLUDED_YAKU_IDS = {31, 32, 33}
TARGET_FILE_PATTERN = "/data/mjsoul/mahjong_game_record_4p_*/*.bin.xz"


def iter_game_kyoku(paifu: Paifu) -> Iterator[Kyoku]:
    game = MjSoulReplay.from_dict(paifu.data)
    for kyoku in game.take_kyokus():
        yield kyoku


def main():
    total_agari = 0
    total_mismatch = 0
    t_riichienv_py = 0

    target_files = list(glob.glob(TARGET_FILE_PATTERN))
    target_files = list(target_files)[:100]
    for path in tqdm.tqdm(target_files, desc="Processing files", ncols=100):
        with lzma.open(path, "rb") as f:
            data = f.read()
            paifu: Paifu = MjsoulPaifuParser.to_dict(data)

        for k, kyoku in enumerate(iter_game_kyoku(paifu)):
            for ctx in kyoku.take_win_result_contexts():
                total_agari += 1

                expected_yakuman = (
                    len(set(ctx.expected_yaku) & set(YAKUMAN_IDS)) > 0
                )

                t0 = time.time()
                res_r_py = HandEvaluator(
                    ctx.tiles,
                    ctx.melds,
                ).calc(
                    ctx.agari_tile,
                    ctx.dora_indicators,
                    ctx.conditions,
                    ctx.ura_indicators,
                )
                t_riichienv_py += time.time() - t0

                # Compare yaku sets excluding dora-only yaku
                sim_yaku = set(res_r_py.yaku) - EXCLUDED_YAKU_IDS
                exp_yaku = set(ctx.expected_yaku) - EXCLUDED_YAKU_IDS

                mismatch = False
                if expected_yakuman:
                    if not res_r_py.yakuman:
                        mismatch = True
                elif not res_r_py.is_win:
                    mismatch = True
                else:
                    if sim_yaku != exp_yaku:
                        mismatch = True
                    if res_r_py.fu != ctx.expected_fu:
                        mismatch = True

                if mismatch:
                    total_mismatch += 1
                    if total_mismatch <= 30:
                        print(
                            f"MISMATCH: {paifu.header['uuid']} "
                            f"kyoku={k} "
                            f"han=(sim={res_r_py.han}, exp={ctx.expected_han}) "
                            f"fu=(sim={res_r_py.fu}, exp={ctx.expected_fu})"
                        )
                        print(
                            f"  sim_yaku={sorted(res_r_py.yaku)} "
                            f"exp_yaku={sorted(ctx.expected_yaku)}"
                        )
                        print(
                            f"  is_win={res_r_py.is_win} yakuman={res_r_py.yakuman} "
                            f"tsumo={ctx.conditions.tsumo} "
                            f"tiles={ctx.tiles} melds={ctx.melds} "
                            f"agari={ctx.agari_tile}"
                        )

    print()
    print(f"Total agari: {total_agari}")
    print(f"Total mismatch: {total_mismatch}")
    print(f"HandEvaluator time: {t_riichienv_py:.2f}s")


if __name__ == "__main__":
    main()
