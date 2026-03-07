import { YAKU_MAP } from './constants';
import type { GameState } from './game_state';
import type { KyokuKeyEvent, KyokuPlayerAction, KyokuResult, KyokuSummary, KyokuWinner, MjaiEvent } from './types';

const MELD_TYPES = new Set(['chi', 'pon', 'daiminkan', 'ankan', 'kakan']);

function meldLabel(type: string): string {
    switch (type) {
        case 'chi': return 'Chi';
        case 'pon': return 'Pon';
        case 'daiminkan': case 'ankan': case 'kakan': return 'Kan';
        default: return type;
    }
}

/**
 * Compute summaries for all kyoku rounds by scanning events.
 * This is lightweight (no game state walking).
 */
export function computeKyokuSummaries(gameState: GameState): KyokuSummary[] {
    const events = gameState.events;
    const kyokus = gameState.kyokus;
    const pc = gameState.config.playerCount;

    return kyokus.map((k, idx) => {
        const startIdx = k.index;
        const endIdx = idx + 1 < kyokus.length ? kyokus[idx + 1].index : events.length;

        const playerActions: KyokuPlayerAction[] = Array.from({ length: pc }, () => ({
            riichi: false,
            tenpai: false,
            houjuu: false,
            hora: false,
            tsumo: false,
            meldTypes: [],
        }));

        let result: KyokuResult | null = null;
        let endScores = [...k.scores];
        let deltas = Array(pc).fill(0);

        for (let i = startIdx; i < endIdx; i++) {
            const evt = events[i];

            switch (evt.type) {
                case 'reach':
                case 'reach_accepted':
                    if (evt.actor !== undefined) {
                        playerActions[evt.actor].riichi = true;
                    }
                    break;

                case 'chi':
                case 'pon':
                case 'daiminkan':
                case 'ankan':
                case 'kakan':
                    if (evt.actor !== undefined) {
                        playerActions[evt.actor].meldTypes.push(evt.type);
                    }
                    break;

                case 'hora':
                    if (evt.actor !== undefined) {
                        const isTsumo = evt.actor === evt.target;
                        playerActions[evt.actor].hora = true;
                        if (isTsumo) playerActions[evt.actor].tsumo = true;
                        if (!isTsumo && evt.target !== undefined) {
                            playerActions[evt.target].houjuu = true;
                        }
                    }
                    break;

                case 'ryukyoku':
                    if (evt.deltas) {
                        deltas = evt.deltas;
                    }
                    if (evt.scores) {
                        endScores = evt.scores;
                    }
                    // Mark tenpai players (positive delta in ryukyoku)
                    if (evt.deltas) {
                        for (let p = 0; p < pc; p++) {
                            if (evt.deltas[p] > 0) {
                                playerActions[p].tenpai = true;
                            }
                        }
                    }
                    result = { type: 'ryukyoku', reason: evt.reason };
                    break;

                case 'end_kyoku':
                    if (evt.meta?.results && evt.meta.results.length > 0) {
                        const winners: KyokuWinner[] = evt.meta.results.map((r: any) => ({
                            actor: r.actor,
                            target: r.target,
                            isTsumo: r.actor === r.target,
                            points: r.score?.points ?? 0,
                            han: r.score?.han ?? 0,
                            fu: r.score?.fu ?? 0,
                            yaku: (r.score?.yaku ?? []).map((id: number) => YAKU_MAP[id] || `Yaku#${id}`),
                        }));
                        result = { type: 'hora', winners };

                        // Extract deltas from hora events
                        for (let j = i - 1; j >= startIdx; j--) {
                            if (events[j].type === 'hora' && events[j].deltas) {
                                deltas = events[j].deltas;
                                break;
                            }
                        }
                    } else if (evt.meta?.ryukyoku) {
                        result = { type: 'ryukyoku', reason: evt.meta.ryukyoku.reason };
                        if (evt.meta.ryukyoku.deltas) deltas = evt.meta.ryukyoku.deltas;
                        if (evt.meta.ryukyoku.scores) endScores = evt.meta.ryukyoku.scores;
                    }

                    // Compute end scores from deltas if not set from ryukyoku
                    if (deltas.some((d: number) => d !== 0)) {
                        endScores = k.scores.map((s, pi) => s + deltas[pi]);
                    }
                    break;
            }
        }

        // Mark hora winners as tenpai
        for (let p = 0; p < pc; p++) {
            if (playerActions[p].hora) {
                playerActions[p].tenpai = true;
            }
        }

        return {
            index: idx,
            round: k.round,
            honba: k.honba,
            startScores: [...k.scores],
            endScores,
            deltas,
            result,
            playerActions,
        };
    });
}

/**
 * Compute key events within a kyoku by scanning events and walking game state.
 * This is heavier as it replays game state for tenpai tracking.
 */
export function computeKyokuKeyEvents(gameState: GameState, kyokuIndex: number): KyokuKeyEvent[] {
    const events = gameState.events;
    const kyokus = gameState.kyokus;
    const pc = gameState.config.playerCount;

    if (kyokuIndex < 0 || kyokuIndex >= kyokus.length) return [];

    const startIdx = kyokus[kyokuIndex].index;
    const endIdx = kyokuIndex + 1 < kyokus.length ? kyokus[kyokuIndex + 1].index : events.length;

    const keyEvents: KyokuKeyEvent[] = [];

    // Phase 1: Scan events for simple key events
    for (let i = startIdx; i < endIdx; i++) {
        const evt = events[i];
        switch (evt.type) {
            case 'reach':
                if (evt.actor !== undefined && (!evt.step || evt.step === '1' || evt.step === 1)) {
                    keyEvents.push({ step: i, type: 'reach', actor: evt.actor, label: 'Riichi' });
                }
                break;
            case 'chi':
            case 'pon':
            case 'daiminkan':
            case 'ankan':
            case 'kakan':
                if (evt.actor !== undefined) {
                    keyEvents.push({
                        step: i,
                        type: evt.type,
                        actor: evt.actor,
                        label: meldLabel(evt.type),
                        detail: tileSummary(evt),
                    });
                }
                break;
            case 'hora':
                if (evt.actor !== undefined) {
                    const isTsumo = evt.actor === evt.target;
                    keyEvents.push({
                        step: i,
                        type: 'hora',
                        actor: evt.actor,
                        label: isTsumo ? 'Tsumo' : 'Ron',
                    });
                }
                break;
            case 'ryukyoku':
                keyEvents.push({
                    step: i,
                    type: 'ryukyoku',
                    actor: -1,
                    label: 'Draw',
                    detail: evt.reason,
                });
                break;
        }
    }

    // Phase 2: Walk game state for tenpai tracking
    const savedCursor = gameState.cursor;

    gameState.jumpTo(startIdx + 1); // After start_kyoku
    const prevWaits: (string | undefined)[] = Array(pc).fill(undefined);

    for (let step = startIdx + 1; step < endIdx; step++) {
        const evt = events[step];

        // Only process dahai - that's when waits are recomputed
        if (evt.type === 'dahai' || evt.type === 'reach_accepted') {
            gameState.jumpTo(step + 1); // State after this event
            const state = gameState.getState();

            for (let p = 0; p < pc; p++) {
                const waits = state.players[p].waits;
                const waitsKey = waits?.sort().join(',') ?? '';
                const prevKey = prevWaits[p] ?? '';

                if (prevKey === '' && waitsKey !== '') {
                    keyEvents.push({ step, type: 'tenpai', actor: p, label: 'Tenpai', detail: waits!.join(' ') });
                } else if (prevKey !== '' && waitsKey === '') {
                    keyEvents.push({ step, type: 'tenpai_lost', actor: p, label: 'Lost Tenpai' });
                } else if (prevKey !== '' && waitsKey !== '' && prevKey !== waitsKey) {
                    keyEvents.push({ step, type: 'wait_change', actor: p, label: 'Wait Change', detail: waits!.join(' ') });
                }

                prevWaits[p] = waitsKey;
            }
        }
    }

    // Restore cursor
    gameState.jumpTo(savedCursor);

    // Sort by step
    keyEvents.sort((a, b) => a.step - b.step);

    return keyEvents;
}

function tileSummary(evt: MjaiEvent): string {
    const tiles = evt.consumed ? [...evt.consumed] : [];
    if (evt.pai) tiles.push(evt.pai);
    return tiles.join(' ');
}
