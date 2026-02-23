import { TILES } from '../tiles';

export class TileRenderer {
    static getTileHtml(tileStr: string): string {
        if (tileStr === 'back') {
            const svg = TILES['back'] || TILES['blank'];
            return `<div class="tile-layer"><div class="tile-bg">${svg}</div></div>`;
        }

        const frontSvg = TILES['front'] || '';
        let fgSvg = TILES[tileStr];
        if (!fgSvg) {
            fgSvg = TILES['blank'] || '';
        }

        return `
            <div class="tile-layer">
                <div class="tile-bg">${frontSvg}</div>
                <div class="tile-fg">${fgSvg}</div>
            </div>
        `;
    }
}
