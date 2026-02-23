const TextToSVG = require('text-to-svg');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const fontPath = path.join(__dirname, '../assets/ZenAntiqueSoft-Regular.ttf');
const textToSVG = TextToSVG.loadSync(fontPath);

// Configuration
const fontSize = 36; // Larger font size for sprite quality, will be scaled down in CSS
const chars = '東西南北局一二三四';
const configs = [
    { suffix: '', color: 'white' }, // Only white as requested
];

// Special case: East Red
const special = [
    { char: '東', suffix: '_red', color: 'red' }
];

async function generateSprite() {
    const images = [];
    const mapping = {};
    let currentX = 0;
    const padding = 2; // Pixel padding between glyphs

    const glyphs = [];
    let maxHeight = 0;

    const margin = 1; // Minimal margin

    // Helper to process char (Pass 1)
    const prepareChar = (char, color, name) => {
        const opPath = textToSVG.font.getPath(char, 0, 0, fontSize);
        const bbox = opPath.getBoundingBox();

        // Add margin to bounding box
        const x1 = Math.floor(bbox.x1) - margin;
        const y1 = Math.floor(bbox.y1) - margin;
        const w = (Math.ceil(bbox.x2) - Math.floor(bbox.x1)) + (margin * 2);
        const h = (Math.ceil(bbox.y2) - Math.floor(bbox.y1)) + (margin * 2);
        const d = opPath.toPathData(2);

        // Create SVG string with expanded viewBox
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="${x1} ${y1} ${w} ${h}">
            <path d="${d}" fill="${color}" />
        </svg>`;

        // Content width (width without margins, for layout)
        const cw = Math.ceil(bbox.x2) - Math.floor(bbox.x1);

        if (h > maxHeight) maxHeight = h;
        glyphs.push({
            name,
            svg: Buffer.from(svg),
            w,
            h,
            cw // Store content width
        });
    };

    // Standard Chars
    for (const char of chars) {
        for (const config of configs) {
            prepareChar(char, config.color, `${char}${config.suffix}`);
        }
    }

    // Special Chars
    for (const s of special) {
        prepareChar(s.char, s.color, `${s.char}${s.suffix}`);
    }

    // Compose Images (Pass 2)
    for (const glyph of glyphs) {
        const topOffset = Math.floor((maxHeight - glyph.h) / 2);

        images.push({ input: glyph.svg, left: currentX, top: topOffset });

        // Mapping stores the rect on the sprite sheet.
        // Y coordinate is row-relative (0) + topOffset.
        mapping[glyph.name] = { x: currentX, y: topOffset, w: glyph.w, h: glyph.h, cw: glyph.cw };

        currentX += glyph.w + padding;
    }

    // Create Sprite
    const spriteBuffer = await sharp({
        create: {
            width: currentX,
            height: maxHeight,
            channels: 4,
            background: { r: 255, g: 255, b: 255, alpha: 0 } // White transparent to prevent dark halos
        }
    })
        .composite(images)
        .png({ compressionLevel: 9, quality: 100, force: true }) // Ensure PNG, full color for alpha
        .toBuffer();

    // Verify Output (optional, save to disk)
    await sharp(spriteBuffer).toFile(path.join(__dirname, '../assets/char_sprite.png'));

    // Generate TypeScript
    const base64 = spriteBuffer.toString('base64');
    const tsContent = `export const CHAR_SPRITE_BASE64 = "data:image/png;base64,${base64}";

export const CHAR_MAP: { [key: string]: { x: number, y: number, w: number, h: number, cw?: number } } = ${JSON.stringify(mapping, null, 4)};
`;

    fs.writeFileSync(path.join(__dirname, '../src/char_assets.ts'), tsContent);
    console.log(`Generated src/char_assets.ts (${tsContent.length} bytes)`);
}

generateSprite().catch(console.error);
