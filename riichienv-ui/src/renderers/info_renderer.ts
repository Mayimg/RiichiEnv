export class InfoRenderer {
    static renderPlayerInfo(
        player: any,
        index: number,
        viewpoint: number,
        currentActor: number,
        onViewpointChange: (idx: number) => void
    ): HTMLElement {
        const infoBox = document.createElement('div');
        infoBox.className = 'player-info-box';
        if (index === viewpoint) {
            infoBox.classList.add('active-viewpoint');
        }

        // Positioning: Absolute relative to pDiv
        Object.assign(infoBox.style, {
            position: 'absolute',
            top: '30px',
            left: '50%',
            transform: 'translateX(140px)',
            marginLeft: '0'
        });

        infoBox.innerHTML = `
            <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 4px; color: white;">
                P${index}
            </div>
        `;

        // Blinking Bar for Active Player
        if (index === currentActor) {
            const bar = document.createElement('div');
            bar.className = 'active-player-bar';
            infoBox.appendChild(bar);
        }

        infoBox.onclick = (e) => {
            e.stopPropagation(); // Prevent bubbling
            if (onViewpointChange) {
                onViewpointChange(index);
            }
        };

        return infoBox;
    }
}
