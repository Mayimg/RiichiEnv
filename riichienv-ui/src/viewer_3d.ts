import { BaseViewer } from './base_viewer';
import { createLayout3DConfig3P, createLayout3DConfig4P, type GameConfig, type LayoutConfig3D } from './config';
import { Renderer3D } from './renderers/renderer_3d';
import type { IRenderer } from './renderers/renderer_interface';
import type { MjaiEvent } from './types';

export class Viewer3D extends BaseViewer {
    /** Create a Viewer3D from an HTMLElement directly (no URL parsing, no containerId). */
    static fromElement(
        el: HTMLElement,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig3D,
    ): Viewer3D {
        Viewer3D._pendingLayout = layout;
        Viewer3D._pendingElement = el;
        try {
            return new Viewer3D('__fromElement__', log, initialStep, perspective, freeze, config, layout);
        } finally {
            Viewer3D._pendingElement = undefined;
            Viewer3D._pendingLayout = undefined;
        }
    }

    private static _pendingElement?: HTMLElement;
    private static _pendingLayout?: LayoutConfig3D;

    constructor(
        containerId: string,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig3D,
    ) {
        let el: HTMLElement;
        let effectiveInitialStep = initialStep;

        if (Viewer3D._pendingElement) {
            el = Viewer3D._pendingElement;
        } else {
            const found = document.getElementById(containerId);
            if (!found) throw new Error(`Container #${containerId} not found`);
            el = found;

            if (typeof initialStep !== 'number') {
                const urlParams = new URLSearchParams(window.location.search);
                const eventStepParam = urlParams.get('eventStep');
                if (eventStepParam) {
                    const parsed = parseInt(eventStepParam, 10);
                    if (!Number.isNaN(parsed)) effectiveInitialStep = parsed;
                }
            }
        }

        Viewer3D._pendingLayout = layout;

        super({
            container: el,
            log,
            initialStep: effectiveInitialStep,
            perspective,
            freeze,
            config,
        });

        Viewer3D._pendingLayout = undefined;
    }

    protected getLayoutInfo(gc: GameConfig, _log: MjaiEvent[]) {
        const lc =
            Viewer3D._pendingLayout ?? (gc.playerCount === 3 ? createLayout3DConfig3P() : createLayout3DConfig4P());
        return {
            contentWidth: lc.contentWidth,
            contentHeight: lc.contentHeight,
            viewAreaWidth: lc.viewAreaWidth,
            viewAreaHeight: lc.viewAreaHeight,
            sidebarStyle: 'grid' as const,
        };
    }

    protected createRenderer(viewArea: HTMLElement, gc: GameConfig, _log: MjaiEvent[]): IRenderer {
        const lc =
            Viewer3D._pendingLayout ?? (gc.playerCount === 3 ? createLayout3DConfig3P() : createLayout3DConfig4P());
        return new Renderer3D(viewArea, lc);
    }
}
