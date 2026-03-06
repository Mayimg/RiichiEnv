import {
    GameConfig, LayoutConfig,
    createLayoutConfig4P, createLayoutConfig3P,
} from './config';
import { Renderer2D } from './renderers/renderer_2d';
import { IRenderer } from './renderers/renderer_interface';
import { MjaiEvent } from './types';
import { BaseViewer } from './base_viewer';

export class Viewer extends BaseViewer {
    /** Create a Viewer from an HTMLElement directly (no URL parsing, no containerId). */
    static fromElement(
        el: HTMLElement,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig
    ): Viewer {
        Viewer._pendingLayout = layout;
        Viewer._pendingElement = el;
        const v = new Viewer('__fromElement__', log, initialStep, perspective, freeze, config, layout);
        Viewer._pendingElement = undefined;
        return v;
    }

    private static _pendingElement?: HTMLElement;
    private static _pendingLayout?: LayoutConfig;

    constructor(
        containerId: string,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig
    ) {
        let el: HTMLElement;
        let effectiveInitialStep = initialStep;

        if (Viewer._pendingElement) {
            el = Viewer._pendingElement;
        } else {
            const found = document.getElementById(containerId);
            if (!found) throw new Error(`Container #${containerId} not found`);
            el = found;

            if (typeof initialStep !== 'number') {
                const urlParams = new URLSearchParams(window.location.search);
                const eventStepParam = urlParams.get('eventStep');
                if (eventStepParam) {
                    const parsed = parseInt(eventStepParam, 10);
                    if (!isNaN(parsed)) effectiveInitialStep = parsed;
                }
            }
        }

        Viewer._pendingLayout = layout;

        super({
            container: el,
            log,
            initialStep: effectiveInitialStep,
            perspective,
            freeze,
            config,
        });

        Viewer._pendingLayout = undefined;
    }

    protected getLayoutInfo(gc: GameConfig, log: MjaiEvent[]) {
        const lc = Viewer._pendingLayout ?? (gc.playerCount === 3 ? createLayoutConfig3P() : createLayoutConfig4P());
        return {
            contentWidth: lc.contentWidth,
            contentHeight: lc.contentHeight,
            viewAreaWidth: lc.viewAreaSize,
            viewAreaHeight: lc.viewAreaSize,
            sidebarStyle: 'column' as const,
        };
    }

    protected createRenderer(viewArea: HTMLElement, gc: GameConfig, log: MjaiEvent[]): IRenderer {
        const lc = Viewer._pendingLayout ?? (gc.playerCount === 3 ? createLayoutConfig3P() : createLayoutConfig4P());
        return new Renderer2D(viewArea, lc);
    }
}
