import { Viewer } from './viewer';
import { Viewer3D } from './viewer_3d';
import { LiveViewer } from './live_viewer';
import { RiichiViewer } from './riichi_viewer';

export { Viewer, Viewer3D, LiveViewer, RiichiViewer };
export type { ViewerOptions, ViewerPosition, KyokuInfo, ViewerEventMap } from './types';

(window as any).RiichiEnvViewer = Viewer;
(window as any).RiichiEnv3DViewer = Viewer3D;
(window as any).RiichiEnvLiveViewer = LiveViewer;
(window as any).RiichiViewer = RiichiViewer;
