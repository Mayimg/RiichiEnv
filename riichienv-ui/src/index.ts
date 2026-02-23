import { Viewer } from './viewer';
import { Viewer3D } from './viewer_3d';
import { LiveViewer } from './live_viewer';

export { Viewer, Viewer3D, LiveViewer };

(window as any).RiichiEnvViewer = Viewer;
(window as any).RiichiEnv3DViewer = Viewer3D;
(window as any).RiichiEnvLiveViewer = LiveViewer;
