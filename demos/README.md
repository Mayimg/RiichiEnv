# Demos

## GameViewer

`GameViewer` renders an interactive 3D replay viewer for Jupyter Notebooks. It takes MJAI-format event logs and displays the game state with metadata such as tenpai waits and winning hand details.

```python
from riichienv.visualizer import GameViewer
```

### Creating a viewer

`GameViewer` provides three class methods for creating a viewer. All methods accept optional keyword arguments:

| Parameter | Type | Description |
|---|---|---|
| `step` | `int \| None` | Initial step to display |
| `perspective` | `int \| None` | Player perspective (0–3) |
| `freeze` | `bool` | Freeze the viewer at the given step |

#### `from_env` — from a RiichiEnv instance

```python
from riichienv import RiichiEnv
from riichienv.visualizer import GameViewer
from riichienv.agents import RandomAgent

agent = RandomAgent()
env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()
while not env.done():
    actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)

GameViewer.from_env(env, perspective=0)
```

#### `from_jsonl` — from a JSONL file

```python
GameViewer.from_jsonl("path/to/game.jsonl", step=100, perspective=0)
```

#### `from_list` — from a list of event dicts

```python
events = [{"type": "start_game", ...}, ...]
GameViewer.from_list(events, step=50, freeze=True)
```

## Notebook Demos

| Notebook | Description |
|---|---|
| `replay_demo.ipynb` | Basic viewer usage with `from_jsonl` |
| `replay_debug.ipynb` | Viewer with programmatically constructed events via `from_list` |
| `replay_penalty_display.ipynb` | Penalty (ryukyoku) display example |
