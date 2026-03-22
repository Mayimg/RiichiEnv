# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RiichiEnv is a high-performance Riichi Mahjong game engine and RL environment. Rust core with Python (PyO3/maturin) and WebAssembly bindings, plus a TypeScript replay viewer.

## Build & Development Commands

### Rust
```bash
cargo check -p riichienv-core                  # Check core (pure Rust)
cargo check -p riichienv-core --features python # Check with PyO3 bindings
cargo test -p riichienv-core                   # Run Rust unit tests
cargo fmt                                       # Format
cargo clippy --all-targets --all-features       # Lint (must pass with -D warnings)
cargo bench --bench agari_bench -p riichienv-core  # Benchmarks
```

### Python
```bash
uv sync --dev                    # Install dev dependencies
uv run maturin develop --release # Build and install Python extension into .venv
uv run pytest                    # Run all Python tests
uv run pytest tests/test_core.py # Run a single test file
uv run pytest tests/test_core.py::test_function_name  # Run a single test
uv run ruff format .             # Format
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ty check                  # Type check
```

### Pre-commit (runs all checks)
```bash
uv run pre-commit run --config .pre-commit-config.yaml
```
Order: commitlint → rustfmt → clippy → ruff-check → ty-check → pytest → ruff-format

### UI (TypeScript)
```bash
cd riichienv-ui && npm install
npm run build         # Full build (WASM + UI)
npm run build:no-wasm # UI only (skip WASM rebuild)
npm run dev           # Dev server with hot-reload
npm run lint          # Biome linter
npm run test          # Vitest
```

### WASM
```bash
rustup target add wasm32-unknown-unknown
wasm-pack build riichienv-wasm --target web
cargo check -p riichienv-wasm --target wasm32-unknown-unknown
```

## Architecture

### Cargo Workspace (3 crates)

- **riichienv-core** — Pure Rust game engine (rlib). All game logic lives here.
- **riichienv-python** — PyO3 cdylib wrapper. Depends on core with `python` feature.
- **riichienv-wasm** — wasm-bindgen cdylib. Depends on core with `wasm` feature, `default-features = false`.

### Core Module Layout (`riichienv-core/src/`)

| Module | Purpose |
|---|---|
| `state/`, `state_3p/` | Game state management (4P and 3P sanma). Sub-modules: `event_handler`, `legal_actions`, `player`, `wall`, `game_mode` |
| `observation/`, `observation_3p/` | Player-facing views and 74-channel feature encoding for ML |
| `replay/` | MJAI and MJSoul replay parsing |
| `agari.rs` | Win detection (agari/tenpai) |
| `hand_evaluator.rs`, `hand_evaluator_3p.rs` | Hand evaluation (yaku, fu, score) |
| `yaku.rs`, `yaku_3p.rs` | Yaku definitions |
| `score.rs` | Score calculation |
| `shanten.rs` | Shanten calculation using lookup tables |
| `action.rs` | Action types |
| `parser.rs` | Hand/tile string parsing |
| `rule.rs` | Game rule configuration |
| `game_variant.rs` | `GameStateVariant` enum dispatching between 4P/3P |
| `errors.rs` | `RiichiError` / `RiichiResult<T>` error types |

### Python Package (`src/riichienv/`)

Python wrapper around the Rust extension (`riichienv._riichienv`). Key exports: `RiichiEnv`, `GameRule`, `Action`, `Observation`, `HandEvaluator`, `calculate_score`, `calculate_shanten`.

### UI Build Chain

```
riichienv-core → riichienv-wasm (wasm-pack) → riichienv-ui (esbuild) → src/riichienv/visualizer/assets/viewer.js.gz
```

WASM binary is inlined into a single JS bundle, then gzip-compressed and copied to the Python package for Jupyter notebook use.

## Key Patterns

### Feature Flags and Conditional Compilation

Use `cfg_attr` for PyO3 annotations on structs:
```rust
#[cfg_attr(feature = "python", pyclass(get_all))]
pub struct Foo { ... }
```

**Do NOT use** `#[cfg_attr(feature = "python", pyo3(get))]` on fields — it doesn't work with `cfg_attr`. Use `get_all`/`set_all` on `pyclass(...)` or manual `#[getter]` methods.

Keep pure Rust logic in regular `impl` blocks. Python wrappers go in separate `#[cfg(feature = "python")] #[pymethods]` blocks with a `_py` suffix:
```rust
impl Foo {
    pub fn compute(&self) -> RiichiResult<u32> { /* ... */ }
}

#[cfg(feature = "python")]
#[pymethods]
impl Foo {
    #[pyo3(name = "compute")]
    pub fn compute_py(&self) -> PyResult<u32> {
        self.compute().map_err(Into::into)
    }
}
```

### Error Handling

Use `RiichiError` / `RiichiResult<T>` from `errors.rs`. Variants: `Parse`, `InvalidAction`, `InvalidState`, `Serialization`. `From<RiichiError> for PyErr` is auto-provided when `python` feature is enabled.

### 4P vs 3P (Sanma)

Separate parallel implementations exist for 4-player and 3-player: `state/` vs `state_3p/`, `observation/` vs `observation_3p/`, `hand_evaluator.rs` vs `hand_evaluator_3p.rs`, `yaku.rs` vs `yaku_3p.rs`. `GameStateVariant` enum dispatches between them.

### Commit Messages

Conventional Commits: `<type>(<scope>): <subject>` (e.g., `feat(env): add event serialization`, `fix(score): correct ura dora calculation`).

## Toolchain

- Rust 1.92.0 (Edition 2024), pinned in `rust-toolchain.toml`
- Python >=3.10,<3.15
- Package manager: `uv` (Python), `npm` (TypeScript)
- Build: `maturin` (PyO3), `wasm-pack` (WASM), `esbuild` (UI)

## Key Documentation

- `docs/DEVELOPMENT_GUIDE.md` — Full development procedures
- `docs/RULES.md` — Game rule specifications
- `docs/DATA_REPRESENTATION.md` — Tile ID formats (TID, MPSZ, MJAI)
- `docs/FEATURE_ENCODING.md` — 74-channel observation tensor encoding
- `docs/SEQUENCE_FEATURE_ENCODING.md` — Sequence feature encoding for RNNs
- `AGENTS.md` — Agent implementation guide
