# Repository Guidelines

## Project Structure & Module Organization
`src/riichienv/` contains the Python package, including the public API, converters, agents, and notebook visualizer hooks. `tests/` holds the main pytest suite, with deeper environment coverage under `tests/env/`. The Rust workspace lives in `riichienv-core/` (game engine), `riichienv-python/` (PyO3 extension), and `riichienv-wasm/` (WASM bindings). `riichienv-ui/` contains the TypeScript viewer, demo files, and Vitest tests in `src/__tests__/`. Keep reference docs in `docs/` and small maintenance utilities in `scripts/`.

## Build, Test, and Development Commands
Install Python tooling with `uv sync --dev`. Build the extension into the active virtualenv with `uv run maturin develop --release`. Run Python tests with `uv run pytest`, Rust tests with `cargo test -p riichienv-core`, and Rust linting with `cargo clippy --all-targets --all-features -- -D warnings`. For the UI, use `cd riichienv-ui && npm ci`, then `npm test` for Vitest and `npm run build` for the full WASM plus bundle pipeline. Before opening a PR, run `uv run pre-commit run --config .pre-commit-config.yaml`.

## Coding Style & Naming Conventions
Python follows Ruff defaults with a 120-character line limit, 4-space indentation, and double quotes. Use `snake_case` for modules, functions, and tests; keep public package entry points in `src/riichienv/`. Rust must stay `cargo fmt` clean and pass clippy without warnings; prefer `snake_case` modules and keep core logic in `riichienv-core` with bindings isolated to wrapper crates or feature-gated modules. UI code is formatted by Biome: 4 spaces, single quotes, semicolons, and 120-character lines.

## Testing Guidelines
Add tests beside the affected surface: pytest files named `test_*.py`, UI tests as `*.test.ts`, and Rust integration tests under `riichienv-core/tests/`. No explicit coverage threshold is defined, but CI runs pytest, cargo test, Vitest, Ruff, Ty, rustfmt, clippy, and Biome checks, so changes should keep all impacted suites green. If you touch the viewer build chain, verify the generated asset at `src/riichienv/visualizer/assets/viewer.js.gz`.

## Commit & Pull Request Guidelines
Commit messages are validated with Conventional Commits via commitlint; use prefixes such as `feat:`, `fix:`, `docs:`, and `chore:`. Keep commits scoped to one logical change. Pull requests should describe the behavior change, list the commands you ran, link related issues, and include screenshots or short recordings for UI or visualization updates.
