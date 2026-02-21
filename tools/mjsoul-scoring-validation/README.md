# mjsoul-scoring-validation

Validate the scoring logic of RiichiEnv against MJSoul game records.

```sh
uv sync
uv run python main.py
uv run python main_3p.py
uv run python validate_env_4p.py
uv run python validate_env_3p.py
```

**NOTE:** `mjsoul-parser` is **not included** in this repository. Users are expected to prepare or use their own parser implementation.
