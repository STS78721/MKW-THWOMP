# tests/translator

Unit tests for SM86 → GLSL translator passes, run without game files or a GPU (`uv run pytest tests/translator`).

| File | What it checks |
|---|---|
| `test_first_store.py` | First-invocation preamble store splitting (`sm86/translate/glsl/first_store.py`) on 1D/2D compute shader preambles |
