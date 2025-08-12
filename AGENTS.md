This project uses a **feature-driven workflow**. Every change must link to a documented feature.

---

**Before you start:**
- Read `README.md` for project goals.
- See [`docs/index.md`](docs/index.md) for documentation structure.
- Keep all docs under `docs/`.
- Do not list feature specs in `docs/index.md`.

**Code standards:**
- Use absolute imports from the `app` package.
- Add docstrings to all classes and functions.
- Run checks before each commit:
  ```bash
  pre-commit run --files <files>
  ```
- Lint and test often:
  ```bash
  make lint
  make test
  ```
**Tests:**
- Unit tests → tests/unit
- Integration tests → tests/integration (run with --runintegration)


## Workflow:

1. Create a feature file:
Copy docs/features/_template.md → docs/features/<feature>.md

2. Read the spec and related docs.
3. Plan your implementation steps and inspect the codebase.
4. Build iteratively:
Write → Lint → Test → Commit

5. Update docs if behavior, API, or structure changes.
6. Open a pull request referencing the feature file. Make sure CI passes.
