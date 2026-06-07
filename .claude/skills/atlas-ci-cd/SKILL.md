---
name: atlas-ci-cd
description: Atlas git, commit, branch, PR, merge, and CI conventions. Covers the commit-message style, the phase-by-phase commit cadence, branching/PR workflow against github.com/hboppana/Atlas, and the (planned) GitHub Actions pipeline. Use when committing, branching, opening or merging PRs, writing commit messages, or setting up CI for Atlas.
---

# Atlas CI/CD & Git Conventions

How work lands in the repo. Remote: **`https://github.com/hboppana/Atlas.git`**, default
branch **`main`**. Read `atlas-architecture` for the phase-by-phase discipline this mirrors.

## Commit messages

Match the existing history — **short, lowercase, descriptive, imperative-ish**, no
Conventional-Commits prefixes. Existing examples:

```
documentation for tensor foundation
ran script with sanity check
script for tinyllama installation
set up repo structure and overall project direction
```

- One logical change per commit. Subject ≈ 50 chars; add a body only when the *why* isn't
  obvious from the subject.
- Always end the commit message with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- **Commit/push only when the user asks.** Don't auto-commit after edits.

## Commit cadence (phase-by-phase)

Atlas is built phase by phase, deeply — commits follow the same grain:

- Commit a **step** when its definition of done is met (e.g. a Phase 1 step: `cmake --build`
  succeeds on MSVC **and** the step's `test_*.exe` passes against the `reference/` oracles).
- Don't bundle multiple half-finished steps into one commit. A green build + passing tests
  is the natural commit boundary.
- Never commit `weights/`, `build/`, `data/papers/`, `data/chunks/`, or `.env` — they are
  gitignored. `reference/` oracles **are** committed (small, used as test fixtures).

## Branching & PRs

- `main` is the default/integration branch. **If asked to commit while on `main`, create a
  branch first** rather than committing directly to `main`.
- Suggested branch names track the phase/step: `phase1/tokenizer`, `phase2/matmul-kernel`,
  `fix/rope-sign`, `docs/forward-validation`.
- Use the **`gh` CLI** for PRs and other GitHub operations.
- Open a PR with a body that summarizes the change, how it was validated (which tests /
  which `reference/` oracle), and ends with:
  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```
- Prefer squash-or-clean merges that keep `main`'s history readable. Don't force-push
  shared branches or skip hooks (`--no-verify`) unless explicitly asked.

## Continuous integration (planned — not yet set up)

No `.github/workflows/` exists yet. When CI is added, it should mirror the local
definition-of-done so green CI == a phase step is actually done:

- **Engine (C++):** configure + `cmake --build` (MSVC and/or gcc), CPU-only path, then run
  `engine/tests/` (`test_tensor`, `test_tokenizer`, `test_forward`, `test_quantize`)
  against the committed `reference/` oracles.
- **Python:** install `requirements.txt` (conda or pip), lint, and run any
  `scripts/`/serving/rag/agent tests for phases that exist.
- **CUDA (Phase 2+):** compiled and benchmarked on **HiPerGator via SLURM**
  (`slurm/build_cuda.sh`, `slurm/benchmark.sh`), not in hosted CI runners.

Add CI jobs incrementally, one per phase as that phase comes online — don't scaffold a
five-phase pipeline up front.

## Maintaining the skills (meta)

This repo's `.claude/skills/` are living docs. On any **bug fix, major change, design
change, or build that departs from the repo layout in `atlas-architecture/repo-structure.md`**, create or update the
relevant skill file as part of the same change (no approval needed). A code change and its
skill update belong in the same commit.
