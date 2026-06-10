---
name: token-efficiency
description: Token/context-efficiency rules for working in Atlas — read docs/skills instead of re-exploring code, targeted reads, trimmed command output, no subagents, targeted test runs. Use at the start of any Atlas work session, when planning a multi-step task, before launching subagents or plan mode, and when running builds/tests or reading large files.
---

# Token Efficiency (Atlas)

Context is the budget. Spend it on the change, not on rediscovering the repo.

## 1. Reuse the recorded map — don't re-derive it

- The `atlas-*` skills + `docs/NN-*.md` are the source of truth for architecture,
  conventions, and measured results. **Read the relevant doc section before reading
  code**; read code only for the lines being changed.
- Never re-read a file already read or edited this session — the harness tracks file
  state, and Edit/Write fail loudly if stale.
- Don't re-verify settled facts (oracle values, pinned thresholds, blob sizes,
  hyperparameters) — they're recorded in the docs and skills.

## 2. Targeted reads, never dumps

- `Grep` with `-n -C <small>` and `head_limit`, or `Read` with `offset`/`limit`, for
  anything over ~200 lines. Whole-file reads only for small files or first contact
  with a file being substantially edited.
- Never read generated/large artifacts: `weights/`, `build/`, `reference/*.npy`,
  `reference/tokenizer/{vocab,merges}.txt`. Their formats and contents are documented.

## 3. Trim command output

- Builds: `cmake --build build 2>&1 | Select-Object -Last <N>` — the last lines carry
  the verdict.
- Tests: `ctest -R <one_test> --output-on-failure` for the test under work; the **full
  suite only at a commit boundary** (test_forward + test_quantize cost ~80 s of wall
  time and dozens of output lines).
- `git status --short`, `git log --oneline -<n>` — never bare `git log`/`git diff` on
  large changes without paging flags.

## 4. No subagents; plan inline

- Do **not** spawn Explore/Plan/general agents for Atlas work — each starts cold and
  re-derives context the session already has, and the skills/docs already encode the
  map. Plan inline from the step's doc.
- Plan mode is for genuinely ambiguous multi-file designs. The Atlas norm — a spec doc
  written first (`docs/NN-*.md`), then step-by-step implementation — already serves
  that purpose more cheaply.
- One early `AskUserQuestion` on a real scope fork is cheaper than building the wrong
  variant.

## 5. Don't pay twice

- Batch independent tool calls in a single message (parallel reads, parallel searches).
- No confirmation re-reads after Edit/Write; no re-running tests when nothing changed.
- One build + one targeted test per change-set; full `ctest` once, before committing.

## 6. Write it down once, save it every session after

- Keeping `docs/` and `.claude/skills/` current as part of each change (the existing
  convention) **is** the token optimization for future sessions: a fact recorded once
  is never re-explored. When a session uncovers something expensive to learn (a build
  gotcha, a measured number, a rejected alternative), it goes in the doc/skill in the
  same commit.
