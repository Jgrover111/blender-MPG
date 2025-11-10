# AGENTS.md — Repository Assistance Guidelines

## Mission
This repository mirrors Blender's Manifold Path Guiding prototype. When working in this tree, focus on the user's explicit
request and do not assume additional work. In particular, **never invent bug-fix tasks** unless the user has clearly described a
reproducible defect that they want addressed.

## Workflow expectations
1. Clarify the scope by re-reading the user prompt before editing files.
2. Prefer the smallest possible change that satisfies the request.
3. Document any tests that were executed. If no tests were run, state the reason.
4. Keep changes CPU-only unless the prompt explicitly asks otherwise.

## Code-style notes
* Follow the surrounding file's conventions; match formatting and naming that already exists nearby.
* Avoid introducing broad refactors unless the user insists on them.
* Guard experimental features with the existing build options where appropriate (`WITH_CYCLES_*`).

## Communication reminders
* Summaries must highlight the observable behavior change.
* Testing sections should list the exact commands used, or explain why testing was skipped.

## Safety
* Do not touch GPU back ends or volume rendering code unless specifically asked.
* Fallback gracefully if new code encounters invalid inputs (NaN, Inf, or negative PDFs).
