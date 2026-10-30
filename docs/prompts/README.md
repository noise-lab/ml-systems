# Prompts

Prompts used to draft course material with an AI coding assistant. Students are
welcome to run them: every input they need (the agenda, past exams, the term page)
is in this repository, so you can generate unlimited practice exams that follow the
same recipe as the real one. You will not get the same questions.

| File | Purpose |
|---|---|
| [`generate-exam.md`](generate-exam.md) | The general recipe: parameters, LaTeX conventions, validation. Read first. The per-exam prompts live beside the past exams they draw on: |
| [`midterm/generate-midterm.md`](../midterm/generate-midterm.md) | Parameters for the standard midterm (nine-week offering). |
| [`final/generate-final.md`](../final/generate-final.md) | Parameters for the standard final (nine-week offering). |
| [`combined/generate-combined.md`](../combined/generate-combined.md) | Parameters for the single combined exam (three-week offering, any year). |
| [`combined/2026/generate-summer-final.md`](../combined/2026/generate-summer-final.md) | Record of the parameters used for the Summer 2026 (Paris) combined exam. |

Typical use with Claude Code, from the repository root:

```
claude "Follow docs/midterm/generate-midterm.md to make me a practice exam in ~/practice/midterm"
```
