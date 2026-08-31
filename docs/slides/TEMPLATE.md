# Lecture Deck Template Spec

Every lecture deck follows this spec. The canonical worked example is
**`01-Overview/`** — read it before authoring a deck. Goal: a **content-driven
rebuild** in Quarto reveal.js (not a slide-for-slide port of the old pptx).

## Folder layout (one per lecture)

```
NN-Name/
  slides.qmd          # the Quarto reveal.js deck (authored)
  coverage-notes.md   # instructor-facing: current-events updates + missing-coverage suggestions
  images/             # extracted originals; reference the genuinely useful ones
  _source-extract.md  # raw pptx extraction (input only; gitignored)
```

## Quarto setup

- Decks inherit `slides/_quarto.yml` (reveal.js defaults, footer, slide numbers) and
  `slides/theme.scss` (UChicago maroon). **Do not** restate format options already in
  the shared config.
- Per-deck front matter is minimal:
  ```yaml
  ---
  title: "Deck Title"
  subtitle: "Security, Privacy, and Consumer Protection"
  author: "Nick Feamster · University of Chicago"
  date: last-modified
  ---
  ```
- The deck **must `quarto render` cleanly** (`quarto render NN-Name/slides.qmd`).

## Slide conventions

- **Titles in Title Case.** Capitalize principal words; lowercase short articles /
  conjunctions / prepositions (*a, an, the, and, or, for, to, of, in, on, by*); always
  capitalize the first and last word. (e.g., "Why This Course Exists", "Is This Course
  for You?")
- `##` starts a slide; `#` starts a section divider. Use `{.center}` for divider/quote
  slides, `{.smaller}` for dense slides.
- Two-column layout:
  ```
  ::: {.columns}
  ::: {.column width="50%"} ... :::
  ::: {.column width="50%"} ... :::
  :::
  ```
- **Bold key terms** (the theme renders them maroon). Keep bullets tight — a slide is a
  cue, not a paragraph.
- **Current-events vignette box** (use at least one per deck, see point 2):
  ```
  ::: {.vignette}
  A short, dated, real-world hook tied to this lecture's theme.
  :::
  ```
- **Speaker notes inline** on most slides:
  ```
  ::: {.notes}
  What to say / the story / the cold-call. Not shown on the slide.
  :::
  ```

## Images (curate — don't dump)

- Reference only the **genuinely useful** images from `images/`: architecture diagrams,
  attack flows, real screenshots that teach something, data plots.
- **Drop** clip-art, logos, decorative headshots, and redundant screenshots.
- Caption figures with `![Caption](images/file.png)`. For multiple images use the
  columns layout (see `01-Overview` "Where This Can Take You").
- If a deck needs a diagram that doesn't exist, describe it in a `::: {.notes}` block or
  in `coverage-notes.md` rather than inventing a misleading one.

## Content approach (the three asks)

1. **Agenda alignment.** Build from `_source-extract.md` **and** the matching section(s)
   of `../../agenda.md` (the detailed record of what was *actually* covered, by
   Meeting). Prefer what was actually taught over the old slide order. If the deck's
   topic spans part of a Meeting, scope to its theme.
2. **Update examples to current events.** Replace dated examples with current ones
   (today is 2026). **Use web search** to ground at least one fresh, accurate, dated
   example per deck; put the freshest hook in a `.vignette`. Don't fabricate facts —
   verify before stating.
3. **Suggest missing coverage.** In `coverage-notes.md`, list (a) the current-events
   updates you made and (b) concrete suggestions for missing coverage on the lecture's
   broad themes. Keep slides focused; put the meta-commentary in coverage-notes, not on
   slides.

## coverage-notes.md format

```
# NN-Name — instructor notes
## Current-events updates made (point 2)
- <YEAR>: ... (date-stamp each refresh so the history is legible)
## Suggested missing coverage on broad themes (point 3)
- ...
## Next-year refresh notes
- Hooks/figures with a shelf life (what will go stale and when)
- Stronger alternative vignettes flagged but not yet used
## Curated images
- which images were used / dropped and why
## Source
- rebuilt from _source-extract.md (N slides) + agenda.md Meeting M
```

The **Next-year refresh notes** section is the to-do list the annual refresh (below)
reads first. Whenever you place a dated hook, jot what will age out and any stronger
alternative you didn't use — future-you (or a future Claude run) starts there.

## Length guidance

Lean. A rebuilt deck is typically **15–30 slides** even if the original had 70–150 —
consolidate, cut redundancy, and keep one idea per slide. The old slide count is not a
target.

## Annual current-events refresh

The decks are built to be **re-pointed at the present each year** without a rebuild.
Run this at the start of each term. It is a *surgical refresh of time-sensitive
content*, **not** a re-authoring — structure, pedagogy, and slide order stay put.

**What to refresh (and only this):**

- The **`.vignette`** hook(s) — swap any example that a new student would read as
  "old news" for the freshest equivalent on the same teaching point.
- **Dated facts and figures** — enforcement totals, fine amounts, adoption stats,
  "~N states," version numbers, "as of <date>" phrasing, and any year literal.
- **Broken or superseded links**, and case statuses that have moved (settled, appealed,
  overturned, a law now in effect).

**How (the loop):**

1. Read the deck's `coverage-notes.md` **Next-year refresh notes** and
   **Current-events updates made** first — that's the standing to-do list and tells you
   what was deliberately left for this year.
2. **Web-search to verify every change.** Replace a fact only with a more current,
   *verified* one (primary source where possible). **Never fabricate** a case, date, or
   figure; if you can't verify a fresher example, keep the existing one and note it.
3. Keep the *teaching point* identical — change the illustration, not the argument. One
   strong fresh hook beats three weak ones; don't pad.
4. Re-stamp `coverage-notes.md`: add a dated bullet under **Current-events updates made**
   for what you changed, and refresh **Next-year refresh notes** with the next things
   likely to age out.
5. `quarto render NN-Name/slides.qmd` and confirm it builds clean.

**Scope guardrails:** don't add/remove slides, restructure, or chase the "missing
coverage" suggestions during a refresh (those are a separate, deliberate editing pass).
Touch only `slides.qmd` and `coverage-notes.md`. Leave `_quarto.yml`, `theme.scss`, and
this file alone.

### Reusable annual-refresh prompt

Paste this to a fresh Claude session each year (edit the year and, optionally, scope to
specific decks):

> Refresh the current-events content of the lecture decks in `docs/slides/` for the
> **<YEAR>** academic year, following the "Annual current-events refresh" section of
> `docs/slides/TEMPLATE.md`. For each deck `NN-Name/`: read its `coverage-notes.md`
> first; web-search to verify; update only the `.vignette` hooks, dated facts/figures,
> and stale links/case statuses; keep structure and teaching points unchanged; re-stamp
> `coverage-notes.md` with what changed and what's likely to go stale next; and
> `quarto render` each deck to confirm it builds. Don't fabricate facts — if you can't
> verify a fresher example, keep the current one and say so. Use one agent per deck and
> parallelize. When done, give me a per-deck summary of what changed, and flag any deck
> whose core example is now genuinely outdated and may warrant a deeper edit.
