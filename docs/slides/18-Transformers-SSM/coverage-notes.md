# 18-Transformers-SSM — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

This deck coalesces `18-State-Space` (the NetSSM deep dive) and the transformer and
state-space half of `19-Generative` (the survey) into one 30-to-40-minute lecture that
follows the diffusion lecture. The two source decks are untouched. Taught order:
diffusion's limits → transformers and their scaling wall → protocols as state machines
(new) → state-space models → NetSSM → evaluation → comparison → summary, with the
BERT-for-TLS slides and the equations in an appendix.

## Where each slide came from

| # | Slide | Source |
|---|---|---|
| 1 | Title | new YAML header (same fields as 18, retitled) |
| 2 | Where Diffusion Falls Short (section) | new; carries the `<style>` block that 18's first section had |
| 3 | What NetDiffusion Cannot Do | new text; figure `images/s35-i37.png` from 19's "NetDiffusion: Text-to-Traffic Synthesis" |
| 4 | The Generative Model Landscape | 18, verbatim |
| 5 | Transformers (section) | 19's "Transformers and LLMs" divider, retitled |
| 6 | Embeddings: Words, Packets, and Flows | 19, verbatim |
| 7 | The Transformer Architecture | 19, verbatim (`images/s13-i22.png`) |
| 8 | The Scaling Problem: O(n²) Attention | 19, verbatim (`images/attention-vs-ssm-cost.svg`); one notes sentence changed from "the subject of Deck 18" to "the second half of this deck" |
| 9 | Attention Has No State | new (Graphviz) |
| 10 | Protocols Are State Machines (section) | new |
| 11 | State Machines Generate Output | new (Graphviz, TCP client states in Mealy notation) |
| 12 | Markov Models: Probabilistic State Machines | new (Graphviz, streaming client as a Markov chain, illustrative probabilities) |
| 13 | State-Space Models (section) | 19's "State Space Models" divider |
| 14 | State Space Models: The Intuition | 18, merged with the one-step recurrence diagram from 19's "State Space Models and Mamba"; the analogy sentence moved to the notes and the "Max length" table row dropped to fit |
| 15 | The State-Space Recurrence | 18, verbatim |
| 16 | Mamba: Selective State Spaces | 18, verbatim |
| 17 | Why SSMs Fit Network Traffic | 18, verbatim |
| 18 | NetSSM: Design and Method (section) | 18 |
| 19 | NetSSM Overview | 18, verbatim |
| 20 | The NetSSM Pipeline | 18, verbatim (`figures/pipeline.png`) |
| 21 | Tokenization and Training | 18, verbatim (`figures/tokenization-bytes-to-tokens.svg`) |
| 22 | Evaluation (section) | 18 |
| 23–28 | the six evaluation slides | 18, verbatim, in 18's order |
| 29 | Comparison and Context (section) | 18 |
| 30 | Comparison to Prior Work | 18, verbatim (`figures/context-generation-length.svg`) |
| 31 | Architecture Summary and Tradeoffs | 19, verbatim |
| 32 | NetLLM: Adapting LLMs to Networking Tasks | 19, verbatim |
| 33 | Privacy Caveat: Synthetic ≠ Automatically Private | 18, verbatim |
| 34 | Limitations and Open Problems | 18, verbatim |
| 35 | Summary (section) | 18 |
| 36 | Key Takeaways | 18, plus one new "Protocols are state machines" paragraph |
| 37 | Discussion Questions | 18, verbatim |
| 38 | Appendix (section) | new divider with a note that the TLS slides show classification, not generation |
| 39–41 | BERT for TLS (three slides) | 19, verbatim (`images/s20-i27.png`, `images/tls-bert-pipeline.png`, `images/tls-handshake-tokenized.png`) |
| 42 | Scaled Dot-Product Attention | 19, verbatim |
| 43 | State-Space Recurrence | 19, verbatim; the back-reference now names "State Space Models: The Intuition" |

## Source slides dropped, and why

- 18 "Why Synthetic Network Data?": motivation is covered by the previous (diffusion) lecture; the landscape table carries the recap.
- 18 "Transformers: Review and Scaling Problem": redundant with 19's "The Scaling Problem" (same cost figure) plus the new "Attention Has No State".
- 19 "State Space Models and Mamba": its diagram was merged into slide 14; its text is covered by 18's four SSM slides.
- 19 "NetSSM: State-Aware Traffic Generation": a table, no figure; covered by 18's NetSSM Overview and Comparison slides.
- 19 "Privacy in Synthetic Data": no figure; 18's privacy vignette covers it (the Shokri and Dwork background moved into the speaker-notes row).
- 19 "Where the Field Stood in 2025" and "Open Challenges and Future Directions": no figures; dated note and the open problems are covered by 18's Limitations slide (NetFound and context transfer moved into its speaker-notes row).
- 19's GAN, diffusion, NetDiffusion and NetShare slides, the "Four Generations" timeline, the motivation slides, the "Broader Applications" divider, the "Summary" slide, and the "GAN Objective" appendix slide: belong to the previous lecture or the survey, not this one.

## Figures

- `figures/` copied from `18-State-Space/figures/` (the five generated SVGs the deck uses, the three paper PNGs, and `evaluation_detailed.png` and `ks_sizes.png`, which the speaker notes point to). `make-figures.py` is a copy of 18's script and regenerates the SVGs into `figures/`.
- `images/` copied from `19-Generative/images/`: `attention-vs-ssm-cost.svg` (generated by `figures.py`, a copy of 19's script, which writes into `images/`), the transformer diagram, the nPrint image, the t-SNE plot, and the two TLS pipeline figures.
- New Graphviz `{dot}` diagrams inline in `slides.qmd`: "Attention Has No State" (attention scores versus one state update), "State Machines Generate Output" (TCP client states, Mealy notation), "Markov Models" (streaming client as a Markov chain).
- Content slides with a picture: 22 of 33.

## Refresh notes

Inherit both source decks' refresh lists (`18-State-Space/coverage-notes.md`,
`19-Generative/coverage-notes.md`): the privacy vignette's venue, the NetSSM citation,
Mamba successors, the comparison numbers, NetLLM and NetFound status. The Markov-chain
probabilities on slide 12 are illustrative and labelled as such; replace them with
measured transition frequencies if a student project produces some.
