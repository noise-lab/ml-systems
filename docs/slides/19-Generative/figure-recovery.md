# Figure recovery: 19-Generative

Source: `docs/slides/19-Generative.pptx`, 53 slides, rendered 2026-10-16 (headless LibreOffice, 110 dpi contact sheet, 220 dpi crops). No pen-ink annotations in this deck.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | skip | | | layering / protocol grid; TLS background the Quarto deck deliberately dropped |
| 3–7 | media | | | TLS handshake, config screenshot, misconfiguration reasons, related work; `s03-i01`–`s07-i18` dropped in coverage-notes |
| 8–11 | skip | | | question, bullets, prior-work bullets |
| 12 | media | | | third-party attention diagram (`s12-i21`) with overlapping annotation boxes; no attribution on the slide, and the deck typesets the attention equation in the appendix |
| 13 | media | "The Transformer Architecture" | | `s13-i22` already embedded |
| 14 | skip | | | bullets |
| 15 | media | new, after "BERT for TLS Misconfiguration Detection" | tls-bert-pipeline.png | `s15-i23` was extracted but never embedded; instructor's own lab figure |
| 16 | keep | new, after the pipeline slide | tls-handshake-tokenized.png | raw scan JSON beside the flattened, colour-coded handshake string; WordPiece overlay (shape 492) stripped |
| 17–19 | skip | | | numbers and a text table, already in the BERT slide |
| 20 | media | "BERT for TLS Misconfiguration Detection" | | `s20-i27` already embedded |
| 21 | media | | | multi-label t-SNE (`s21-i28`); the binary one is embedded |
| 22 | skip | | | takeaways |
| 23 | media | "Generative Adversarial Networks" | | `s23-i29` already embedded |
| 24 | skip | | | bullets |
| 25 | skip | | | grey box flowchart (dataset types to uses); restates "What Generative Models Offer" |
| 26 | media | | | clip-art icons, dropped |
| 27–29, 31 | skip | | | name ovals plus bullets |
| 30 | media | | | NetShare accuracy table (`s30-i33`, `s30-i34`), dropped in coverage-notes |
| 32 | skip | | | bullets |
| 33 | media | "NetDiffusion Pipeline Overview" | | `s33-i35` already embedded |
| 34 | media | | | nPrint field-size figure (`s34-i36`); `s35-i37` already shows the encoding |
| 35 | media | "NetDiffusion: Text-to-Traffic Synthesis" | | `s35-i37` already embedded |
| 36–42 | skip | | | five-box step pipeline with per-step highlight; superseded by the paper's Fig. 1 already on "NetDiffusion Pipeline Overview" |
| 38, 40, 41 | media | | | ControlNet and post-processing intermediates, dropped in coverage-notes |
| 43 | skip | | | dataset table (`s43-i44`) plus a text table |
| 44, 45 | skip | | | ">30%", ">70%", ">65%" circles; numbers are on the evaluation slide |
| 46 | media | "NetDiffusion: Evaluation Results" | | `s46-i45` already embedded |
| 47–49 | media | | | balancing bar charts and compatibility tables, dropped in coverage-notes |
| 50, 51 | skip | | | bullets |
| 52 | keep | new, after "GANs for Network Traffic: DoppelGANger and NetShare" | netshare-eval-comparison.png | two drawn tables: NetShare's reported accuracy vs. train-on-synthetic/test-on-real |
| 53 | keep | new, after "NetDiffusion Pipeline Overview" | netdiffusion-ablation-controlnet.png | NetDiffusion paper Fig. 5; the image (`ppt/media/image55.png`) was never extracted |

## Captured

- `images/netshare-eval-comparison.png` (1800×779) — source: `19-Generative.pptx` slide 52, 220 dpi render, title and slide number cropped out, downscaled from 2830 px.
- `images/netdiffusion-ablation-controlnet.png` (2260×744) — source: `19-Generative.pptx` slide 53, `ppt/media/image55.png` copied from the archive (NetDiffusion, ACM SIGMETRICS 2024, Fig. 5).
- `images/tls-bert-pipeline.png` (1800×264) — source: `19-Generative.pptx` slide 15, `images/s15-i23.png` with transparent margins trimmed and flattened to white.
- `images/tls-handshake-tokenized.png` (1800×762) — source: `19-Generative.pptx` slide 16, shape `Google Shape;492` (WordPiece group) stripped, 220 dpi render cropped at 160,360–2900,1520, downscaled from 2740 px.

## Left out

- Slide 12 attention diagram: the base picture is someone else's figure with no source on the slide, and the appendix already typesets the attention equation.
- Slides 36–42 five-box pipeline: the paper's Fig. 1 is already embedded and carries the same steps with more detail.
- Slide 25 dataset flowchart: text in boxes, no intuition beyond the existing bullets.
