# Figure recovery: 17-Diffusion

Source: `docs/slides/17-Diffusion.pptx`, 37 slides, rendered 2026-10-16 (headless LibreOffice, 110 dpi contact sheet, 220 dpi captures). No pen-ink annotations on any slide.

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title |
| 2 | keep | new, "Network Trace Datasets: What and Why" (before "The Network Data Problem") | trace-datasets-what-why.png | drawn flowchart; builds are entrance-only, final state captured |
| 3 | skip | | | three clip-art icons (s03-i01..i03), dropped by the Sept audit |
| 4–6, 8 | skip | | | bullets with ns-3/YANS/MGEN and DoppelGANger/NetShare ovals, decorative |
| 7 | media | "Early Approach: GAN-Based" | s07-i05.png | already embedded; s07-i04 (NetFlow row) dropped by audit |
| 9 | skip | | | bullets |
| 10 | skip | | | U-Net architecture picture, unattributed third-party figure |
| 11, 12 | media | | s11-i06, s12-i07 | dropped by audit in favour of s13-i08 |
| 13 | media | "The Forward Process" | s13-i08.png | already embedded |
| 14 | skip | | | training-loop picture with timestep slider, unattributed third-party explainer |
| 15 | keep | new, "Forward and Reverse in One Picture" (after "The Reverse Process") | ddpm-markov-chain.png | embedded PNG never extracted by the earlier pass; Ho et al. 2020 Figure 2, credited in caption |
| 16 | skip | | | text-conditioning picture (boy on rocket), unattributed third-party figure |
| 17 | media | "Pipeline Overview" | s17-i09.png | already embedded |
| 18 | media | | s18-i10 | nPrint header-layout picture plus one rounded box; dropped by audit |
| 19–22 | skip | | | intermediate states of the slide-26 strip plus pictures (s19-i11, s22-i12..i14); s22-i14 already embedded |
| 23 | skip | | | bullets |
| 24 | media | | s24-i15 | TCP rule listing, dropped by audit |
| 25 | media | "Step 3: ControlNet" | s25-i16.png | already embedded |
| 26 | keep | new, "The Four Steps at a Glance" (before "Step 1") | netdiffusion-four-step-pipeline.png | final state of the drawn strip; the nPrint picture under it (s26-i17) excluded from the crop |
| 27 | media | "Evaluation Setup" | s27-i18.png | already embedded; drawn three-cell table below it restates "Baseline: NetShare" |
| 28, 29 | skip | | | big-number circles (>30 %, >70 %, >65 %); the deck has them as a table and bullets |
| 30 | media | "ML Accuracy" | s30-i19.png | already embedded |
| 31 | media | | s31-i20, s31-i21 | class-balancing tables, dropped by audit |
| 32, 33 | media | "Non-ML Compatibility" | s32-i22, s33-i23 | already embedded |
| 34, 35 | skip | | | bullets |
| 36 | skip | | | drawn NetShare-replication tables; audit dropped slides 35–37 |
| 37 | skip | | | ablation figure (paper Fig. 5), audit dropped it |

## Captured images

- `images/trace-datasets-what-why.png` (1739×766) — source: `17-Diffusion.pptx` slide 2, 220 dpi render, box 560,540,2320,1350, auto-trimmed, nothing stripped.
- `images/ddpm-markov-chain.png` (2058×312) — source: `17-Diffusion.pptx` slide 15, raw `ppt/media/image11.png`, auto-trimmed. Figure 2 of Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models" (NeurIPS 2020); credited in the slide caption.
- `images/netdiffusion-four-step-pipeline.png` (2833×408) — source: `17-Diffusion.pptx` slide 26, 220 dpi render, box 30,860,2905,1298, auto-trimmed; picture 16 below the strip left out of the crop.

No `speaker-notes.md` exists for this deck, so no rows were added.
