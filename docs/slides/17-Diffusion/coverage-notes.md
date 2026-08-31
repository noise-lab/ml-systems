# 17-Diffusion — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026:** Added a `.vignette` hook for **NetSSM** (arXiv:2503.22663, March 28 2025;
  published in Proceedings of the ACM on Networking / CoNEXT 2025,
  doi:10.1145/3786289). NetSSM is a direct follow-on from the same research group
  (Chu, Jiang, Liu, Bhagoji, Bronzino, Schmitt, Feamster) that produced NetDiffusion.
  It uses the Mamba selective state-space architecture to generate multi-flow network
  traces 8–78× longer than transformer baselines. Source verified at
  arxiv.org/abs/2503.22663 and dl.acm.org/doi/abs/10.1145/3786289.

## Suggested missing coverage on broad themes (point 3)

- **Differential privacy for synthetic traffic:** Chapter 7 covers membership inference
  and differential privacy at a high level; a lab exercise applying DP-SGD to a
  GAN or diffusion model on network data would make the privacy–utility tradeoff
  concrete.
- **Benchmark datasets and evaluation methodology:** There is no consensus benchmark
  for traffic generation quality. A slide comparing evaluation protocols across papers
  (JSD, EMD, downstream classifier accuracy, Wireshark parse success rate) would help
  students evaluate future claims critically.
- **Payload generation:** NetDiffusion and NetSSM focus on headers. Realistic payload
  synthesis (for application-layer testing) is an open problem worth a slide.
- **Attack-traffic synthesis for IDS evaluation:** The "what-if analysis" use case
  (generate unseen attack traffic for red-teaming classifiers) is mentioned but not
  elaborated. A short example would make the security application concrete.
- **NetFound foundation model:** Chapter 7 covers NetFound (Guthula et al., 2023)
  briefly; a comparison slide (NetDiffusion / NetSSM / NetFound positioning) could
  anchor the "where is this going?" discussion.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items to re-verify:

- **NetSSM vignette** (March 2025): check whether a published journal/conference version
  supersedes the arXiv preprint; update the DOI/venue if so. Also check whether
  NetSSM has been extended or outperformed by a successor by mid-2026.
- **Accuracy figures** ("> 30%", "> 70%", "> 65%"): these are from the NetDiffusion
  SIGMETRICS 2024 paper. If a 2026 reproduction or rebuttal changes them, update.
- **Stable Diffusion 1.5** as the base model: SD 1.5 may be superseded by a newer
  checkpoint or SDXL-class model; check if NetDiffusion has been updated.
- Check the **Architectural Selection Framework** paper (arXiv:2410.16326) — it claims
  an empirical comparison of 12 synthetic traffic methods; if a more comprehensive
  benchmark emerges, update the limitations slide.

## Curated images

Images used:

| File | Slide | Reason kept |
|---|---|---|
| `s13-i08.png` | Forward process | Shows gradual noise addition t=0 → t=999; ideal intuition pump |
| `s17-i09.png` | Pipeline overview | Official NetDiffusion architecture figure from the paper |
| `s22-i14.png` | Traffic-to-image | Clean nPrint-encoded TCP flow with labeled axes |
| `s25-i16.png` | ControlNet | Shows full ControlNet pipeline from real traffic → edge → generation → post-processing |
| `s27-i18.png` | Evaluation setup | Dataset overview table (macro-service / application / flow counts) |
| `s30-i19.png` | ML accuracy | Mixing-rate curves comparing NetDiffusion vs. NetShare across classifiers |
| `s32-i22.png` | Non-ML compatibility | Wireshark capinfos validation log |
| `s33-i23.png` | Non-ML compatibility | Network analysis task coverage table |
| `s07-i05.png` | GAN limitations | Accuracy collapse table: real pcap vs. NetFlow vs. NetShare |

Images dropped:

| File | Reason |
|---|---|
| `s03-i01.png` | Decorative padlock icon — clip-art, zero informational value |
| `s03-i02.png` | Not reviewed (presumed decorative given slide 3 content) |
| `s03-i03.jpg` | Not reviewed (presumed decorative) |
| `s11-i06.png` | Generic "original image → pure noise" pair — s13-i08.png is more informative (multi-timestep strip) |
| `s12-i07.png` | Forward-process step diagram (3 panels) — concept adequately covered in text + s13-i08.png |
| `s18-i10.png` | Unknown / not clearly needed given s17-i09.png covers the full pipeline |
| `s22-i12.png` | Canny edge-extraction output only (black-and-white edges) — partial diagram; s25-i16.png shows the full ControlNet pipeline |
| `s22-i13.png` | Duplicate of s22-i14.png (same nPrint image, lower contrast) |
| `s24-i15.png` | Not reviewed; dependency-tree detail covered in text |
| `s26-i17.png` | Not reviewed; step 4 (image-to-pcap conversion) covered by s17-i09.png pipeline figure |
| `s31-i20.png` | Class-balancing table (before/after statistics) — adequately summarized in text |
| `s31-i21.png` | Presumed duplicate or related class-balancing chart |
| `s07-i04.png` | Single-row NetFlow record screenshot — too small to read; concept explained in text |

## Source

- Rebuilt from `_source-extract.md` (37 slides) + book Chapter 7 ("Generative Models")
  of *Machine Learning for Networking* (`/text/generative.rst`).
- Current-events vignette sourced from arXiv:2503.22663 (NetSSM, March 2025) and
  dl.acm.org/doi/abs/10.1145/3786289 (ACM on Networking).
- Final deck: 20 slides (within 15–30 target range).

## Divergences from source slides

- Source slides 1, 35–37 (title slide, ablation study details, NetShare replication)
  were dropped — not needed for the lecture narrative.
- The book's four-category use-case list (augmentation / simulation / privacy /
  what-if) replaces the source's more scattered motivation. Book order preserved.
- "Text conditioning" (source Slide 16) was merged into the "Conditional Generation"
  slide rather than given its own slide — the same content at lower redundancy.
- The limitations section follows the book's enumerated list (4 limitations) rather than
  the more open-ended "Possibilities and Challenges" framing of source Slide 34.
