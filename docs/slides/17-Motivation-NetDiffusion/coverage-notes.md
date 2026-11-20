# 17-Motivation-NetDiffusion — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

This deck coalesces `17-Diffusion` (the NetDiffusion deep dive) and the first
half of `19-Generative` (the architecture survey) into one 25-to-40-minute
lecture: why synthetic traffic, what to generate, GANs, how diffusion works,
NetDiffusion's four steps, evaluation, limits, and a bridge to the sequence
models lecture (transformers, state machines, state-space models, NetSSM).
The source decks are untouched.

## Where each slide came from

| # | Slide | Source |
|---|---|---|
| 1 | Title | YAML header from 17, retitled |
| 2 | Why Synthetic Network Data? (divider) | 17 |
| 3 | Network Trace Datasets: What and Why | 17 |
| 4 | The Network Data Problem | 17, with the reasons 19's "Why Network Data Is Scarce" added that 17 lacked (labels, lossy collection, the capture may not contain the event, staleness) folded into the right column |
| 5 | What Synthetic Data Buys Us | 17, with the what-if use spelled out as counterfactual traffic and a zero-day bullet; diagram kept |
| 6 | What Should We Generate? | New, built from 17's "Why Raw Packets Matter": pcaps vs. flow records vs. logs vs. the thing that produced the traffic; keeps the accuracy-gap chart |
| 7 | Generative Architectures (divider) | 17 |
| 8 | Four Generations of Generative Models | 19 |
| 9 | Generative Adversarial Networks (GANs) | 19 |
| 10 | GANs for Network Traffic: DoppelGANger and NetShare | 19; the Limitations column rewritten as "Where GANs fall short" with the four shortfalls (no convergence guarantee, mode collapse, aggregate statistics not packets, fooling the discriminator is not realism), so no separate slide was needed; NetShare attribution corrected to Yin et al. |
| 11 | NetShare: Reported Accuracy vs. Train-on-Synthetic, Test-on-Real | 19 |
| 12 | How Diffusion Models Work (divider) | 17 |
| 13 | The Forward Process: Adding Noise | 17 (figure narrowed to 62% to clear the footer) |
| 14 | The Reverse Process: Denoising to Generate | 17 |
| 15 | Forward and Reverse in One Picture | 17 |
| 16 | Conditional Generation with Text Prompts | 17, plus the Stable Diffusion latent-space bullet from 19's "How Diffusion Models Work" |
| 17 | NetDiffusion (divider) | 17 |
| 18 | Pipeline Overview | 17 |
| 19 | The Four Steps at a Glance | 17 |
| 20 | Step 1: Traffic-to-Image Conversion | 17 |
| 21 | One Packet, One Row | 17 |
| 22 | Step 2: LoRA Fine-Tuning | 17 |
| 23 | Step 3: ControlNet-Guided Generation | 17, relaid with the ControlNet strip full-width and the six-fingers analogy in the problem statement |
| 24 | Why Fine-Tuning and ControlNet Both Matter | 19 (the paper's Fig. 5 ablation; the only 19 NetDiffusion slide with a figure 17 lacked) |
| 25 | Step 4: Protocol-Compliance Post-Processing | 17, with the majority-vote and "engineering, not modeling" points from 19's post-processing slide folded into the goal bullet and notes |
| 26 | Evaluation (divider) | 17 |
| 27 | Evaluation Setup | 17 |
| 28 | Statistical Similarity Results | 17 |
| 29 | ML Accuracy: Train-on-Synthetic, Test-on-Real | 17 |
| 30 | Non-ML Compatibility | 17 |
| 31 | Limitations and What Comes Next (divider) | 17's "What Comes Next?" retitled |
| 32 | Current Limitations | 17, with two limits added (traffic is not really an image; one flow per image, no multi-flow sessions) in the diagram and bullets |
| 33 | Diffusion vs. Alternatives | 17 |
| 34 | Open Challenges | 17 |
| 35 | Next: Transformers and State-Space Models | New bridge slide replacing 17's "Follow-on Work: NetSSM (2025)": three limitation-to-remedy pairs (tokens instead of pixels; a learned state; long, multi-flow context) with a Graphviz map, and the next-lecture line |
| 36 | Key Takeaways | 17, with a what-to-generate bullet added and the limitations bullet pointing to sequence models rather than NetSSM results |
| 37 | Appendix: The Math (divider) | 17 |
| 38 | Forward Process: Equations | 17 |
| 39 | Reverse Process: Equations | 17 |
| 40 | GAN Objective | 19 (`.large` added under the font gate) |

## Source slides dropped

- 17 "Early Approach: GAN-Based Traffic Synthesis": its generator/discriminator
  bullets are 19's GAN slide, its DoppelGANger/NetShare bullets are slide 10,
  and its accuracy table (s07-i05) is the same Table 1 that slide 6's chart and
  slide 11's recovered table show.
- 17 "Why Raw Packets Matter": rebuilt as slide 6.
- 17 "Follow-on Work: NetSSM (2025)": replaced by the bridge slide; NetSSM
  belongs to the next lecture and still appears in the comparison table.
- 19 "Why Network Data Is Scarce" and "What Generative Models Offer": folded into
  slides 4 and 5.
- 19 "How Diffusion Models Work": its Graphviz chain duplicates the DDPM Figure 2
  slide; its Stable Diffusion bullets went to slide 16.
- 19 "NetDiffusion: Text-to-Traffic Synthesis", "NetDiffusion Pipeline Overview",
  "Protocol Compliance via Post-Processing", "Evaluation Results", "Limitations
  of Diffusion Models for Networking": all covered in more detail by 17's step,
  evaluation, and limitations slides.
- Everything in 19 from "Transformers and LLMs" onward (embeddings, the
  transformer, BERT for TLS, the scaling problem, SSMs and Mamba, NetSSM,
  NetLLM, privacy, architecture summary, the attention and SSM equations):
  the next lecture's material.

## Images and figures

`images/` holds only what this deck uses: fifteen files from `17-Diffusion/images`
(the pptx recoveries, the paper clippings, and the five `gen-*.svg` figures) and
four from `19-Generative/images` (the timeline, the GAN flow, the NetShare
comparison table, the ControlNet ablation). `figures.py` is 17's generator
(the `gen-*.svg` files); `figures-landscape.py` is 19's (the timeline, plus the
attention-vs-SSM cost figure this deck does not use). Neither was run; the SVGs
were copied.

Graphviz diagrams in `slides.qmd`: uses (slide 5), conditioning (16), LoRA (22),
post-processing (25), limitations (32), and the new bridge map (35).

## Next-year refresh notes

Same items as the two source decks: the NetSSM venue line on the comparison
slide, the NetDiffusion headline numbers, and the base model named on the
pipeline and LoRA slides (SD 1.5 in the paper, SD 3 in the current
`noise-lab/NetDiffusion` code).
