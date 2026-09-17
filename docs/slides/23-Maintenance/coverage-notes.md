# 23-Maintenance — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Sources used

- **LEAF: Navigating Concept Drift in Cellular Networks.** Liu, Bronzino, Schmitt, Bhagoji,
  Feamster, Garcia Crespo, Coyle, Ward. Proc. ACM on Networking, vol. 1, CoNEXT2, 2023.
  arXiv 2109.03011 (LaTeX source, latest arXiv version with the four-year dataset,
  Jan 2018 to Mar 2022, 412 / 898 eNodeBs). The task brief mentioned "400+ base stations,
  8 months"; that describes an earlier arXiv version. The deck follows the source that was
  actually downloaded (four years, two datasets). All numbers in the deck are copied from
  the tex tables.
- **CATO: End-to-End Optimization of ML-Based Traffic Analysis Pipelines.** Wan, Liu,
  Bronzino, Feamster, Durumeric. arXiv 2402.06099 (LaTeX source). Code:
  github.com/stanford-esrg/cato.
- **AC-DC: Adaptive Ensemble Classification for Network Traffic Identification.** Jiang,
  Liu, Naama, Bronzino, Schmitt, Feamster. arXiv 2302.11718 (LaTeX source, Feb 2023;
  the body text calls the system "Adaptive Constraint-Driven Classification"). The
  instructor referred to this paper as "JITI". Five slides plus an appendix formula.
  **Scope note**: the instructor's brief described AC-DC as handling "drift and class
  imbalance" by weighting classifiers as the data distribution moves. The paper does
  not do that: its pool of LightGBM classifiers differs in *feature requirements*
  (cost), and the scheduler switches classifier and batch size in response to *traffic
  rate and memory availability*, not data drift. The deck presents it faithfully as
  "the operating environment drifts too" and frames the ensemble that way. If the
  instructor meant a different paper, the slides need replacing.
- **ServeFlow: A Fast-Slow Model Architecture for Network Traffic Analysis.** Liu, Shaowang,
  Wan, Chae, Marques, Krishnan, Feamster. arXiv 2402.03694 (e-print downloaded during
  authoring, Oct 2024 revision). Trimmed to two slides at the instructor's request.
- Traffic Refinery (Bronzino et al., 2021) is mentioned in one bullet and on the CATO
  comparison slide; no figures from it.

## Current-events updates made (point 2)

- **2026-09**: The `.vignette` uses the LEAF COVID-19 story itself (April 2020 lockdown:
  NRMSE of every downlink-volume model rises, LEAgram shows overestimation at high-volume
  sites through Nov 2020). Web searches in September 2026 for a verifiable 2025–2026
  public post-mortem of a production network ML model degrading from drift (queries on
  "concept drift postmortem", "stale model outage network operator", "model drift caused
  outage 2026") returned only vendor blogs and general guidance, no dated, attributable
  incident. Rather than invent one, the deck says explicitly in the vignette's speaker
  notes that the hook is drawn from the paper.
- Candidate supporting fact surfaced by search but **not used** (unverified by me against
  the primary source): a *Scientific Reports* (Nature) 2022 study by Vela et al.,
  "Temporal quality degradation in AI models," reported temporal degradation in 91% of
  models tested. Verify the citation and number before adding it to the vignette.

## Suggested missing coverage on broad themes (point 3)

- **Label delay and proxy metrics.** The 180-day forecast horizon means error-based
  detection lags by six months. A slide on proxy signals (input drift tests, prediction
  entropy, agreement between old and new models, canary models) would close the gap the
  vignette's thought question opens.
- **Online / streaming learning** as an alternative to batch retraining (river, Vowpal
  Wabbit style), and why LEAF deliberately does not fine-tune (manual cost).
- **Drift in classifiers and security.** LEAF is regression; Transcend (Jordaney et al.,
  2017) and CADE (Yang et al., 2021) do drift explanation for malware classifiers with
  conformal / contrastive methods. One slide would connect to the Security lecture.
- **Model versioning and rollback** as engineering practice (feature stores, shadow
  deployment, A/B against the incumbent model). The playbook slide mentions it in one
  bullet.
- **Hardware serving** (N3IC, Taurus, Homunculus, BoS, LEO) is only in the CATO related
  work; a short slide on where inference runs (SmartNIC, switch, CPU) would round out the
  cost discussion.
- **Hands-on**: a notebook that trains a model on the first month of a public time series
  (e.g., CESNET-TimeSeries24 from the Time-Series lecture), runs KSWIN from `river` on the
  error stream, and compares periodic vs. triggered retraining would make Table 3 concrete.

## Next-year refresh notes

- **Vignette**: replace the LEAF COVID story with a dated 2026–2027 public incident if one
  appears (look for operator or cloud-provider post-mortems, SIGCOMM/NSDI/CoNEXT industry
  talks, or the CESNET group's follow-ups). Keep the teaching point (sudden exogenous
  shock, delayed labels, overestimation cost).
- **ServeFlow venue**: cited as arXiv 2402.03694; check whether it has a conference
  version and update the section-divider citation.
- **CATO venue**: cited as arXiv 2402.06099; check for the published version (likely
  NSDI/SIGCOMM-family) and update.
- **AC-DC venue**: cited as arXiv 2302.11718; check for a published version and confirm
  with the instructor that this is the paper meant by "JITI" (see Scope note above).
- **LEAF author list** on the section divider is abbreviated ("et al."); the full list is
  in Sources above.
- The "Most Flows Are Easy" ServeFlow tables and the headline numbers (76.3%, 40.5x,
  48.5k flows/s) are from the abstract of the Oct 2024 revision; re-check if the paper is
  revised.

## Curated images

All figures were converted from the paper sources with `pdftoppm -r 170` (PNG originals
copied as-is). 50 images referenced in the deck.

**LEAF (arXiv 2109.03011)** — included: Fig. 1a–d (drift across models for DVol, PU, DTP,
GDR), Fig. 7a–b (REst, CDR, appendix), Fig. 2a–b (training size / period), Fig. 3
(framework), Fig. 4 / 8a and 8b (LEAplots), Fig. 5a–b (LEAgrams before/after), Fig. 6a–b
(DVol and CDR cost/accuracy trade-off), Fig. 10a–d (trade-off for the other four KPIs),
Fig. 9a, 9b, 9d (NRMSE over time before/after, DVol, PU, REst). Four **unpublished
source figures** (`leaf-x-sudden/gradual/incremental/recurring.png`) illustrate the drift
taxonomy with real normalized-volume series; they ship with the arXiv source but are not
in the CoNEXT figure set, and the captions say so. `leaf-x-kswin.png` is the authors'
annotated KSWIN detection figure, also from the source but not in the published paper
(the published Appendix B describes KSWIN in prose). Verify with the authors that these
are the figures they intended before reusing outside the course.
Omitted: Fig. 9c, 9e, 9f (DTP, CDR, GDR over time; redundant with the table on that
slide; 9f is described in notes), the 3D `seasonal.png` and `trend.png` (unreadable at
slide size), signal-processing / STFT figures, morphology and per-model CDF figures
(not in the published version).

**Tables reproduced** from LEAF: Table 1 (datasets), Table 2 (target KPI
characteristics, Evolving dataset), Table 3 (periodic retraining), Table 4 (all four
model families, split over two slides), Table 5 (Fixed vs. Evolving with LEAF*), Table 7
(95th-percentile error). Table 6 (Fixed-dataset KPI characteristics), Table 8 and 9 (the
full versions with the Naive-90 column) are omitted; Naive-90 numbers appear in notes.

**CATO (arXiv 2402.06099)** — included: Fig. 1 (serving pipeline), Fig. 2a–b (packet
depth vs. F1 / cost), Fig. 3 (design), Fig. 5a–d (latency and throughput fronts), Fig. 6
(Traffic Refinery comparison), Fig. 7 (Pareto quality), Fig. 8 (convergence), Fig. 9
(profiler ablation), Fig. 10a–b (sensitivity). Omitted: Fig. 4 (conditional-compilation
code listing; described in notes). Tables reproduced: Table 2 (use cases), Table 3
(max packet depth), Table 5 (wall-clock). Omitted: Table 1 (notation, in appendix math
slide instead), Table 4 (67 candidate features, too long).

**AC-DC (arXiv 2302.11718)** — included: Fig. 1 (TTD pipeline phases), Fig. 2
(framework), Fig. 3 (throughput vs. traffic rate), Fig. 4 (minimum memory), Fig. 5 (F1
vs. TTD at minimum memory), Fig. 6 (batch size vs. memory availability), Fig. 7 (batch
size vs. traffic rate). Tables reproduced: Table 1 (flow-statistics vs. packet-capture
trade-off), Table 2 (dataset), Table 3 (F1 and TTD with no memory constraint). Omitted:
appendix Figs. 8–11 (baseline TTD/memory detail, packet-size distributions, fine-grained
memory), Table 4 (37 header fields), Table 5 (bits and importance per feature), Table 6
(top-15 pool members); Algorithm 1 is paraphrased on the framework slide. Equations 1–2
are in the math appendix.

**ServeFlow (arXiv 2402.03694)** — included: Fig. 2 (latency–accuracy teaser), Fig. 4
(architecture). Tables reproduced: Table 1 (F1 vs. packets), Table 2 (compute/inference
time). Omitted: Fig. 3 (flow-collection CDFs), Fig. 5 (Pareto front of candidate
models; described in a bullet), Fig. 6–13 and Tables 3–9 (evaluation detail beyond the
two-slide budget). Seven converted but unused PNGs were deleted from `images/`.

## What to verify before teaching

- Layout was verified on 2026-09-16 by printing `slides.html?print-pdf` with headless
  Google Chrome and rasterizing every page: 55 pages for title plus 54 slides, no slide
  spills to a second page. Every image carries an explicit `{height="…px"}` cap (185–250
  px for four-across rows, 200 px for three-across, 280–340 px for two-across, up to
  420 px for single figures). To re-check after edits:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new
  --print-to-pdf=out.pdf "file://$PWD/slides.html?print-pdf"` then `pdfinfo out.pdf`
  (page count should equal slide count plus one) and `pdftoppm -r 50 -png`.
- The KSWIN critical-value expression in the appendix is the standard two-sample KS
  approximation, not copied from the paper; the paper gives no formula.
- The deck asserts that earlier lectures covered the COVID traffic shift, adversarial
  adaptation, and the TTL/source-IP spurious-feature example; adjust the lecture names on
  the "We Have Seen This Before" slide if those were taught elsewhere.

## Source

- Authored from the three paper sources above (no `_source-extract.md`; this is a new
  deck with no pptx predecessor). 44 content slides plus 6 section dividers and a
  4-slide math appendix. `quarto render 23-Maintenance/slides.qmd` builds clean
  (verified 2026-09-16); every referenced image path exists.

## Verification (2026-09-16)

Independent pass over every figure and table against the LaTeX sources and PDFs
(figure numbers taken from `pdftotext` captions; every deck PNG was compared
pixel-for-pixel against a `pdftoppm -r 170` rendering of the candidate source
file, and all 50 matched exactly, so no PNG is cropped, rotated, or a wrong
panel). Sub-panel letters follow the `\subfloat`/`\subfigure` order in the tex.
Author lists on the three section dividers match the tex `\author` blocks.

### Figures (slide → image → source file → paper figure → result)

| Slide | Image | Source file | Paper fig. | Result |
|---|---|---|---|---|
| A Taxonomy of Drift | leaf-x-sudden.png | sudden.pdf | not in paper (source only) | OK |
| A Taxonomy of Drift | leaf-x-gradual.png | gradual.pdf | not in paper (source only) | OK |
| A Taxonomy of Drift | leaf-x-incremental.png | incremental.pdf | not in paper (source only) | OK |
| A Taxonomy of Drift | leaf-x-recurring.png | recurring.pdf | not in paper (source only) | OK |
| Drift Is Everywhere | leaf-fig1a-dvol.png | model_NRMSE.pdf | LEAF Fig. 1a (Volume) | OK |
| Drift Is Everywhere | leaf-fig1b-pu.png | model_NRMSE_peakactive_ues.pdf | LEAF Fig. 1b | OK |
| Drift Patterns Differ by KPI | leaf-fig1c-dtp.png | model_NRMSE_ue_downlink_throughput.pdf | LEAF Fig. 1c | OK |
| Drift Patterns Differ by KPI | leaf-fig1d-gdr.png | model_NRMSE_rtp_gap_duration_ratio_avg.pdf | LEAF Fig. 1d | OK |
| Drift Patterns Differ by KPI | leaf-fig7a-rest.png | model_NRMSE_rrc_establishmentatt.pdf | LEAF Fig. 7a (appendix) | OK |
| Drift Patterns Differ by KPI | leaf-fig7b-cdr.png | model_NRMSE_s1u_sip_calldrop.pdf | LEAF Fig. 7b (appendix) | OK |
| Drift Persists Regardless of Training Set | leaf-fig2a-size.png | size_NRMSE.pdf | LEAF Fig. 2a | OK |
| Drift Persists Regardless of Training Set | leaf-fig2b-period.png | period_NRMSE.pdf | LEAF Fig. 2b | OK |
| The LEAF Framework | leaf-fig3-framework.png | framework.pdf | LEAF Fig. 3 | OK |
| Step 1: Detect Drift | leaf-x-kswin.png | KSWIN.pdf | not in paper (source only) | **fixed caption**: figure also marks irregular-data alarms in late 2019 and April 2021 (matches Appendix B text), not only mid-2019 |
| LEAplot | leaf-fig4-leaplot-g1.png | leaplot_group1.pdf | LEAF Fig. 4 = Fig. 8a | OK |
| LEAplot | leaf-fig8b-leaplot-g2.png | leaplot_group2.pdf | LEAF Fig. 8b (appendix) | OK |
| LEAgram | leaf-fig5a-leagram.png | directed_leagram.png | LEAF Fig. 5a | OK |
| LEAgram | leaf-fig5b-leagram-mitigated.png | directed_leagram_mitigated.png | LEAF Fig. 5b | OK (source PNG has a transparent border; renders on slide background) |
| Cost vs. Accuracy | leaf-fig6a-dvol-tradeoff.png | Dvol_tradeoff.pdf | LEAF Fig. 6a | OK |
| Cost vs. Accuracy | leaf-fig6b-cdr-tradeoff.png | CDR_tradeoff.pdf | LEAF Fig. 6b | OK |
| Trade-Off for the Other Four KPIs | leaf-fig10a-pu-tradeoff.png | PU_tradeoff.pdf | LEAF Fig. 10a | OK |
| Trade-Off for the Other Four KPIs | leaf-fig10b-dtp-tradeoff.png | DTP_tradeoff.pdf | LEAF Fig. 10b | OK |
| Trade-Off for the Other Four KPIs | leaf-fig10c-rest-tradeoff.png | REst_tradeoff.pdf | LEAF Fig. 10c | OK |
| Trade-Off for the Other Four KPIs | leaf-fig10d-gdr-tradeoff.png | GDR_tradeoff.pdf | LEAF Fig. 10d | OK (speaker note fixed: naive-90 is also below zero for GDR, −4.20%, Table 8) |
| LEAF Over Time | leaf-fig9a-dvol-mit.png | CatBoost_pdcp_vol_dl_drb_mb_nrmse_e2e_mitigation.pdf | LEAF Fig. 9a | **fixed bullet**: LEAF is the orange line, Static is blue |
| LEAF Over Time | leaf-fig9b-pu-mit.png | CatBoost_peakactive_ues_nrmse_e2e_mitigation.pdf | LEAF Fig. 9b | OK |
| LEAF Over Time | leaf-fig9d-rest-mit.png | CatBoost_rrc_establishmentatt_nrmse_e2e_mitigation.pdf | LEAF Fig. 9d | OK |
| Accuracy Is Not the Only Objective | cato-fig1-pipeline.png | serving_pipeline.pdf | CATO Fig. 1 | OK |
| Representation Is a Cost Knob | cato-fig2a-depth-f1.png | pktdepth_vs_perf.pdf | CATO Fig. 2a | OK |
| Representation Is a Cost Knob | cato-fig2b-depth-cost.png | pktdepth_vs_cost.pdf | CATO Fig. 2b | OK |
| CATO: Optimizer Plus Profiler | cato-fig3-design.png | design.pdf | CATO Fig. 3 | OK |
| Pareto Fronts: Latency | cato-fig5a-iot-latency.png | compare_latency_iot.pdf | CATO Fig. 5a | OK |
| Pareto Fronts: Latency | cato-fig5b-video-latency.png | compare_latency_video.pdf | CATO Fig. 5b | OK |
| Pareto Fronts: Live Traffic | cato-fig5c-app-latency.png | compare_latency_app.pdf | CATO Fig. 5c | OK |
| Pareto Fronts: Live Traffic | cato-fig5d-app-throughput.png | compare_throughput_app.pdf | CATO Fig. 5d | OK |
| How Good Is the Search? | cato-fig7-pareto-quality.png | pareto_quality.pdf | CATO Fig. 7 | OK |
| How Good Is the Search? | cato-fig8-convergence.png | convergence.pdf | CATO Fig. 8 | OK |
| Why Measure Instead of Estimate | cato-fig9-profiler-ablation.png | profiler_ablation.pdf | CATO Fig. 9 | OK |
| Sensitivity and Wall-Clock Cost | cato-fig10a-damping.png | damp_convergence.pdf | CATO Fig. 10a | OK |
| Sensitivity and Wall-Clock Cost | cato-fig10b-init.png | init_convergence.pdf | CATO Fig. 10b | OK |
| CATO vs. Traffic Refinery | cato-fig6-traffic-refinery.png | compare_traffic_refinery.pdf | CATO Fig. 6 | OK |
| AC-DC: The Environment Drifts Too | acdc-fig1-ttd.png | tdd.pdf | AC-DC Fig. 1 | OK |
| AC-DC: Pool and Scheduler | acdc-fig2-framework.png | module.drawio.pdf | AC-DC Fig. 2 | OK |
| AC-DC Results: Throughput | acdc-fig3-throughput.png | eval_unlimited_handled_rate.pdf | AC-DC Fig. 3 | OK |
| AC-DC Results: Memory | acdc-fig4-memory.png | mem_comparison_eval.pdf | AC-DC Fig. 4 | **fixed caption**: the 118.8x / 126x figures are for the MPR = 0.85 configuration (1.06 GB vs. 126.81 / 134.37 GB), as in the paper's MPR paragraph |
| AC-DC Results: Memory | acdc-fig5-tradeoff.png | ttdvsperf_minimum_mem.drawio.pdf | AC-DC Fig. 5 | OK |
| AC-DC Adapts | acdc-fig6-batch-vs-memory.png | dcm_batch_mem_fix_r.pdf | AC-DC Fig. 6 | OK |
| AC-DC Adapts | acdc-fig7-batch-vs-rate.png | dcm_batch_fix_m.pdf | AC-DC Fig. 7 | OK |
| ServeFlow | sf-arch.png | ServeFlow.pdf | ServeFlow Fig. 4 | OK |
| ServeFlow | sf-teaser.png | teaser.pdf | ServeFlow Fig. 2 | OK |

### Tables (slide → paper table → result)

| Slide | Paper table | Result |
|---|---|---|
| The Setting | LEAF Table 1 (`tab:data`) | OK, all cells match |
| Six Forecasting Targets | LEAF Table 2 (`tab:target_KPIs`, Evolving) | OK, all cells match |
| Naive Periodic Retraining | LEAF Table 3 (`tab:retraining`) | OK, all 36 values and retrain counts match |
| Results: Tree Ensembles | LEAF Table 4 (`tab:delta_nrmse`), CatBoost + ExtraTrees | OK, values, counts, and bold (gray) cells match |
| Results: LSTM and KNN | LEAF Table 4, LSTM + KNeighbors | OK, values, counts, and bold cells match |
| LEAF Over Time | LEAF Table 7 (`tab:kpi_error`) | OK |
| Does It Hold When the Network Grows? | LEAF Table 5 (`tab:delta_nrmse_evolv`) | OK, including the double-bold GDR and DTP rows |
| Step 3: Mitigate (dispersion table) | not a paper table; summarizes Section 4.3 text | OK (linear/cubic weights, 95% threshold match) |
| Use Cases | CATO Table 2 (`tab:use_cases`) | OK |
| Packet Depth: How Long to Wait | CATO Table 3 (`tab:vary_max_depth`) | OK, all 42 cells match |
| Sensitivity and Wall-Clock Cost | CATO Table 5 (`tab:wallclock`, appendix) | OK |
| AC-DC: The Environment Drifts Too | AC-DC Table 1 (`tab:overalltradeoff`) | OK |
| AC-DC Results: Throughput | AC-DC Table 3 (`tab:f1eval`) | OK, bold AC-DC row as in tex |
| AC-DC Results: Throughput | AC-DC Table 2 (`tab:stats`) | OK |
| Most Flows Are Easy | ServeFlow Table 1 (`tab:f1_scores`) | OK |
| Most Flows Are Easy | ServeFlow Table 2 (`tab:time_bottleneck`) | OK |

### Prose claims checked

All quantitative claims in slide bodies and speaker notes were checked against
the tex (412 / 898 eNBs, 699,381 / 1,084,837 logs, 224 KPIs, 180-day horizon,
18x training speed-up, 169 retrains, 47.79%, 30.8% fewer retrains, 0.34–2.83%,
32.68%, 10.3–76.9% and 17–71.8%, 7.71–50.13%, 71.52%, 44.56%, 2x–4x dispersion,
32 features in Group 1, top 5% suburban; CATO 3,200 configs / 5 days / 7,000
years, 11–79x, 817–2000x, 3600x, 2.2–2900x, 1.6–3.7x, 2^67 × 50, 0.970 / 7.9 s
vs. 0.979 / 0.1 s, 0.963 / 0.962 / 0.960, 2.6x / 19x, 37%, HVI 0.98 / 0.88 /
0.86 / 0.77, 1.6%, 0.95 / 0.39 / 0, 87 / 240 / 1,295 / 1,469 iterations, 2.76x,
δ = 0.4, 344 ns, 9.5 h / 2 h / 546.7 s, 67 features / 1,600 lines; AC-DC 855x,
163x, 76.3% / 74.7% / 30%, 0.753 / 0.99, 90 classifiers, 176.95x / 48.98x, 117%,
0.089 / 0.12, 0.80, 118.8x / 126x, 3,500 flows/s, 15,000 flows/s, 10 GB, worked
example 6.75 GB; ServeFlow 1.8x–141.3x, 10^3–10^6 ms, 76.3%, 16 ms, 40.5x,
48.5k flows/s, 16 cores). Two prose fixes beyond the captions above:

- "Prior work picks 10, 50, or 'all' packets with little justification" →
  "10, 50, or 100 packets ..., or waits for the whole flow" (CATO Section 2 says
  10, 50, or 100).
- Speaker note on the Fig. 10 slide: "only LEAF is below zero" for GDR →
  naive 90-day retraining is also below zero (−4.20%, Table 8).

Not verifiable from the sources: the four drift-taxonomy panels and the KSWIN
figure are not referenced anywhere in the tex (not even commented out), so the
labels "sudden / gradual / incremental / recurring" rest on the source file
names only; the "recurring" series (`recurring.pdf`) reads more like a
two-level regime change than a strict recurrence. `quarto render` verified
clean after the edits.
