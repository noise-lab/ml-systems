# 20-Timeseries — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026**: Added a verified current-events vignette using the **CESNET-TimeSeries24** dataset
  (Luxemburk et al., *Scientific Data*, February 2025; doi:10.1038/s41597-025-04603-x).
  The dataset covers 40 weeks of real ISP traffic from 275,000 IP addresses and was published
  specifically to address over-optimism in lab-based anomaly detection benchmarks.
  Date-stamped source: nature.com/articles/s41597-025-04603-x (accessed June 2026).

- **2026**: Added a slide on time-series foundation models (TimesFM, Chronos, Moirai) based on
  verified 2025 sources including the NeurIPS 2025 workshop on time-series foundation models and
  the MachineLearningMastery.com 2026 round-up. These models are not fabricated; each has a
  primary publication or product page.

## Suggested missing coverage on broad themes (point 3)

- **Vector Autoregression (VAR)**: The deck covers univariate ARIMA thoroughly but only mentions
  multivariate models (VAR) in the comparison table. A worked example of forecasting two correlated
  network metrics simultaneously (e.g., bytes and packets per second) would strengthen the
  multivariate section.
- **Changepoint detection**: Distinct from anomaly detection — algorithms like PELT or BOCPD detect
  structural breaks (e.g., a routing change that permanently shifts latency). Not covered; could add
  one slide.
- **Online / streaming ARIMA**: The deck mentions "streaming ARIMA" in the comparison table but
  does not explain how sliding-window re-fitting or recursive least squares makes ARIMA usable in
  real-time monitoring. Worth adding for students going into operations.
- **Evaluation metrics for forecasting**: MAE, RMSE, MASE, and their relationship to anomaly
  detection thresholds. Currently only the AIC/BIC criteria appear.
- **Hands-on component**: The deck would benefit from a Jupyter notebook companion using
  `statsmodels.tsa` on the CESNET-TimeSeries24 or a sample CAIDA/RIPE trace.

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events refresh"
(web-verify; swap only for something fresher and confirmed). Items likely to age:

- **CESNET-TimeSeries24 vignette**: Published Feb 2025. Likely to remain a canonical benchmark
  for 1-2 more years. Re-verify: has it been superseded by a larger or more widely adopted
  dataset? Check for follow-up papers citing it.
- **Foundation models table** (TimesFM, Chronos, Moirai): This space is moving rapidly.
  By 2027, these may all have successors; Google TimesFM 2.5 is already out as of mid-2025.
  Re-verify version numbers, parameter counts, and whether any have been deprecated.
- **NeurIPS 2025 workshop reference**: Referenced in speaker notes for the foundation models
  slide. Confirm it ran and check if a proceedings volume exists.
- **"Largely supplanted" claim for LSTMs**: The book (Chapter 5) notes that LSTMs/GRUs have
  been "mostly supplanted" by foundation models. Monitor whether this consensus holds or whether
  lightweight LSTM-based systems regain favor for edge deployment.

Stronger alternative vignettes flagged but not used:
- **MamNet (arxiv 2507.00304, 2025)**: Hybrid Mamba/attention model for network traffic
  forecasting. Could replace or complement the CESNET vignette if published in a peer-reviewed
  venue (it was arxiv-only as of June 2026).

## Curated images

Four images were available in `images/`:

- **s02-i01.png** — USED. Real latency time series for multiple destinations. Valuable for
  motivating the lecture (showing what network time-series data looks like, with visible spikes).
- **s05-i02.png** — USED. Renders the full ARMA(p,q) combined equation. Useful on the ARMA slide.
- **s05-i03.png** — USED. Renders the MA(q) formula. Used on the Moving-Average slide.
- **s05-i04.png** — USED. Renders the AR(p) formula. Used on the Autoregressive slide.

All four images are genuinely useful (equations or data plots) — none were clip-art or decorative.

## Source

- Rebuilt from `_source-extract.md` (7 original slides, very sparse) + `agenda.md` Meeting 13
  (RNN / sequential data section) and Meeting 6 (time series representation types).
- Book alignment: Chapter 5 (Supervised Learning), sections "Recurrent Neural Networks," "LSTM
  and GRU Architectures," and "RNNs for Networking" in "Machine Learning for Networking"
  by Nick Feamster et al.
- The original PowerPoint deck covered only ARMA/ARIMA at a surface level. This rebuild
  substantially expands the content to include SARIMA, RNNs/LSTMs, anomaly detection framing,
  foundation models, and practical pitfalls — all grounded in the book's Chapter 5.
- Where the original slides and the book disagree: the original deck treated ARMA as the primary
  model without mentioning deep learning; the book positions classical models as baselines and
  emphasizes RNNs/LSTMs as the more powerful modern approach (now itself being supplanted by
  foundation models). This deck follows the book's ordering and emphasis.
