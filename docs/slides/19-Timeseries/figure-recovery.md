# Figure recovery: 20-Timeseries

Source: `docs/slides/20-Timeseries.pptx`, 7 slides, rendered 2026-10-16 (headless LibreOffice, 110 dpi; no pen ink, no animation builds in the archive).

Outcome: **nothing to recover.** The legacy deck has no drawn diagrams. Its only pictures are four embedded PNGs that the earlier extraction already saved to `images/` and that `slides.qmd` already uses (`s02-i01.png` on "Network Latency: A Real Time Series"; `s05-i02/03/04.png` on the appendix ARMA, MA, and AR slides). The remaining slides are a title, two section dividers, and three text-only bullet slides whose content the Quarto deck already covers in expanded form ("ARIMA: Adding Integration (Differencing)", "Seasonal ARIMA (SARIMA)").

| pptx slide | class | target Quarto slide | figure name | notes |
|---|---|---|---|---|
| 1 | skip | | | title slide |
| 2 | media | "Network Latency: A Real Time Series" | (existing `images/s02-i01.png`) | one embedded latency plot, already embedded; nothing drawn around it |
| 3 | skip | | | "Anomaly Detection" divider, empty body |
| 4 | skip | | | "Timeseries Analysis" divider |
| 5 | media | appendix "Autoregressive Models (AR)", "Moving-Average Models (MA)", "ARMA: Putting It Together" | (existing `images/s05-i04.png`, `s05-i03.png`, `s05-i02.png`) | three equation images beside bullets, all already embedded; the Quarto deck also typesets the same equations in LaTeX |
| 6 | skip | | | "ARIMA Parameters" bullets only; covered by the p/d/q table on "ARIMA: Adding Integration (Differencing)" |
| 7 | skip | | | "Seasonal ARIMA" bullets only; covered by "Seasonal ARIMA (SARIMA)" |

No images captured, no slides added, no speaker-notes rows (the deck has no `speaker-notes.md`).
