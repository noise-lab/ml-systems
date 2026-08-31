# 10-LinearRegression — instructor notes

Instructor-facing companion to `slides.qmd`. Not shown to students.

## Current-events updates made (point 2)

- **2026:** Added a `.vignette` box citing a September 2025 peer-reviewed paper:
  "A Machine Learning Approach to Investigating Key Performance Factors in 5G
  Standalone Networks," *Electronics*, DOI 10.3390/electronics14193817, published
  26 September 2025. Verified via DOI link and ResearchGate. The paper builds a 5G
  standalone testbed and compares Linear Regression, Decision Tree, Random Forest,
  Gradient Boosting, and XGBoost for KPI prediction — a real, dated example of
  linear regression used as a baseline in network engineering research.

## Suggested missing coverage on broad themes (point 3)

- **Logistic regression:** The book's Chapter 5 covers logistic regression
  (classification) immediately after linear regression, including the sigmoid function
  and the DNS query/response example. The source slides did not include logistic
  regression in this deck; it could be its own short deck or appended here.
- **Cross-validation in practice:** The deck introduces polynomial degree as a
  hyperparameter to tune with cross-validation, but does not show the train/validation
  error curves or the k-fold procedure. Adding one slide with the train-vs-validation
  error curve (as a function of polynomial degree) would sharpen the connection to
  the bias-variance diagram.
- **Network-specific data pitfalls:** The ACK packet problem is covered; another
  important pitfall is **concept drift** in traffic data (traffic patterns change
  over time, so a model trained on Monday traffic may fail on weekend traffic).
  Worth a brief mention when discussing training vs. test distribution assumptions.
- **sklearn pipeline demo:** A short code snippet (`PolynomialFeatures` →
  `StandardScaler` → `Ridge`) would ground the abstract math in something students
  can run immediately. Currently covered in labs; could be referenced from slides.
- **Coordinate descent / path algorithms for Lasso:** The deck notes there is no
  closed-form for Lasso but does not explain what solver is used. A brief mention
  of coordinate descent or the LARS algorithm would round out the picture for
  students who ask "so how does it actually solve it?"

## Next-year refresh notes

Refresh the dated content below per `../TEMPLATE.md` → "Annual current-events
refresh" (web-verify; swap only for something fresher and confirmed).

- **2025 vignette (Electronics, DOI 10.3390/electronics14193817):** This paper is
  from September 2025 and will feel dated by 2027-2028. Re-search for a more recent
  paper comparing linear regression baselines in a 5G / 6G or network-ML context.
  Good search term: `"linear regression" "baseline" 5G OR network performance
  prediction` in Google Scholar filtered to the past year.
- **5G framing:** By 2027 the "5G is new" angle will be stale; the vignette may need
  to shift to Open RAN, 6G, or satellite/LEO network ML (e.g., Starlink latency
  prediction). Watch for papers in IEEE INFOCOM, IMC, or ACM SIGCOMM.
- **sklearn / scikit-learn version numbers:** If any lab materials or linked notebooks
  reference a specific sklearn version, verify compatibility with whatever Python
  stack the course is using that year.

## Curated images

| Image | Used? | Rationale |
|---|---|---|
| `s03-i01.png` | Yes | Scatter plot (temperature vs. park people) — good intuition before math |
| `s03-i02.png` | Yes | Fitted line with residuals — illustrates RSS visually |
| `s03-i03.png` | No | Least-squares formula image ($m^*, b^*$) — rendered as inline LaTeX instead |
| `s05-i04.png` | No | Multiple-input formula ($\hat{Y} = \beta_0 + \sum X_j\beta_j$) — rendered inline |
| `s05-i05.png` | No | Compact matrix form ($\hat{Y} = X^T\hat{\beta}$) — rendered inline |
| `s05-i06.png` | No | RSS formula (summation form) — rendered inline |
| `s05-i07.png` | No | RSS matrix form — rendered inline |
| `s05-i08.png` | No | Normal equation derivative — rendered inline |
| `s05-i09.png` | No | Closed-form $\hat{\beta} = (X^TX)^{-1}X^Ty$ — rendered inline |
| `s06-i10.png` | No | Journal paper header (short-term traffic forecasting) — decorative/citation-only |
| `s08-i11.png` | No | Z-score formula ($z_j = \hat{\beta}_j/\hat{\sigma}\sqrt{v_j}$) — rendered inline |
| `s08-i12.png` | No | F-statistic formula — rendered inline |
| `s12-i13.png` | No | Basis expansion formula ($f(X) = \sum \beta_m h_m(X)$) — rendered inline |
| `s14-i14.png` | Yes | Polynomial basis functions plot — genuinely teaches the shape of $\phi_k$ |
| `s16-i15.png` | Yes | Gaussian RBF basis functions plot — illustrates locality of RBFs |
| `s23-i16.png` | Yes | Bias-variance tradeoff diagram — canonical teaching figure |
| `s26-i17.png` | No | Ridge regression argmin formula — rendered inline |
| `s26-i18.png` | No | Ridge closed form ($\hat{\beta}^{\text{ridge}} = (X^TX+\lambda I)^{-1}X^Ty$) — inline |
| `s27-i19.png` | No | Lasso argmin formula — rendered inline |
| `s28-i20.png` | No | Elastic net penalty formula — rendered inline |

Formula images were replaced with inline LaTeX for consistent rendering and
accessibility. Plots that teach visual intuition (scatter, fit, polynomial basis,
RBF, bias-variance curve) were retained.

## Source

- Rebuilt from `_source-extract.md` (29 slides) aligned to the "Linear Models"
  section of Chapter 5 ("Supervised Learning") of *Machine Learning for Networking*
  (`text/supervised.rst`).
- Where source slides and book disagreed (terminology and notation), the book's
  notation was used ($\mathbf{w}$ / $\mathbf{x}$ dot-product form in introduction,
  $\beta$ in matrix derivations matching the book's treatment).
- The book's "ACK packet practical pitfall" subsection was added as a slide; the
  original source slides did not cover it but it is a strong networking-specific
  teaching point in the book.
- Logistic regression (Chapter 5, book) was deliberately excluded from this deck as
  it is a classification model and warrants its own lecture (see "Suggested missing
  coverage" above).
