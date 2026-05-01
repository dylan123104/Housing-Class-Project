# Final Report Outline — ISyE 521
## "Predicting Moscow Apartment Prices Using Machine Learning"

**Authors:** Mason Reitmeier, Dylan Hicks, Justin Zens
**Course:** ISyE 521 — Machine Learning in Action
**Page budget:** 8 single-spaced pages (excluding appendices)

> **How to use this file:** This is a structural guide for the final report, not the report itself. Each section below has a "writing prompt" describing the rhetorical goal, followed by subsection bullets that say what to cover. Italic lines describe figures/tables to insert. Inline numbers come from `CLAUDE.md` so you can quote them directly without re-deriving anything. Sections are ordered as required by `project_instructions.pdf` (Introduction, Data, Methods, Results, Discussion, Conclusions and Future Directions, References) with an Abstract added at the top.

---

## 1. Abstract (~0.5 pp)

**Writing prompt:** The abstract is the only thing many graders will read in full. Mirror the separately submitted abstract structure (Problem, Data, Methods, Results, Conclusions) but tighten where possible. Every sentence earns its place.

### 1.1 Problem statement (1–2 sentences)
- Moscow's housing market has unusually high price variance and identical-spec apartments can swing tens of millions of rubles
- Frame the three research questions in compressed form:
  1. How accurately can apartment prices be predicted?
  2. Which features do Moscow residents prioritize?
  3. Which features matter most for prediction?

### 1.2 Data (1 sentence)
- 18,374 cleaned apartment listings from a Kaggle Moscow dataset, with 12 features after preprocessing

### 1.3 Methods (1–2 sentences)
- Eight models trained on a unified preprocessing pipeline with an 80/20 holdout: linear (OLS, Lasso, Ridge), trees (Decision Tree untuned and tuned, Random Forest tuned), boosting (AdaBoost tuned), deep learning (PyTorch MLP), and a Stacking ensemble
- Metrics: Train/Test R², RMSE on log scale, MAE in rubles, MAPE percent

### 1.4 Results (1–2 sentences)
- Random Forest is the strongest model: Test R² = 0.946, RMSE log = 0.249, MAPE = 15.07%
- Stacking essentially tied RF (Test R² = 0.945, MAPE = 14.93%); tree-based models outperformed linear baselines by roughly 8 R² points

### 1.5 Conclusions (1 sentence)
- Area dominates predictive importance; renovation tier (Designer premium) and Moscow city location command the largest buyer-signal premiums

---

## 2. Introduction (~0.75 pp)

**Writing prompt:** Sell the reader on why the problem matters before naming any models. The presentation rubric explicitly weights problem motivation, so mirror that emphasis here.

### 2.1 Background and motivation
- Moscow has one of the most volatile urban housing markets in Europe; identical-spec apartments can differ by tens of millions of rubles
- Accurate price prediction supports buyers (avoid overpaying), sellers (set listing prices), and developers (project ROI)
- ML is well-suited to this domain because price is shaped by interactions among location, size, and quality features that linear pricing rules miss

### 2.2 Problem statement
- State the three research questions verbatim from `CLAUDE.md`:
  1. How accurately can apartment prices in Moscow be predicted?
  2. Which apartment features do Moscow residents prioritize (i.e., most influence price)?
  3. What features are most important for price prediction?
- Important distinction to flag for the reader:
  - **Q2 vs Q3** — Q2 is best answered by interpretable linear coefficients (what *buyers* signal they value through pricing); Q3 is best answered by non-linear feature importance (what the *model* leans on, including interactions)

### 2.3 Approach summary (one paragraph)
- Cleaned the 22,676-row Kaggle dataset down to 18,374 valid records
- Engineered features and one-hot encoded categoricals
- Trained 8 models on a shared pipeline using an 80/20 holdout
- Compared performance with R², log-RMSE, MAE, and MAPE
- Analyzed feature importance via standardized linear coefficients (for buyer signal) and Random Forest permutation importance (for predictive importance)

### 2.4 Report roadmap (2–3 sentences)
- One sentence per remaining section so the reader knows what is coming
- Example: "Section 3 describes the dataset and cleaning pipeline; Section 4 details the modeling pipeline and 8 models; Section 5 reports performance; Section 6 ties results back to the research questions; Section 7 closes with conclusions and future work."

---

## 3. Data (~1.25 pp)

**Writing prompt:** Lead with provenance, then walk the reader through the cleaning funnel. End with the log transformation since it sets up the modeling section.

### 3.1 Data source
- Kaggle: Moscow apartment listings dataset (cite URL in References)
- 22,676 raw rows × 12 columns
- Self-reported listings, not transaction prices — flag now and revisit in Limitations
- Snapshot vintage (~2021); briefly note generalization implications

### 3.2 Variable description
- Target: **Price** (₽)
- Numeric features (7): Minutes to metro, Area, Living area, Kitchen area, Floor, Number of floors, Number of rooms
- Categorical features (4): Apartment type (2 levels), Metro station (317 levels), Region (Moscow vs Moscow region), Renovation (4 levels: Cosmetic, Designer, European, No renovation)
- Note: Number of rooms has 12 distinct values; treated here as ordinal numeric (fewer parameters than 12 dummies)
- *Table 1: Variable name, type (numeric / categorical / target), levels or range, brief description*

### 3.3 Cleaning pipeline (walk through with row counts)
- **Step 1 — Whitespace and types:** strip strings, coerce numeric columns via `pd.to_numeric`; zero null values introduced
- **Step 2 — Missing values and duplicates:** no missings; 1,851 duplicate rows dropped → 20,825 rows
- **Step 3 — Categorical validation:** confirmed unique counts (Apartment type 2, Region 2, Renovation 4, Metro station 317)
- **Step 4 — Sanity-check filters:** dropped 2,451 rows that violated:
  - Floor > Number of floors (impossible)
  - Kitchen area > Total area (impossible)
  - Living area + Kitchen area > Total area (impossible)
- **Step 5 — Index reset.** Final shape: 18,374 × 12
- *Figure 1: Cleaning funnel — bar chart or Sankey diagram showing 22,676 → 20,825 → 18,374*

### 3.4 Target transformation
- Raw Price skewness ≈ 7.63 (heavy right tail; outliers dominate the loss for any squared-error model)
- Log transformation reduces skew to ≈ 1.13 — still moderate, but workable
- Justifies using `Log_Price` as the target for linear models (closer to homoscedastic, more normal residuals)
- Tree models do not require it but use the same target for cross-model comparability
- *Figure 2: Side-by-side histograms — Price (raw) vs Log_Price*

### 3.5 Why Metro station was dropped
- 317 unique values would explode dimensionality under one-hot encoding
- Replaced by Region (Moscow city vs oblast) plus Minutes to metro (numeric proximity)
- Acknowledge this as a deliberate trade-off; finer-grained location signal is lost (revisited in Limitations and in Future Directions)

---

## 4. Methods (~1.75 pp)

**Writing prompt:** This is the densest section. Start with the shared pipeline, then go model-by-model in increasing complexity. The reader should be able to reproduce the work from this section alone.

### 4.1 Feature engineering
- One-hot encoding (`drop_first=True`, `dtype=int`) on Apartment type, Region, Renovation
- Number of rooms kept as numeric (ordinal treatment)
- Reference categories established by `drop_first`: **Cosmetic** renovation, **Moscow** city, **New building** apartment type — coefficients are interpreted relative to these
- Final feature matrix: **12 features** (7 numeric + 5 dummies)

### 4.2 Train/test split
- 80/20 split, `random_state=42` for reproducibility → 14,699 train rows / 3,675 test rows
- Same test set is used to evaluate every model — flag this honestly here, then revisit as a limitation in Section 6

### 4.3 Preprocessing pipeline
- sklearn `ColumnTransformer`:
  - `StandardScaler` on the 7 numeric features (zero mean, unit variance)
  - One-hot dummies passed through unscaled (already 0/1)
- Why a shared pipeline: standardized coefficients across linear models are directly comparable
- Tree-based models do not need scaling but use the same pipeline for code uniformity
- *Figure 3: Pipeline block diagram — raw data → cleaning → ColumnTransformer → train/test split → 8 models → metrics*

### 4.4 Models (one short paragraph each, in increasing complexity)

#### OLS Linear Regression
- Baseline; no tuning
- Tells the reader how far simple linearity gets

#### Lasso Regression
- 10-fold `LassoCV` to pick alpha automatically
- Best alpha = 0.000852
- Zeroed out only `Floor` (11 of 12 features kept)

#### Ridge Regression
- 10-fold `RidgeCV` to pick alpha automatically
- Best alpha = 6.5793
- Kept all 12 features
- Designed to handle Area / Living area collinearity

#### Decision Tree (untuned)
- Fully grown
- Included to baseline overfitting behavior (Train R² = 1.00 vs Test R² = 0.887)

#### Decision Tree (Tuned)
- `GridSearchCV`, 5-fold, 45 combinations
- Best: `max_depth=10, min_samples_leaf=10, min_samples_split=2`
- Label is **"Tuned"** rather than "Optimized" because the search space is small

#### Random Forest (Tuned)
- `RandomizedSearchCV`, `n_iter=20` of 324 candidates, 3-fold
- Best: `n_estimators=300, max_depth=20, max_features='sqrt', min_samples_leaf=1`
- Why Randomized: full grid was infeasible; sampling explores trade-offs efficiently with a fixed budget

#### AdaBoost (Tuned)
- `GridSearchCV`, 5-fold, 45 combinations
- Base learner fixed at `DecisionTreeRegressor(max_depth=3)`
- Best: `n_estimators=300, learning_rate=0.05, loss='exponential'`
- Note: `n_estimators=300` is the grid ceiling; results are best within the constrained search

#### Neural Network (PyTorch MLP)
- Architecture: **128 → 64 → 32 → 1**, ReLU activations, Dropout(0.2)
- Optimizer: Adam, `weight_decay=1e-3` (L2 regularization)
- GPU training (CUDA)
- Two-phase training: 75/25 val split for monitoring loss curves and confirming architecture, then a fresh model retrained on the full X_train (14,699 rows) for the reported metrics
- 300 epochs final; `y_train.values.copy()` used to avoid the PyTorch non-writable tensor warning

#### Stacking Ensemble
- sklearn `StackingRegressor`
- 5 base learners: Ridge, Tuned DT, Tuned RF, AdaBoost, PyTorch MLP (wrapped via a custom `TorchMLPRegressor` sklearn-compatible class)
- Meta-learner: `RidgeCV`
- 5-fold `KFold` (shuffle=True, random_state=42)
- `n_jobs=1` on the outer StackingRegressor required to avoid PyTorch/CUDA multiprocessing conflicts; RF's internal `n_jobs=-1` still parallelizes within each fold

### 4.5 Hyperparameter tuning strategy
- **Grid for small spaces** (DT, AdaBoost — 45 combos); **Random for large** (RF — 20 of 324 sampled)
- Justification of choice belongs in this paragraph

### 4.6 Evaluation metrics
- **Train R² and Test R²** — goodness-of-fit and overfit gap
- **RMSE on log scale** — primary accuracy metric (log-rubles); robust to outliers
- **MAE in rubles** — context metric; ruble-denominated absolute error
- **MAPE percent** — context metric; percentage error
- **Why RMSE in rubles is intentionally dropped:** exponentiating log predictions amplifies outliers, so raw-ruble RMSE becomes misleading

---

## 5. Results (~2.25 pp)

**Writing prompt:** The leaderboard table is the centerpiece. Lead with EDA so the reader has feature intuition before the leaderboard, then drill into coefficients and feature importance. End with a best-model spotlight.

### 5.1 Exploratory findings
- **Numeric correlations vs Log_Price** (strongest to weakest):
  - Area 0.80, Living area 0.74, Number of rooms 0.67, Kitchen area 0.63
  - Floor 0.11, Number of floors 0.08
  - Minutes to metro -0.08
- **Multicollinearity flag:** Area × Living area = 0.91 (high; affects linear coefficient stability if not regularized)
- **Categorical effects from box plots:**
  - Designer renovation > European > No renovation > Cosmetic on median price
  - Moscow city has higher median than Moscow oblast
  - New building has a small premium over Secondary
- **Heteroscedasticity:** the Area-vs-Price scatter shows variance growing with size — another reason to use log target
- *Figure 4: Correlation heatmap — 7 numeric features and Log_Price*
- *Figure 5: 2×2 grid of box plots — Apartment type, Region, Renovation, Number of rooms vs Log_Price (log scale)*

### 5.2 Model performance leaderboard
- *Table 2: 9-row leaderboard with 5 metric columns. Sort or annotate by Test R² descending.*
- Columns: Model | Train R² | Test R² | RMSE (log) | MAE (₽) | MAPE (%)
- Numbers to fill in (from `CLAUDE.md`):
  - **OLS** — 0.8694 / 0.8694 / 0.3888 / 26.5M / 28.45%
  - **Lasso** — 0.8694 / 0.8694 / 0.3889 / 26.8M / 28.41%
  - **Ridge** — 0.8694 / 0.8694 / 0.3888 / 26.5M / 28.41%
  - **DT untuned** — 1.000 / 0.8868 / 0.3619 / 15.1M / 22.6%
  - **DT tuned** — 0.9473 / 0.9243 / 0.2961 / 13.1M / 19.0%
  - **RF tuned** — 0.9917 / 0.9464 / 0.2491 / 11.6M / 15.07%
  - **AdaBoost tuned** — 0.8861 / 0.8864 / 0.3626 / 15.1M / 30.32%
  - **Neural Net** — 0.9159 / 0.9065 / 0.3290 / 15.1M / 18.47%
  - **Stacking** — 0.9897 / 0.9450 / 0.2523 / 11.8M / 14.93%

### 5.3 Linear model coefficients (interpretation as buyer signal)
- All three linear models (OLS, Lasso, Ridge) agree on every coefficient within ~0.02 — coefficients are stable on the standardized scale (multicollinearity was not the disaster the EDA suggested, thanks to standardization)
- Top drivers (standardized; positive = price up):
  - **Designer renovation** +0.65 vs Cosmetic baseline
  - **No renovation** +0.41 vs Cosmetic — counterintuitive; likely confounded with new construction in this dataset (worth a sentence of caveat)
  - **Area** +0.40
  - **Region (Moscow oblast)** -0.43 vs Moscow city — strong location penalty
- Lasso zeroed out only `Floor`; Ridge kept all 12 — regularization did not materially change the story
- Note for the writer: the earlier perceived "Area jumped from 0.006 to 0.40" was a units/standardization artifact, not a multicollinearity fix
- *Figure 6: Side-by-side bar chart of standardized coefficients for OLS, Lasso, and Ridge — visually confirms agreement*

### 5.4 Tree-based feature importance
- **MDI (Mean Decrease in Impurity) from RF:** Top three are Area, Living area, Number of rooms
  - Caveat to state: MDI inflates correlated and high-cardinality features
- **Permutation importance from RF (authoritative ranking):**
  - Area 0.36 (dominant)
  - Apartment type 0.09
  - Region 0.07
  - Living area 0.05
  - Number of rooms drops from 3rd (MDI) to 8th (permutation), inflated previously by 0.91 correlation with Area
- Show both, but lead with permutation as the trustworthy interpretation
- *Figure 7: Side-by-side bar charts — RF MDI top-10 and RF permutation top-10*

### 5.5 Neural Network training behavior
- Two-phase training rationale recap (val phase to confirm architecture; full retrain for reported metrics)
- Loss curve: monotonic decrease on training; validation plateaued around epoch 200
- *Figure 8: Training and validation loss curves from the 75/25 monitoring phase*

### 5.6 Stacking ensemble weights
- Meta-learner (RidgeCV) coefficients on the 5 base predictions:
  - **RF: +0.955** (dominant)
  - **NN: +0.209** (secondary positive contribution)
  - DT tuned: -0.013 (essentially zero)
  - Ridge: -0.017 (essentially zero)
  - **AdaBoost: -0.134** (slight negative — adding it hurts)
- Interpretation: RF already captured most of the learnable signal; ensemble complexity does not justify the marginal MAPE gain
- *Figure 9: Bar chart of meta-learner coefficients across the 5 base learners*

### 5.7 Best model spotlight: Random Forest
- Test R² = 0.9464; explains ~95% of log-price variance
- RMSE log = 0.2491; MAE = ₽11.6M; MAPE = 15.07%
- Train R² = 0.9917 vs Test R² = 0.9464 — modest overfit gap; not catastrophic, especially given the depth=20 setting
- Stacking essentially ties RF on primary metrics (Test R² = 0.9450, MAPE = 14.93%) but adds substantial operational complexity for negligible improvement → recommend RF as the production choice
- One sentence here recommending RF is the clean takeaway

---

## 6. Discussion (~1.25 pp)

**Writing prompt:** Close the loop on the three research questions, then explain *why* the results came out this way, then own the limitations. The grader is looking for thoughtful interpretation, not just a recitation of metrics.

### 6.1 Answers to the three research questions

#### Q1 — How accurately can prices be predicted?
- Random Forest reaches Test R² = 0.946 with MAPE ≈ 15%
- Linear models cap out at ~0.87 — an 8-point R² gap that motivates non-linear models
- Practical takeaway: a typical RF prediction lands within ~15% of the listing price

#### Q2 — Which features do Moscow residents prioritize? (linear coefficients)
- **Renovation tier** is the single largest categorical effect (Designer +0.65 standardized)
- **Region** (Moscow city vs oblast) carries a -0.43 oblast penalty
- **Area** is the dominant numeric driver (+0.40)
- Why this question is best answered by linear coefficients: they reflect the pricing signal that buyers and sellers explicitly negotiate around

#### Q3 — Which features matter most for prediction? (RF permutation importance)
- **Area dominates** (0.36) — far ahead of any other feature
- **Apartment type** (0.09) and **Region** (0.07) rank next
- Living area (0.05) and Number of rooms appear less important than EDA suggested — both are absorbed by Area through the 0.91 correlation
- Why this question is best answered by permutation importance: it reflects what the model actually leans on, including non-linear interactions invisible to linear coefficients

### 6.2 Why tree-based models outperformed linear
- Non-linear interactions matter (e.g., Area × Renovation; Region × Apartment type) — linear models cannot represent these without explicit interaction terms
- Trees are insensitive to feature scale and outliers
- Trees are robust to multicollinearity (Area vs Living area), which directly addresses the largest data quality concern surfaced in EDA

### 6.3 Why Stacking did not beat Random Forest
- Meta-learner placed 95% weight on RF predictions
- The signal RF did not capture was largely noise (or the small marginal contribution from NN at +0.209)
- Stacking adds operational complexity (5 models to train, version, and monitor); the marginal MAPE gain of 15.07% → 14.93% does not justify it
- Honest recommendation: RF is the production-ready model

### 6.4 Why AdaBoost performed unexpectedly poorly on MAPE
- 30.32% MAPE despite a respectable Test R² = 0.886
- Exponential loss focuses on high-value outliers, sacrificing percentage accuracy on typical listings
- Useful broader lesson on **metric–objective alignment** — worth a sentence in the report

### 6.5 Practical implications
- **For buyers:** Designer renovation and Moscow city location command the largest premiums; budget-constrained buyers can save substantially by considering oblast listings
- **For sellers:** Area is the strongest lever; renovation upgrades pay off most clearly at the Designer tier
- **For developers:** Renovation tier ROI is the clearest investment signal in the data

### 6.6 Limitations
- **Single 80/20 holdout** rather than nested cross-validation — model ranking reflects one split; nested CV would give honest comparative confidence intervals
- **Listing prices, not transaction prices** — actual sales may diverge meaningfully from list
- **2021 snapshot** — Moscow market conditions and the ruble's valuation have shifted since
- **Metro station feature dropped** due to 317-category cardinality — finer-grained location signal lost; clustering metro stations into neighborhoods could recover much of it
- **Test set reuse caveat** — every model was tuned on the training data and evaluated on the same holdout; with 8 models there is a small selection-bias effect on the reported "best"
- **AdaBoost log-MSE objective vs MAPE reporting** — explicit metric mismatch contributed to its poor MAPE; a different loss could change AdaBoost's standing

---

## 7. Conclusions and Future Directions (~0.5 pp)

**Writing prompt:** Tight wrap-up. Restate the headline finding, then point at concrete next steps a follow-up team could pick up.

### 7.1 Summary (one paragraph)
- Random Forest is the best-performing model: Test R² = 0.946, MAPE = 15.07%
- Area dominates predictive importance; renovation tier and Moscow city location dominate the buyer signal
- Stacking ensemble confirmed RF had already absorbed most of the learnable signal — ensemble gains were marginal

### 7.2 Future directions
- **Gradient boosting** (XGBoost, LightGBM, CatBoost) — most promising natural extension; often beats Random Forest on tabular data
- **Nested cross-validation** for honest confidence intervals on model ranking
- **Geographic features beyond Region** — cluster the 317 metro stations into neighborhoods, add distance-to-city-center, district-level demographics
- **Time-series extension** if multi-year data becomes available — captures market dynamics
- **Transaction-price ground truth** instead of listing prices to address one of the largest limitations
- **Interpretability tools** (SHAP, partial dependence plots) for richer answers to Q3

---

## 8. References (~0.5 pp)

> List with full citations; format consistently (APA or IEEE — pick one and stick with it). Numbers below are placeholders.

- Kaggle Moscow apartment dataset — *insert exact URL and access date*
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32.
- McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference.
- Harris, C. R. et al. (2020). *Array programming with NumPy.* Nature 585, 357–362.
- Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering 9(3), 90–95.
- Waskom, M. (2021). *seaborn: statistical data visualization.* Journal of Open Source Software 6(60), 3021.
- Breiman, L. (2001). *Random Forests.* Machine Learning 45, 5–32.
- Freund, Y., and Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning.* JCSS 55(1), 119–139.
- Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks 5(2), 241–259.

---

## 9. Appendix (optional — does NOT count toward 8-page limit)

> Use the appendix only for material that is genuinely useful to a reproducer or skeptical grader. Do not pad.

### A.1 Hyperparameter grids
- Full grid for each tuned model: Decision Tree, Random Forest, AdaBoost, Neural Network
- Note which were Grid vs Randomized

### A.2 Linear coefficient tables
- Full OLS / Lasso / Ridge coefficient tables (all 12 features, not just the top drivers from Section 5.3)
- Include both standardized and unstandardized values if space permits

### A.3 Random Forest feature importance tables
- Full MDI ranking (all 12 features)
- Full permutation importance ranking (all 12 features)

### A.4 Untuned Decision Tree details
- Train R² = 1.00, Test R² = 0.8868
- Tree depth, leaf count

### A.5 Neural Network architecture diagram
- Layer-by-layer description (128 → 64 → 32 → 1, ReLU, Dropout(0.2))
- Optimizer settings, weight decay, batch size, epochs

### A.6 Code repository pointer
- GitHub link if applicable, or note "code available on request"

### A.7 Reproducibility notes
- Random seeds (random_state=42 throughout)
- Python and library versions for pandas, numpy, scikit-learn, PyTorch

---

## Page Budget Summary

| Section | Target |
|---|---|
| Abstract | 0.5 pp |
| Introduction | 0.75 pp |
| Data | 1.25 pp |
| Methods | 1.75 pp |
| Results | 2.25 pp |
| Discussion | 1.25 pp |
| Conclusions and Future Directions | 0.5 pp |
| References | 0.5 pp |
| **Total** | **~8.75 pp** — slight buffer for trimming during writing |

Trim priorities if over 8 pages: (1) compress Methods model paragraphs into one combined paragraph per family (linear, tree, NN); (2) move the linear coefficient bar chart and NN loss curve to the appendix; (3) shorten EDA findings to a single paragraph since the figures speak for themselves.
