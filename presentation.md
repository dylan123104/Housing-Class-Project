# Presentation Reference: Moscow Apartment Price Prediction

Companion document to `presentation_outline.md`. Organized by section rather than by slide. Use this as the source of truth to copy numbers, findings, and context from while building the deck.

---

## 1. Project Overview

**Title:** Predicting Moscow Apartment Prices: A Machine Learning Comparison Study
**Team:** Mason Reitmeier, Dylan Hicks, Justin Zens
**Course:** ISyE 521, Machine Learning in Action, Spring 2026
**Format:** Pecha Kucha, 15 content slides, 20 seconds per slide, 5 minutes total, all three team members speak.

### Research questions

1. How accurately can apartment prices in Moscow be predicted?
2. Which apartment features do Moscow residents prioritize, meaning which features most influence price?
3. Which features are most important for price prediction?

---

## 2. Data Source

- **Source:** Kaggle dataset of Moscow apartment listings.
- **Raw shape:** 22,676 rows by 12 columns.
- **Target variable:** `Price` (Russian rubles), later transformed to `Log_Price`.

### Variables

| Variable | Type | Notes |
|---|---|---|
| Price | numeric (target) | Russian rubles, heavily right-skewed in raw form |
| Apartment type | categorical | 2 values: New building, Secondary |
| Metro station | categorical | 317 unique values, dropped from modeling |
| Minutes to metro | numeric | Travel time to nearest metro station |
| Region | categorical | 2 values: Moscow, Moscow region |
| Number of rooms | numeric (ordinal) | 12 distinct values, treated as continuous |
| Area | numeric | Total apartment area |
| Living area | numeric | Living space subset of total area |
| Kitchen area | numeric | Kitchen space subset of total area |
| Floor | numeric | Floor the apartment is on |
| Number of floors | numeric | Total floors in the building |
| Renovation | categorical | 4 levels: Without, Cosmetic, European-style, Designer |

---

## 3. Data Cleaning

Implemented in `CleaningAndEDA.ipynb`. Final cleaned dataset saved to `data_cleaned.csv`.

### Pipeline and exact numbers

| Step | Rows removed | Running total |
|---|---|---|
| Raw data loaded | 0 | 22,676 |
| Whitespace strip and type coercion | 0 | 22,676 |
| Duplicate removal | 1,851 | 20,825 |
| Sanity check: Floor > Number of floors | 1,502 | (removed as part of combined sanity pass) |
| Sanity check: Kitchen area > Total area | 1 | (same) |
| Sanity check: Living + Kitchen > Total | 956 | (same) |
| Combined sanity-check removal | 2,451 | **18,374** |

**Final cleaned dataset:** 18,374 rows, 12 columns. Index reset with `reset_index(drop=True)`.

### Cleaning notes

- No missing values were found in the raw data.
- The three sanity checks overlap, so the total removed in the combined pass is 2,451, not the naive sum of 1,502 + 1 + 956.
- `Log_Price` was added as a new column for modeling.
- The cleaned CSV was exported for teammate sharing.

---

## 4. EDA Findings

### Target variable

- Raw `Price` skewness: **7.63** (extreme right skew).
- After log transform, `Log_Price` skewness: **1.13** (moderate, but workable).
- Every model predicts `Log_Price`.

### Correlations with `Log(Price)`

| Feature | Correlation |
|---|---|
| Area | **0.80** |
| Living area | **0.74** |
| Number of rooms | **0.67** |
| Kitchen area | **0.63** |
| Floor | 0.11 |
| Number of floors | 0.08 |
| Minutes to metro | -0.08 |

### Multicollinearity

- **Area and Living area correlate at 0.91.** This is the big one. Without regularization, linear models can get unstable; Lasso and Ridge handle it fine (see results below).

### Categorical findings

- **Renovation:** Designer apartments command a clear premium over Cosmetic or Without. European-style sits between Cosmetic and Designer.
- **Region:** Moscow city prices well above Moscow region on every metric.
- **Apartment type:** New building slightly above Secondary, but the gap is smaller than Renovation or Region.
- **Number of rooms:** Behaves as ordinal. Higher room counts track higher prices, with steps visible in the distribution.

### Scatter plot observations

- Strong positive linear relationship between Area and Price.
- Heteroscedasticity is visible: variance grows with apartment size. Log transform on the target helps.
- Minutes to metro shows only a weak negative relationship.

### Metro station decision

- 317 unique metro stations was too many distinct values to one-hot usefully.
- Replaced with the two-level `Region` feature and the numeric `Minutes to metro`.

---

## 5. Modeling Setup

### Target and features

- **Target:** `Log_Price` (natural log of Price).
- **Final feature count:** 12 features — 7 numeric + 5 one-hot dummies.
- **Numeric features (7):** Minutes to metro, Number of rooms, Area, Living area, Kitchen area, Floor, Number of floors.
- **Categorical features (one-hot, drop_first=True):** Apartment type, Region, Renovation.
- **Reference categories (dropped):** Cosmetic renovation, Moscow city, New building.

### Train/test split

- 80/20 split, `random_state=42`.
- Train: 14,699 rows. Test: 3,675 rows.
- Same split used across every model for comparability.

### Preprocessing

- `ColumnTransformer` with `StandardScaler` on the 7 numeric features and pass-through one-hot dummies.
- The exact same preprocessor is used across OLS, Lasso, and Ridge so coefficient magnitudes are directly comparable on the standardized scale.
- For tree-based models the scaling doesn't matter for fits but we keep the same pipeline for pipeline consistency.

### Metrics reported

- Train R-squared and Test R-squared
- RMSE on the log scale (primary accuracy metric)
- MAE in rubles (context metric, after exponentiating predictions)
- MAPE in percent (context metric)
- RMSE in rubles was dropped because exponentiating log predictions amplifies tail errors, producing misleading numbers.

---

## 6. Model Results

### OLS Linear Regression (baseline)

- Test R-squared: **0.8694**
- RMSE log: **0.3888**
- MAE: 26,525,122 rubles
- MAPE: 28.45%
- All 12 features retained. Fit through the same ColumnTransformer as Lasso and Ridge so coefficients are on a comparable standardized scale.

### Lasso Regression

- Best alpha (from 10-fold LassoCV): 0.000852
- Features zeroed: `Floor` only (11 of 12 kept)
- Test R-squared: **0.8694**
- RMSE log: **0.3889**
- MAE: 26,833,143 rubles
- MAPE: 28.41%
- Coefficients nearly identical to OLS. No multicollinearity blow-up, despite the 0.91 Area/Living area correlation.

### Ridge Regression

- Best alpha (from 10-fold RidgeCV): 6.5793
- All 12 features kept
- Test R-squared: **0.8694**
- RMSE log: **0.3888**
- MAE: 26,468,105 rubles
- MAPE: 28.41%
- All three linear models agree on every coefficient within about 0.02. Coefficient estimates are stable.

### Decision Tree (Initial, untuned)

- Fully grown tree, no hyperparameter constraints.
- Train R-squared: 1.00 (memorized training data)
- Test R-squared: **0.8868**
- RMSE log: 0.3619
- MAE: 15.1M rubles
- MAPE: 22.60%
- Still beats linear models on test despite severe overfitting, which motivates tuning.

### Decision Tree (Tuned)

- Search: GridSearchCV, 5-fold, 45 hyperparameter combinations.
- Best parameters: `max_depth=10`, `min_samples_leaf=10`, `min_samples_split=2`.
- Train R-squared: 0.9473
- Test R-squared: **0.9243**
- RMSE log: **0.2961**
- MAE: 13.1M rubles
- MAPE: 18.96%
- Label is "Tuned" rather than "Optimized" because the grid exhausts 45 combinations, which is a constrained search.

### Random Forest (Tuned)

- Search: RandomizedSearchCV, n_iter=20, 3-fold CV, samples 20 of 324 possible combinations.
- Best parameters: `n_estimators=300`, `max_depth=20`, `max_features='sqrt'`, `min_samples_leaf=1`.
- Train R-squared: 0.9917
- Test R-squared: **0.9464** (best single model)
- RMSE log: **0.2491** (best single model)
- MAE: **11.6M rubles** (best single model)
- MAPE: **15.07%** (best single model)
- Feature importance: see Feature Importance section below. Area is #1 by both MDI and permutation importance.

### AdaBoost (Tuned)

- Base learner: `DecisionTreeRegressor(max_depth=3)`.
- Search: GridSearchCV, 5-fold, 45 combinations.
- Best parameters: `n_estimators=300` (at grid ceiling), `learning_rate=0.05`, `loss='exponential'`.
- Train R-squared: 0.8861
- Test R-squared: **0.8864**
- RMSE log: 0.3626
- MAE: 15.1M rubles
- MAPE: **30.32%** (worst MAPE of any model)
- n_estimators hit the ceiling of the grid, so results are best within a constrained search. Exponential loss focuses on high-value outliers, which hurts MAPE even when R-squared looks reasonable.
- No permutation importance was computed for this model; only MDI.

### Neural Network (PyTorch MLP)

- Architecture: 12 input to 128 to 64 to 32 to 1.
- Activation: ReLU with Dropout(0.2) between hidden layers.
- Optimizer: Adam, `weight_decay=1e-3`.
- Training: 300 epochs, batch size 64, GPU (CUDA).
- Two-phase training: 75/25 validation split inside X_train was used to monitor loss curves and confirm the architecture, then a fresh model was retrained on the full X_train (14,699 rows) for the reported metrics.
- `y_train.values.copy()` was required to avoid a non-writable tensor warning from PyTorch.
- Train R-squared: 0.9159
- Test R-squared: **0.9065**
- RMSE log: 0.3290
- MAE: 15.1M rubles
- MAPE: 18.47%
- Falls just below the tuned decision tree on primary metrics, ties it on MAPE.

### Stacking Ensemble

- Base learners (5): Ridge, Tuned Decision Tree, Tuned Random Forest, AdaBoost, PyTorch MLP (via a `TorchMLPRegressor` sklearn wrapper that inherits from `BaseEstimator` and `RegressorMixin`).
- Meta-learner: RidgeCV with alphas from `np.logspace(-3, 3, 50)`, 5-fold.
- CV: 5-fold `KFold` with `shuffle=True`, `random_state=42`.
- `n_jobs=1` on StackingRegressor was required to avoid PyTorch/CUDA multiprocessing conflicts. Random Forest's internal `n_jobs=-1` still applies within each fold.
- Train R-squared: 0.9897
- Test R-squared: **0.9450**
- RMSE log: **0.2523**
- MAE: 11.85M rubles
- MAPE: **14.93%** (best MAPE of any model)

### Stacking meta-learner weights

| Base model | Weight |
|---|---|
| Random Forest | **+0.955** |
| Neural Network | +0.209 |
| Decision Tree | -0.013 |
| Ridge | -0.017 |
| AdaBoost | -0.134 |
| Intercept | +0.025 |

The meta-learner placed 95% of its trust in Random Forest. The negative weights on Decision Tree, Ridge, and AdaBoost mean those predictions were actively subtracted during blending, which confirms Random Forest had already absorbed most of the learnable signal.

---

## 7. Feature Importance

### Linear-model standardized coefficients

All three linear models (OLS, Lasso, Ridge) agree on every coefficient within about 0.02. The top standardized coefficients, which answer research question 2 (what do buyers prioritize), are:

| Feature | Coefficient |
|---|---|
| Designer renovation (vs Cosmetic baseline) | +0.65 |
| Moscow region (vs Moscow city baseline) | -0.43 |
| Area | +0.40 |
| No renovation (vs Cosmetic baseline) | +0.40 |
| Secondary apartment (vs New building baseline) | +0.33 |

Reference categories: the positive coefficient on "No renovation" is relative to Cosmetic, meaning apartments listed with no renovation actually sold for more than apartments listed with cosmetic renovation. This is a data quirk worth flagging but does not affect the top-driver conclusions.

### Random Forest permutation importance (authoritative)

This answers research question 3 (what matters for prediction). Permutation importance is the authoritative ranking because MDI overstates correlated and high-cardinality features.

| Feature | Permutation importance |
|---|---|
| Area | 0.3614 |
| Apartment type (Secondary dummy) | 0.0900 |
| Region | 0.0715 |
| Living area | 0.0495 |
| Kitchen area | 0.0233 |

### Random Forest MDI importance (for comparison)

| Feature | MDI importance |
|---|---|
| Area | 0.3239 |
| Living area | 0.2019 |
| Kitchen area | 0.0979 |
| Apartment type (Secondary) | 0.0923 |
| Region | 0.0673 |

### MDI vs permutation divergence

- **Living area** drops from 0.20 (MDI) to 0.05 (permutation). The 0.91 correlation with Area inflates MDI but permutation correctly discounts it.
- **Number of rooms** also drops substantially: 3rd by MDI but 8th by permutation.
- The permutation ranking puts Apartment type and Region ahead of Living area, which matches the linear-model coefficient rankings.

---

## 8. Model Comparison

Sorted by Test R-squared, descending:

| Model | Train R² | Test R² | RMSE log | MAE (₽M) | MAPE |
|---|---|---|---|---|---|
| Random Forest | 0.9917 | **0.9464** | **0.2491** | **11.6** | 15.07% |
| Stacking Ensemble | 0.9897 | 0.9450 | 0.2523 | 11.85 | **14.93%** |
| Decision Tree (Tuned) | 0.9473 | 0.9243 | 0.2961 | 13.1 | 18.96% |
| Neural Network | 0.9159 | 0.9065 | 0.3290 | 15.1 | 18.47% |
| Decision Tree (Initial) | 1.0000 | 0.8868 | 0.3619 | 15.1 | 22.60% |
| AdaBoost | 0.8861 | 0.8864 | 0.3626 | 15.1 | 30.32% |
| Ridge | 0.8599 | 0.8694 | 0.3888 | 26.47 | 28.41% |
| OLS | 0.8599 | 0.8694 | 0.3888 | 26.53 | 28.45% |
| Lasso | 0.8599 | 0.8694 | 0.3889 | 26.83 | 28.41% |

### Key observations

- Random Forest and Stacking are effectively tied. RF wins on R-squared and MAE by a hair; Stacking wins on MAPE.
- Both winners sit about 8 percentage points of R-squared above the linear baselines.
- All three linear models are essentially identical.
- The initial decision tree's perfect 1.0 train R-squared is a textbook overfit; tuning brings it down to 0.95 train with 0.92 test.
- AdaBoost's R-squared is middle of the pack but its MAPE is the worst due to exponential loss chasing high-value outliers.
- Test set reuse caveat: all models evaluated on the same 80/20 holdout, so the ranking reflects one split and not a nested cross-validation guarantee.

---

## 9. Takeaways

### Answers to the research questions

1. **How accurately can we predict Moscow apartment prices?** Test R-squared of 0.946 from Random Forest or the Stacking ensemble. About 95% of log-price variation is explained. MAPE of 15% translates to roughly 4 million rubles off on a typical listing.
2. **What do buyers prioritize?** Renovation quality (especially Designer), Moscow city location, apartment size, and apartment type (New vs Secondary). This comes from the standardized linear coefficients.
3. **What matters most for prediction?** Area dominates at 0.36 permutation importance. Apartment type (0.09) and Region (0.07) come next. Living area drops substantially once its correlation with Area is controlled.

### Limitations

- Results come from a single 80/20 train/test split, not nested cross-validation.
- The dataset contains listing prices, not actual transaction prices, so there is a listing bias.
- Dataset is a 2021 snapshot, so current-market results may differ.
- Metro station with 317 unique values was dropped; finer-grained geography might help.

### Future directions

- Gradient boosting models (XGBoost, LightGBM, CatBoost).
- Geographic features beyond the binary Region flag (distance to center, neighborhood fixed effects).
- Time-series features if date-stamped data becomes available.
- Target transformation alternatives (Box-Cox, quantile).

---

## 10. Deliverable status (per CLAUDE.md)

- Proposal: submitted 2026-03-05.
- Preliminary Report: pending.
- Abstract: pending.
- Presentation: in preparation (this outline).
- Final Report: pending.
- Peer feedback: pending, depends on other teams' presentations.
