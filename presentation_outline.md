# Presentation Outline: Predicting Moscow Apartment Prices

**Course:** ISyE 521 — Machine Learning in Action
**Team:** Mason Reitmeier, Dylan Hicks, Justin Zens
**Format:** Pecha Kucha — 15 content slides, each auto-advancing after 20 seconds (5 minutes total, excluding title slide)

---

## How to use this outline

This outline proposes **more than 15 slide ideas** so the team has flexibility. The 15-slide default path is labeled **[CORE]**. Optional slides are listed at the bottom and can swap in by combining or cutting a CORE slide (each CORE slide notes where combining is natural).

Rough section budgets (ranges, not caps):
- Problem & Data: 2-3 slides
- Data Cleaning: 1 slide
- EDA: 2-4 slides
- Methodology: 1-2 slides
- Modeling: 5-7 slides
- Results & Conclusions: 2-3 slides

Speaker rotation: roughly 5 slides per person. Pick based on who knows each section best (whoever led the OLS coding should talk to Slide 8, whoever trained the NN should take Slide 12, etc.) rather than strict ordering.

Slide design rule: one dominant visual per slide, heading of 6-8 words, no more than 2-3 short bullets. A Pecha Kucha slide the audience has to read is a dead slide.

---

## Title Slide (not counted toward 15)

**On-slide content:**
- Title: "Predicting Moscow Apartment Prices"
- Subtitle: "A Machine Learning Comparison Study"
- Team names: Mason Reitmeier, Dylan Hicks, Justin Zens
- Course: ISyE 521, Spring 2026

**Visual/image recommendation:** Full-bleed photo of the Moscow skyline, preferably one that shows a mix of Soviet-era apartment blocks and modern residential towers. Good sources: Unsplash searches for "Moscow residential" or "Moscow apartments", Pexels, Wikimedia Commons. Avoid the usual Red Square / Saint Basil's shot; the deck is about housing, not tourism.

---

## Slide 1: Motivation and Research Questions — [Problem] — [CORE]

**Purpose:** Hook the audience in the first 20 seconds and set up the three questions the rest of the deck answers.

**On-slide content:**
- Short hook: Moscow housing is expensive and prices vary widely
- Three research questions, numbered:
  1. How accurately can we predict Moscow apartment prices?
  2. Which features drive price (what Moscow buyers prioritize)?
  3. Which features matter most for prediction?

**Talking points (20s):** Moscow is one of Europe's most expensive housing markets, and listing prices can swing by tens of millions of rubles for apartments that look similar on paper. We wanted to see how well machine learning could predict those prices, which apartment features buyers seem to pay the most for, and which features actually matter to a model.

**Visual/image recommendation:** Split layout. Left half: a Moscow street or residential skyline photo (dimmed 30% so text pops). Right half: three numbered question bubbles stacked vertically, each with a small icon (ruble symbol for accuracy, house icon for buyer priorities, bar-chart icon for feature importance).

**Combine/swap notes:** Can merge with Slide 2 if the team wants an extra slide somewhere else; the combined "Problem + Data" slide would show the research questions on the left and the dataset summary on the right.

---

## Slide 2: Data Source — [Data] — [CORE]

**Purpose:** Tell the audience where the data came from and what it contains in under 20 seconds.

**On-slide content:**
- Source: Kaggle, Moscow apartment listings dataset
- Size: 22,676 listings, 12 columns
- Target: Price (rubles)
- 7 numeric features, 4 categorical features (listed briefly)

**Talking points (20s):** We pulled a Kaggle dataset of about 22,700 Moscow apartment listings, each with 12 attributes: the asking price plus seven numeric features like total area, floor number, and minutes to the nearest metro, and four categorical features covering apartment type, region, renovation level, and metro station.

**Visual/image recommendation:** Clean two-column data dictionary table. Column 1: variable name. Column 2: type and a six-word description. Highlight the target row (Price) with a subtle background color. Skip the Kaggle logo screenshot; it wastes visual real estate.

**Combine/swap notes:** Optional "Data Sample Preview" slide (see bottom) can replace this if the team prefers showing five real rows from the CSV instead of a dictionary table.

---

## Slide 3: Data Cleaning Pipeline — [Data] — [CORE]

**Purpose:** Show the audience the cleaning process in a single glance and credibly claim a clean dataset.

**On-slide content:**
- Starting row count: 22,676
- Removed 1,851 duplicates (down to 20,825)
- Removed 2,451 sanity-check failures (Floor > Number of floors: 1,502; Kitchen > Total: 1; Living + Kitchen > Total: 956)
- Final: 18,374 rows
- Note: no missing values in the raw data

**Talking points (20s):** The raw data needed two passes. First we removed about 1,850 exact duplicates. Then we dropped around 2,450 rows that failed logical sanity checks, things like the apartment's floor being higher than the building's total floors, or kitchen plus living area adding up to more than the total. That left us with 18,374 clean listings.

**Visual/image recommendation:** Horizontal funnel with three boxes left to right. Box 1: "22,676 raw". Arrow down labeled "minus 1,851 duplicates". Box 2: "20,825". Arrow down labeled "minus 2,451 invalid". Box 3: "18,374 clean" (colored differently to pop). Below the funnel, three tiny icons with the three sanity-check rules. Keep it uncluttered.

---

## Slide 4: Target Variable and the Log Transform — [EDA] — [CORE]

**Purpose:** Justify why every model in the deck predicts Log_Price rather than Price.

**On-slide content:**
- Raw Price skewness: 7.63 (extreme right skew)
- After log transform: 1.13 (workable)
- All models use Log_Price as target

**Talking points (20s):** Raw apartment prices were extremely skewed, with a skewness value of 7.63 driven by a long tail of very expensive listings. Log-transforming the price brings that skewness down to 1.13, close enough to normal that linear models and neural networks behave well. Every model we trained predicts Log_Price.

**Visual/image recommendation:** Two histograms side by side. Left: raw Price with the long right tail. Right: Log_Price looking roughly bell-shaped. Big horizontal arrow between them labeled "log transform". Add "skew: 7.63 to 1.13" as a caption under the arrow. Both plots are already in CleaningAndEDA.ipynb; screenshot or regenerate with matching styling.

**Combine/swap notes:** Compact enough to merge into Slide 5 by tucking mini histograms into a corner of the correlation heatmap.

---

## Slide 5: What Drives Price (Correlations) — [EDA] — [CORE]

**Purpose:** Give the audience a one-glance ranking of which numeric features correlate with price.

**On-slide content:**
- Top correlations with Log(Price):
  - Area: 0.80
  - Living area: 0.74
  - Number of rooms: 0.67
  - Kitchen area: 0.63
- Weak: Floor 0.11, Number of floors 0.08, Minutes to metro -0.08
- Callout: Area vs Living area correlates at 0.91 (multicollinearity risk)

**Talking points (20s):** Area is the single strongest numeric predictor of log price, at 0.80. Living area, room count, and kitchen area all follow closely behind. Minutes to metro was surprisingly weak, only negative 0.08. One thing to flag: area and living area correlate at 0.91 with each other, which hurts linear model stability unless we regularize.

**Visual/image recommendation:** Correlation heatmap from the notebook, but zoomed to emphasize the Log_Price row. Use a diverging red-blue colormap. Circle or box the Area cell in bright yellow. Crop the heatmap to cut whitespace so it fills the slide.

---

## Slide 6: Categorical Feature Drivers — [EDA] — [CORE]

**Purpose:** Show that non-numeric features (renovation, region, apartment type) also carry substantial price signal.

**On-slide content:**
- Designer renovation commands a clear premium
- Moscow city prices well above Moscow region
- New building prices slightly above Secondary

**Talking points (20s):** Not everything that matters is numeric. Designer-renovation apartments command a huge premium over cosmetic or no-renovation units. Properties inside Moscow city proper sell for noticeably more than those in the surrounding Moscow region. And new buildings carry a modest premium over secondary market listings. These will show up as strong categorical effects in the models.

**Visual/image recommendation:** Two box plots side by side on a shared log-price y-axis. Left box plot: Renovation (4 levels ordered Without, Cosmetic, European, Designer). Right box plot: Region (Moscow city vs Moscow region). Both already exist in CleaningAndEDA.ipynb. Keep colors distinct between the two sub-plots so they don't visually merge.

**Combine/swap notes:** Can swap for the optional "Scatter Plot: Area vs Price" slide if the team prefers a numeric-relationship story over categorical.

---

## Slide 7: Methodology and Preprocessing — [Methods] — [CORE]

**Purpose:** Explain the exact preprocessing every model saw, so the results are comparable.

**On-slide content:**
- 80/20 train/test split, random_state=42, 14,699 train / 3,675 test
- 12 features total: 7 numeric + 5 one-hot dummies
- ColumnTransformer: StandardScaler on numeric, one-hot with drop_first=True on categorical
- Baselines (dropped categories): Cosmetic renovation, Moscow city, New building
- Metro station dropped (317 unique values too sparse)

**Talking points (20s):** We used an 80/20 train-test split with a fixed random seed so every model sees the same rows. Numeric features got standardized, categoricals were one-hot encoded with one level dropped as the baseline. Metro station had 317 unique values so we dropped it in favor of the region variable. Same pipeline, same 12 features, every model.

**Visual/image recommendation:** Left-to-right pipeline diagram. Three boxes connected by arrows. Box 1: "Raw features (11 inputs)". Box 2: "ColumnTransformer" (split internally into two sub-boxes: "StandardScaler (7 numeric)" and "OneHot drop_first (5 dummies)"). Box 3: "Model (12 features)". Use a clean sans-serif font.

---

## Slide 8: Linear Baselines — OLS, Lasso, Ridge — [Methods/Results] — [CORE]

**Purpose:** Establish the baseline performance. Three linear models all converge on the same answer.

**On-slide content:**
- All three: Test R-squared around 0.869, RMSE log 0.389, MAPE 28.4%
- Lasso (alpha = 0.000852) zeroed Floor only; Ridge (alpha = 6.58) kept all 12
- Top standardized coefficients (all three agree within 0.02):
  - Designer renovation +0.65
  - Moscow region -0.43
  - Area +0.40
  - No renovation +0.40
  - Secondary apartment +0.33

**Talking points (20s):** Our three linear baselines, OLS, Lasso, and Ridge, all landed at a test R-squared around 0.87 and a mean absolute percentage error near 28 percent. Lasso only zeroed out the Floor feature, and the coefficient rankings agreed across all three methods. Designer renovation and Moscow location were the biggest price drivers.

**Visual/image recommendation:** Horizontal bar chart of Ridge's standardized coefficients, sorted by absolute magnitude. Positive bars in green, negative in red. Small metrics box in the top-right corner: "Test R-squared = 0.869 | MAPE = 28.4%". Pull the chart directly from modeling.ipynb.

**Combine/swap notes:** Can expand to 2-3 slides (one per linear model) if the team wants depth here, at the cost of cutting one tree-based model slide later.

---

## Slide 9: Decision Trees — Initial vs Tuned — [Methods/Results] — [CORE]

**Purpose:** Show the overfit-to-controlled transition and the first meaningful jump in test performance.

**On-slide content:**
- Initial tree (fully grown): Train R-squared = 1.00, Test R-squared = 0.887 (classic overfit)
- Tuned tree (GridSearchCV, 5-fold, 45 combos): max_depth=10, min_samples_leaf=10
- Tuned: Test R-squared = 0.9243, MAPE = 19.0%

**Talking points (20s):** Our first decision tree memorized the training data, hitting a perfect R-squared of 1.0 on train but only 0.89 on test. After grid-searching over 45 combinations of depth and leaf size, the tuned tree settled at a max depth of 10 with at least 10 samples per leaf. Test R-squared jumped to 0.92 and MAPE dropped to 19 percent.

**Visual/image recommendation:** Two side-by-side mini bar charts. Left pair: "Initial" with a tall train bar at 1.00 and a shorter test bar at 0.89 (dramatic gap). Right pair: "Tuned" with a train bar at 0.95 and a test bar at 0.92 (tight gap). Label each bar with its value. The visual tells the overfitting story in under two seconds.

---

## Slide 10: Random Forest — The Best Single Model — [Methods/Results] — [CORE]

**Purpose:** Introduce the best-performing single model and show what features it relied on.

**On-slide content:**
- RandomizedSearchCV: 20 iterations, 3-fold CV, 20 of 324 possible combinations sampled
- Best parameters: n_estimators=300, max_depth=20, max_features='sqrt', min_samples_leaf=1
- Test R-squared = 0.9464, RMSE log = 0.2491, MAE = 11.6M rubles, MAPE = 15.07%
- Permutation importance: Area dominates (0.36), then Apartment type (0.09), Region (0.07)

**Talking points (20s):** Random Forest was our best single model. A randomized search over 20 hyperparameter combinations settled on 300 trees with a max depth of 20. It hit a test R-squared of 0.946 and a MAPE around 15 percent. Permutation importance confirmed Area is by far the most important feature, followed by apartment type and region.

**Visual/image recommendation:** Permutation importance horizontal bar chart from modeling.ipynb, sorted descending. Use a single color (green or dark blue). Corner callout box with the metrics: "R-squared 0.946 | MAPE 15.1% | MAE 11.6M rubles". The importance chart answers research questions 2 and 3, so it earns the slide real estate over an architecture diagram.

---

## Slide 11: AdaBoost — [Methods/Results] — [CORE]

**Purpose:** Present a boosting approach honestly; it was tuned fully but did not beat random forest.

**On-slide content:**
- GridSearchCV (5-fold, 45 combos), base learner = DecisionTreeRegressor(max_depth=3)
- Best: n_estimators=300 (at grid ceiling), learning_rate=0.05, loss='exponential'
- Test R-squared = 0.8864, MAPE = 30.32% (worst MAPE of any model)
- Exponential loss chases high-value outliers, which hurts MAPE

**Talking points (20s):** AdaBoost used depth-3 trees as weak learners and grew to 300 of them, with exponential loss that focused on hard, high-priced examples. R-squared came out at 0.886, middle of the pack, but MAPE was actually the worst in the lineup at 30 percent. The exponential loss over-fit to the tail, which hurts percentage errors even when R-squared looks fine.

**Visual/image recommendation:** Two-bar comparison showing Test R-squared (0.886, middling, colored neutral) next to MAPE (30.3%, colored red to flag as worst). This "R-squared OK but MAPE worst" story lands in five seconds. Alternative: AdaBoost's MDI feature importance bar chart from the notebook if the team prefers a consistent-feature-importance visual theme.

**Combine/swap notes:** Can merge with Slide 12 into a single "Other methods we tried" slide if the team wants to free up a slide for the optional Limitations slide or a deeper Feature Importance slide.

---

## Slide 12: Neural Network (PyTorch MLP) — [Methods/Results] — [CORE]

**Purpose:** Show a deep-learning attempt and contextualize where it lands relative to trees.

**On-slide content:**
- Architecture: 12 input to 128 to 64 to 32 to 1, ReLU activation, Dropout 0.2
- Optimizer: Adam, weight_decay = 1e-3
- Training: 300 epochs, batch size 64, trained on GPU (CUDA)
- Two-phase training: 75/25 val split for architecture tuning, then fresh retrain on full train set
- Test R-squared = 0.9065, RMSE log = 0.329, MAPE = 18.47%
- Sits between linear and tuned-tree models

**Talking points (20s):** We also built a four-layer MLP in PyTorch: 128, 64, 32 hidden units with ReLU, dropout for regularization, Adam optimizer, GPU-trained over 300 epochs. We used a held-out validation split to sanity-check the architecture, then retrained fresh on the full training set. It hit a test R-squared of 0.91, which beats linear models but falls short of the tuned decision tree.

**Visual/image recommendation:** Left half: layer-stack architecture diagram. Stacked rectangles labeled "Input 12", "Linear 128 + ReLU + Dropout", "Linear 64 + ReLU + Dropout", "Linear 32 + ReLU", "Output 1". Right half: training and validation loss curve from the notebook, showing loss dropping and then flattening. Simple and clean.

---

## Slide 13: Stacking Ensemble — [Methods/Results] — [CORE]

**Purpose:** Show the final ensemble and deliver the memorable finding that Random Forest was carrying most of the weight.

**On-slide content:**
- 5 base learners: Ridge, Tuned Decision Tree, Tuned Random Forest, AdaBoost, PyTorch MLP
- Meta-learner: RidgeCV, 5-fold cross-validation
- Test R-squared = 0.9450 (ties Random Forest), MAPE = 14.93% (best of any model)
- Meta-learner weights:
  - Random Forest: +0.955
  - Neural Network: +0.209
  - Decision Tree: -0.013
  - Ridge: -0.017
  - AdaBoost: -0.134

**Talking points (20s):** Finally we stacked all five models using a Ridge meta-learner. The result essentially tied Random Forest on R-squared but edged ahead slightly on MAPE. What's revealing is the meta-learner's weights: it put 95 percent of its trust in Random Forest and 20 percent in the neural net, while assigning negative weights to the others, meaning they were actively subtracted.

**Visual/image recommendation:** Funnel diagram. Left side: five labeled boxes for the base models stacked vertically. Middle: five arrows converging to a "RidgeCV meta-learner" box on the right. Arrow thickness proportional to the weight, with the numeric weight labeled on each arrow (+0.955 for RF is thickest, +0.209 for NN medium, the three negatives shown as thin red arrows). This is the most memorable visual in the modeling section.

---

## Slide 14: Model Comparison — [Results] — [CORE]

**Purpose:** Show every model on a single screen so the audience can see the full ranking.

**On-slide content:**
- Full comparison table, sorted by Test R-squared descending:

| Model | Test R² | RMSE log | MAPE |
|---|---|---|---|
| Random Forest | 0.9464 | 0.2491 | 15.07% |
| Stacking | 0.9450 | 0.2523 | 14.93% |
| Tuned DT | 0.9243 | 0.2961 | 18.96% |
| Neural Net | 0.9065 | 0.3290 | 18.47% |
| Initial DT | 0.8868 | 0.3619 | 22.60% |
| AdaBoost | 0.8864 | 0.3626 | 30.32% |
| Ridge | 0.8694 | 0.3888 | 28.41% |
| OLS | 0.8694 | 0.3888 | 28.45% |
| Lasso | 0.8694 | 0.3889 | 28.41% |

- Callout: Random Forest and Stacking tie; both sit roughly 8 points R-squared above linear baselines
- Caveat: one holdout split, not nested cross-validation

**Talking points (20s):** Here's the full leaderboard. Random Forest and Stacking essentially tie at the top around 0.946 R-squared. Linear models cluster together at 0.87, eight points lower. The tuned decision tree and neural net sit in the middle. Worth noting these are all on one 80/20 split, so the ordering reflects that one split, not a nested cross-validation guarantee.

**Visual/image recommendation:** Horizontal grouped bar chart. Each model gets two bars side by side: Test R-squared (blue, primary axis) and MAPE (orange, secondary axis). Sorted by R-squared descending. Numeric labels on each bar. Alternative: color-coded table with the top row (Random Forest) highlighted. Bar chart is more Pecha Kucha friendly; table is safer for precise numbers. Lean toward the bar chart.

**Combine/swap notes:** If the team finds this slide too crowded, split it into a chart-focused comparison slide plus the optional Feature Importance Deep Dive slide.

---

## Slide 15: Conclusions and Future Work — [Conclusions] — [CORE]

**Purpose:** Close the deck by answering the three research questions from Slide 1 and briefly flagging limitations and next steps.

**On-slide content:**
- Answers to the three research questions:
  1. Accuracy: Test R-squared of 0.946, about 95% of price variation explained; MAPE of 15% translates to roughly 4 million rubles off on a typical listing
  2. Buyer priorities (linear coefficients): Renovation quality (especially Designer), Moscow city vs region, total Area, apartment type (New vs Secondary)
  3. Feature importance (Random Forest permutation): Area dominates (0.36), then Apartment type (0.09) and Region (0.07)
- Limitations: single train/test split, listing vs transaction prices, dataset is a 2021 snapshot
- Future: geographic features beyond binary Region, gradient boosting (XGBoost, LightGBM), time-series analysis if dated data becomes available

**Talking points (20s):** To wrap up: we can predict Moscow apartment prices with about 95 percent variance explained. Buyers pay the biggest premiums for designer renovations and Moscow-city locations. And for prediction specifically, area dominates, with apartment type and region close behind. Our main limitations are one train-test split and the use of listing prices. Future work would add finer geography, gradient boosting, and time-series features.

**Visual/image recommendation:** Three-column summary infographic. Column 1 (Accuracy): big "0.946" with "R-squared" below. Column 2 (Buyer Priorities): three stacked icons for Renovation, Location, Area. Column 3 (Feature Importance): horizontal mini-bar chart showing Area > Apartment type > Region. Bottom strip in smaller text: "Limitations: single split, listing bias, 2021 snapshot | Future: geo features, XGBoost, time series". Closes the visual loop from Slide 1.

---

## Optional / Swap-in slides

Drop any of these into the deck by combining or cutting a CORE slide above. Each is written the same way so the team can swap in wholesale.

### [OPTIONAL] Why This Matters (Stakes)

**Purpose:** Expand the motivation with a concrete buyer angle.

**On-slide content:**
- Moscow apartments commonly range from 10M to 100M+ rubles
- A 15% prediction error equals roughly 3-4M rubles on a typical 20M ruble listing
- Real-world stakes: mispriced listings hurt both buyers and sellers

**Talking points (20s):** Before diving into methods, why does this matter? Moscow apartments routinely sell for 20, 50, even 100 million rubles. A 15 percent prediction error, which is where our best model lands, means being off by three to four million rubles on a typical listing. For buyers and sellers, that's the difference between a fair deal and a very bad one.

**Visual/image recommendation:** Large centered text, "±₽3M" overlaid on a dimmed Moscow street photo, with small caption below reading "typical prediction error on a ₽20M listing".

**When to use:** Slide 1 feels abstract or the team wants a stronger hook. Combine Slides 1 and 2 to free space.

---

### [OPTIONAL] Data Sample Preview

**Purpose:** Make the raw data concrete by showing actual rows.

**On-slide content:**
- Screenshot or rendered table of 5 rows from data.csv
- Highlight one column per bullet point of context

**Talking points (20s):** Here's what the raw data actually looks like. Each row is one listing: price on the left, then apartment type, region, metro station, area and so on. Some apartments have 15 minutes to the nearest metro, some have 45. Area ranges from tiny studios to huge family apartments. This is the messy ground truth before any cleaning.

**Visual/image recommendation:** A well-formatted table of 5 diverse rows (mix of price ranges, renovation types, regions). Render in the presentation tool, not screenshotted from Jupyter. Use alternating row shading.

**When to use:** Replace Slide 2 if the team prefers concrete over abstract, or insert right after Slide 2.

---

### [OPTIONAL] Scatter Plot: Area vs Price

**Purpose:** Show the strongest numeric relationship visually, plus heteroscedasticity.

**On-slide content:**
- Scatter plot of Area (x) vs Price (y, log scale)
- Trend line overlay
- Callout: correlation 0.80, notice the variance grows with size

**Talking points (20s):** Area is our strongest predictor, correlating with log price at 0.80. Here's what that looks like. Clear positive relationship, but notice that the spread of prices grows wider for bigger apartments. That's heteroscedasticity, and it's part of why we log-transformed the target. The log transform compresses that growing spread.

**Visual/image recommendation:** Single large scatter plot. X-axis: Area. Y-axis: Log_Price. Semi-transparent dots (alpha 0.3) because there are 18,000 of them. Red trend line overlaid. Already in CleaningAndEDA.ipynb.

**When to use:** Swap for Slide 6 if the team finds the categorical box plots less compelling than a clean numeric scatter.

---

### [OPTIONAL] Feature Importance Deep Dive

**Purpose:** Dedicated slide that shows both linear coefficients and RF permutation importance agreeing on the top features.

**On-slide content:**
- Two bar charts side by side: linear-model standardized coefficients (left) and Random Forest permutation importance (right)
- Callout: both methods rank Area, Renovation, Region, Apartment type in the top 5
- Divergence note: Living area ranks high by MDI but drops under permutation

**Talking points (20s):** Different methods, same answer. The linear models say renovation and location matter most by coefficient size. Random Forest's permutation importance says area and apartment type matter most for prediction accuracy. The two methods disagree slightly on ordering but agree on the top four features: area, renovation, region, and apartment type.

**Visual/image recommendation:** Two horizontal bar charts side by side on the same slide, shared y-axis labels. Left: linear model coefficients. Right: permutation importance. Color-code the matching features the same way across both panels so the eye tracks them.

**When to use:** Split Slide 14 into "Comparison (numbers)" and "Feature Importance" if Slide 14 feels too dense.

---

### [OPTIONAL] Limitations and Caveats

**Purpose:** Honest accounting of what the results do not prove.

**On-slide content:**
- Single 80/20 train/test split (not nested CV)
- Listing prices, not transaction prices
- Dataset is a 2021 snapshot
- Metro station dropped due to 317 unique values
- Moscow-specific findings may not generalize

**Talking points (20s):** Some honest caveats. First, our rankings come from a single 80/20 split, not nested cross-validation. Second, listing prices are not the same as sale prices. Third, the data is a 2021 snapshot, so current results may differ. And we dropped the metro station feature entirely because it had 317 unique values, which is more information than our models could use well.

**Visual/image recommendation:** Four warning-icon bullets on a clean neutral background. Each bullet gets one concrete limitation. Alternative: a four-quadrant grid with one limitation per quadrant.

**When to use:** Frees Slide 15 to focus purely on the positive conclusion message.

---

### [OPTIONAL] Team Contributions

**Purpose:** Short slide listing who led what part of the project.

**On-slide content:**
- Mason Reitmeier: led X
- Dylan Hicks: led Y
- Justin Zens: led Z

**Talking points (20s):** Quick credits. Mason owned the initial data cleaning and EDA. Dylan handled the linear models, decision trees, and overall modeling pipeline. Justin built the neural network and the stacking ensemble. All three of us contributed to analysis, interpretation, and the writeup.

**Visual/image recommendation:** Three headshots or monogram circles in a row, each with a name and a two-word contribution label underneath.

**When to use:** Some instructors appreciate it; others consider it filler. Easy to insert right before conclusions or on the title slide.

---

## How to pick your final 15

1. Start from the 15 CORE slides above.
2. If a section feels thin or crowded, check the Combine/swap notes on that slide.
3. Pull in an optional slide only by cutting or combining a core slide, never by going to 16.
4. Final check before rehearsal: count the content slides, exclude the title, confirm exactly 15.
5. Rehearse with a timer. 20 seconds is short. Poorly rehearsed Pecha Kucha presentations are obvious (the rubric explicitly says this).
