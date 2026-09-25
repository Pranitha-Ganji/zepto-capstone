# Module 2 — Analytics Pipeline (`/analytics`)
**Zepto Customer Analytics & Predictive Modeling Engine**

---

## 1. Overview & Cohesive Architecture
The Zepto Analytics Pipeline represents a complete end-to-end data science lifecycle. Built around customer-style demographic and behavioral data, it takes the classic Titanic dataset from raw ingestion, through defensible data cleaning, exploratory visual storytelling, and statistical profiling, directly into an industrial-grade predictive modeling and hyperparameter tuning pipeline.

The dataset is ingested **exactly once** via `sns.load_dataset('titanic')` and immediately persisted as a committed offline fallback `titanic.csv` inside `/analytics`. Every subsequent step—from univariate analysis in `01_eda.ipynb` to modeling, SMOTE balancing, hyperparameter search, and linear regression in `02_modeling.ipynb`—builds consistently upon this shared, cleaned data.

---

## 2. Part A — Profiling, Cleaning & The Data Story

### Task 1: Dataset Profiling & Committed Fallback
- **Ingestion & Fallback:** Loaded once and committed as `titanic.csv` inside `/analytics` (`df.to_csv("titanic.csv", index=False)`).
- **Dimensions:** 891 rows, 15 columns.
- **Missing Value Percentages (Pre-Cleaning):**
  - `deck`: 688 missing (**77.22%**)
  - `age`: 177 missing (**19.87%**)
  - `embarked`: 2 missing (**0.22%**)
  - `embark_town`: 2 missing (**0.22%**)

### Task 2: Missing-Value Handling Strategy (Threshold Rule)
We applied the strict percentage-based threshold rule:
1. **`deck` (77.22% missing — exceeds 30% threshold):** **Dropped column.** Imputing over three-quarters of missing values would introduce severe synthetic bias and model hallucination. Dropping the column preserves model integrity.
2. **`age` (19.87% missing — between 5% and 30% threshold):** **Imputed with median (28.0 years).** The median is selected over the mean because the age distribution is positively skewed with extreme outliers; median imputation preserves sample size without shifting the central tendency.
3. **`embarked` & `embark_town` (0.22% missing — under 5% threshold):** **Dropped affected rows (2 rows).** Dropping only two rows retains 99.78% of the dataset without distorting demographic distributions.

Cleaned dataset size: **889 rows, 14 columns, 0 remaining nulls.**

### Task 3: Univariate Analysis, IQR Outliers & Skewness
- **Age Distribution:** Spans 0.42 to 80.00 years.
  - $Q_1 = 22.00$, $Q_3 = 35.00$, $IQR = 13.00$.
  - Bounds: $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR] = [2.50, 54.50]$.
  - **Outlier count:** **65 outliers (7.31%)**, representing infant passengers below 2.5 years and senior passengers above 54.5 years.
- **Fare Distribution:** Spans £0.00 to £512.33.
  - $Q_1 = 7.90$, $Q_3 = 31.00$, $IQR = 23.10$.
  - Bounds: $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR] = [-26.76, 65.66]$.
  - **Outlier count:** **114 outliers (12.82%)**, representing first-class luxury suites and multi-passenger bookings.
- **Fare Skewness Determination:**
  - $\text{Mean} = 32.10$
  - $\text{Median} = 14.45$
  - $\text{Mode} = 8.05$
  - $\text{Skewness} = 4.80$
  - **Written Conclusion:** The ordering **$\text{Mean (32.10)} > \text{Median (14.45)} > \text{Mode (8.05)}$** firmly establishes that the fare distribution is heavily **RIGHT-SKEWED (positively skewed)**, characterized by a long right tail driven by premium ticket prices.

### Task 4: Bivariate Analysis & 6x6 Correlation Matrix
- **Survival Rates (Computed via Boolean Masking):**
  - **By Sex:** Female = **74.04%** (231/312) | Male = **18.89%** (109/577)
  - **By Pclass:** Class 1 = **62.62%** (134/214) | Class 2 = **47.28%** (87/184) | Class 3 = **24.24%** (119/491)
  - **By Sex and Pclass Together:**
    - Female, Class 1: **96.74%** (89/92)
    - Female, Class 2: **92.11%** (70/76)
    - Female, Class 3: **50.00%** (72/144)
    - Male, Class 1: **36.89%** (45/122)
    - Male, Class 2: **15.74%** (17/108)
    - Male, Class 3: **13.54%** (47/347)
- **6x6 Correlation Matrix (Excluding `adult_male` & `alone`):**
  Features: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`.
  - `adult_male` and `alone` are strictly excluded as they are derived/redundant flags rather than independent measured features.

| Feature | survived | pclass | age | sibsp | parch | fare |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **survived** | 1.000 | -0.336 | -0.070 | -0.034 | 0.083 | 0.255 |
| **pclass** | -0.336 | 1.000 | -0.337 | 0.082 | 0.017 | **-0.548** |
| **age** | -0.070 | -0.337 | 1.000 | -0.233 | -0.171 | 0.094 |
| **sibsp** | -0.034 | 0.082 | -0.233 | 1.000 | **0.415** | 0.161 |
| **parch** | 0.083 | 0.017 | -0.171 | **0.415** | 1.000 | 0.218 |
| **fare** | 0.255 | **-0.548** | 0.094 | 0.161 | 0.218 | 1.000 |

- **Top Two Strongest Off-Diagonal Correlations:**
  1. **`pclass` & `fare` ($r = -0.548, |r| = 0.548$):** This is the strongest correlation in the dataset. Because passenger class is numerically coded as 1, 2, and 3, this negative correlation demonstrates that upper ticket classes (lower numeric tier) paid substantially higher ticket prices.
  2. **`sibsp` & `parch` ($r = 0.415, |r| = 0.415$):** This positive correlation indicates that passengers traveling with siblings/spouses (`sibsp`) were significantly more likely to also travel with parents/children (`parch`), reflecting coordinated family units rather than solitary travelers.

### Task 5: Multivariate Data Story
Four distinct charts were generated and saved to `analytics/plots/03_multivariate_data_story.png`:
1. **Survival Rate by Passenger Class & Sex (Bar Plot):** Shows an overwhelming gender gap amplified by socio-economic class. Females in Class 1 (96.74%) and Class 2 (92.11%) experienced near-total survival, whereas Class 3 males suffered 86.46% mortality.
2. **Age Distribution by Sex & Survival (Split Violin Plot):** Demonstrates that among males, survival was concentrated in young boys under 10 years of age, whereas adult males experienced low survival across all ages. For females, survival remained consistently high across all age deciles.
3. **Fare vs. Age Dispersed by Survival (Scatter Plot):** High-fare passengers (£100+) survived almost universally across all age brackets. Passengers paying under £30 experienced heavy mortality, demonstrating that cabin deck placement and economic privilege directly dictated evacuation priority.
4. **Survival Rate by Total Family Size (Bar Plot):** Solo travelers had low survival (~30%). Small family units (2–4 members) achieved optimal survival (>55–70%) due to mutual assistance, whereas large families (5+ members) saw survival drop below 20% due to difficulty coordinating large groups during evacuation.

### Task 6: Exploratory Standardization Sanity Check
Before-and-after Z-score normalization check ($z = \frac{x - \mu}{\sigma}$):
- `age`: Raw Mean = 29.32, Std = 12.98 $\rightarrow$ Transformed Mean = $2.80 \times 10^{-16} \approx 0.0$, Std = 1.00.
- `fare`: Raw Mean = 32.10, Std = 49.70 $\rightarrow$ Transformed Mean = $1.36 \times 10^{-16} \approx 0.0$, Std = 1.00.
- *Note:* This sanity check was performed on the pre-split dataset solely to inspect distributional centering. The predictive modeling pipeline enforces strict train-only fitting.

---

## 3. Part B — Predictive Modeling Pipeline

### Task 7 & 8: Stratified Split & Train-Only Preprocessing
- **Class Balance:** Deceased (0) = 549 (61.75%), Survived (1) = 340 (38.25%).
- **Stratification Justification:** Stratified splitting is essential because random sampling across an imbalanced 62:38 target distribution risks variance in class ratios between training and testing sets, corrupting evaluation metrics. Stratification guarantees identical class proportions across folds.
- **Leakage Prevention:** All scalers, imputers, and encoders are contained in a `ColumnTransformer` that is **fit exclusively on the 80% training split (711 rows)** and applied in **transform-only mode on the 20% test split (178 rows)**.

### Task 9 & 10: Model Training & Evaluation Suite
Three classifiers were trained on the identical training fold and evaluated on the test fold:

| Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8146** | **0.8070** | 0.6765 | **0.7360** | **0.8582** |
| **Decision Tree (max_depth=4)** | 0.8090 | 0.8036 | 0.6618 | 0.7258 | 0.8209 |
| **Random Forest (n=100)** | 0.8034 | 0.7619 | **0.7059** | 0.7328 | 0.8359 |

- The Decision Tree was rendered via `plot_tree` with feature names and class labels, saved as `analytics/plots/04_decision_tree_structure.png`.
- ROC curves for all three models were plotted together and saved to `analytics/plots/05_roc_curves_comparison.png`.

### Task 11: Imbalance Handling Comparison
Evaluated across three training configurations on Random Forest:

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Unweighted)** | **0.7619** | **0.7059** | **0.7328** |
| **`class_weight='balanced'`** | 0.7460 | 0.6912 | 0.7176 |
| **SMOTE (Train fold only)** | 0.7581 | 0.6912 | 0.7231 |

- **Conclusion:** The **Baseline (Unweighted)** model produced the strongest balance of precision (0.7619) and F1-score (0.7328). Because the target imbalance is moderate (62:38), aggressive synthetic oversampling or penalization caused false positives that reduced precision without yielding a compensating recall boost.

### Task 12: Hyperparameter Tuning with GridSearchCV & OOB Score
- Grid search over: `n_estimators: [50, 100, 200]`, `max_depth: [4, 6, 8, None]`, `max_features: ['sqrt', 'log2']`.
- Estimator initialized with `RandomForestClassifier(oob_score=True, random_state=42)`.
- **Best Parameters:** `{'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 100}`
- **Best 5-Fold CV F1 Score:** **0.7749**
- **Corresponding Out-of-Bag (OOB) Score:** **0.8101**

### Task 13: Regression Side-Task (Predicting Fare)
Multivariate Linear Regression predicting `fare` from demographic and travel features:
- **MAE:** **17.88**
- **RMSE:** **40.49**
- **$R^2$:** **0.3854**
- **Adjusted $R^2$:** **0.3601**
- **Heteroscedasticity Analysis:** The residual plot (`analytics/plots/06_regression_residuals.png`) displays a pronounced fan/funnel shape, where residual variance expands dramatically at higher predicted fares. This confirms substantial **heteroscedasticity** due to high-value luxury tickets that violate constant variance assumptions.

---

## 4. Final Model Comparison & Recommendation

### Unified Benchmark Table

#### Classification Models (Classification Metric Scale)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8146** | **0.8070** | 0.6765 | **0.7360** | **0.8582** |
| **Decision Tree** | 0.8090 | 0.8036 | 0.6618 | 0.7258 | 0.8209 |
| **Random Forest** | 0.8034 | 0.7619 | **0.7059** | 0.7328 | 0.8359 |

#### Regression Model (Continuous Metric Scale)
| Model | Target | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multivariate Linear Regression** | Fare (£) | 17.88 | 40.49 | 0.3854 | 0.3601 |

### Final Written Deployment Recommendation
**Recommended Model for Deployment: Logistic Regression (with Tuned Random Forest as Secondary Ensemble).**
- **Justification:** Logistic Regression achieved the highest overall accuracy (81.46%), the highest precision (80.70%), the highest F1 score (0.7360), and the highest discrimination capability with an AUC of 0.8582. Its linear coefficients guarantee strict regulatory interpretability and microsecond inference latency in production.
- Where capturing non-linear interactions is paramount, the Tuned Random Forest serves as a powerful alternative with an Out-of-Bag (OOB) score of 0.8101 and the highest test recall (0.7059).

---

## 5. Pipeline Export & Verification
The full end-to-end fitted pipeline (`ColumnTransformer` preprocessing + `RandomForestClassifier`) is serialized to:
`analytics/titanic_best_pipeline.joblib`

Verified via `joblib.load()` on raw, unpreprocessed inputs, confirming automated imputation, encoding, scaling, and prediction without external preprocessing scripts.
