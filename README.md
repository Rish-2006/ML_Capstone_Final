
# Machine Learning Capstone Project

**E-Commerce Sales Prediction, Customer Churn Classification, and Wholesale Customer Segmentation**

> **Project scope:** Review 1 covers regression and the first five classification models. Review 2 extends classification to ten models and introduces clustering. Results documented here are the saved notebook results; they are not claims of deployment or independent external validation.

## Table of Contents

1. [Project overview and objectives](#1-project-overview-and-objectives)
2. [Datasets and problem statements](#2-datasets-and-problem-statements)
3. [Repository structure and execution](#3-repository-structure-and-execution)
4. [Review 1: regression](#4-review-1-regression)
5. [Review 1 and 2: churn classification](#5-review-1-and-2-churn-classification)
6. [Review 2: customer clustering](#6-review-2-customer-clustering)
7. [Evaluation concepts and formulas](#7-evaluation-concepts-and-formulas)
8. [Main findings, limitations and future work](#8-main-findings-limitations-and-future-work)
9. [Rubric alignment and viva guide](#9-rubric-alignment-and-viva-guide)

## 1. Project overview and objectives

This capstone demonstrates **supervised learning** (regression and binary classification) and **unsupervised learning** (clustering) on three business datasets. It follows a reproducible notebook workflow: data inspection → exploratory data analysis (EDA) → preprocessing and feature engineering → modeling → evaluation → interpretation.

**Objectives:**

- Predict numerical product sales (`Units_Sold`) from product, marketing and date features.
- Predict whether an e-commerce customer will churn (`Churn`: 0 = no, 1 = yes).
- Identify groups of wholesale customers with similar annual spending patterns without using target labels.
- Compare models using task-appropriate metrics, explain model selection and recognize limitations.

**Learning types:** Regression predicts numerical quantities; classification predicts discrete labels; clustering discovers groups without labeled targets.

## 2. Datasets and problem statements

| Task | File | Size | Target or grouping basis |
|---|---|---|---|
| Regression | `Ecommerce_Sales_Prediction_Dataset.csv` | 1,000 records | `Units_Sold` |
| Binary classification | `E Commerce Dataset.xlsx` (sheet `E Comm`) | 5,630 records, 20 original columns | `Churn` |
| Clustering | `Wholesale customers data.csv` | 440 records, 8 original columns | Six annual-spending features; no target |

### 2.1 Sales prediction

Inputs include `Date`, `Product_Category`, `Price`, `Discount`, `Customer_Segment` and `Marketing_Spend`. The dependent variable `Units_Sold` is numeric, so regression is appropriate. The notebook explores whether the observed features explain sales quantity.

### 2.2 Customer churn prediction

Customer information includes tenure, device and payment preferences, city tier, warehouse distance, satisfaction, complaints, order activity and cashback. `CustomerID` is an identifier and is excluded from modeling. The binary `Churn` label is the supervised target. A churn prediction may help a business prioritize retention efforts, although model scores alone do not establish real-world intervention value.

### 2.3 Wholesale customer segmentation

The six spending variables are `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper` and `Delicassen`. The dataset also has `Channel` and `Region`, which are not used as spending features in the clustering analysis. The objective is to discover purchasing segments rather than predict a preexisting label.

## 3. Repository structure and execution

A clean submission structure is:

```text
ML_Capstone/
├── data/
│   ├── E Commerce Dataset.xlsx
│   ├── Ecommerce_Sales_Prediction_Dataset.csv
│   └── Wholesale customers data.csv
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   ├── review2_classification.ipynb
│   └── review2_clustering.ipynb
├── README.md
└── requirements.txt
```

**Important:** The supplied ZIP may contain duplicate datasets, notebook checkpoints, local Git metadata or virtual-environment files. The tree above is the **recommended cleaned submission layout**, not a guarantee that the archive currently has only these items.

### Environment setup

Use Python 3 with JupyterLab/Jupyter Notebook and the dependencies in `requirements.txt`. Typical packages are `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `openpyxl`, `jupyterlab` and `notebook` (check the repository's actual requirements file for exact versions).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: `regression.ipynb` → `classification.ipynb` → `review2_classification.ipynb` → `review2_clustering.ipynb`. **The Review 2 classification notebook may depend on variables created by the earlier classification notebook.** If it does, run its required upstream cells in the same kernel or refactor it to load and preprocess its own data. Paths in `pd.read_csv()` / `pd.read_excel()` must match the actual working directory; the churn workbook uses `sheet_name="E Comm"`. For a final demonstration, restart the kernel and run all cells to identify hidden state dependencies.

## 4. Review 1: regression

### 4.1 Exploratory data analysis

The notebook inspects data shape, column types, missing values, duplicates, summary statistics and distributions. It plots:

- `Units_Sold` histogram: most observations are concentrated near 25–35 units, approximately bell-shaped.
- `Price` vs `Units_Sold` and `Marketing_Spend` vs `Units_Sold` scatterplots: no strong visible trend.
- Numerical correlation heatmap: `Price` (~0.02), `Discount` (~−0.03) and `Marketing_Spend` (~0.01) have near-zero **linear** correlation with `Units_Sold`.
- Distributions and counts for numeric and categorical features.
- Boxplots: possible outliers in `Units_Sold`; no prominent outliers in the displayed price, discount or marketing-spend boxplots.

Near-zero Pearson correlation does **not** prove that nonlinear relationships are impossible.

### 4.2 Feature preparation

1. Parse `Date` into datetime format.
2. Engineer `Year`, `Month`, `Day` and `DayOfWeek`.
3. Exclude the raw date and `Units_Sold` from predictors.
4. One-hot encode `Product_Category` and `Customer_Segment` (`drop_first=True`).
5. Create an 80/20 train/test split (`random_state=42`): **800 training / 200 test records**, with **13 encoded predictors**.
6. Standardize using `StandardScaler`: fit on training features, transform training and test features using the same fitted scaler.

Standardization is especially relevant to distance-based, kernel and regularized models; decision trees do not inherently require it. The held-out test set is used for evaluation rather than model fitting.

### 4.3 Regression algorithms and results

| Model | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Linear Regression | -0.0438 | 7.5134 | 5.9628 |
| Polynomial Regression | -0.1166 | 7.7710 | 6.1912 |
| Ridge Regression | -0.0437 | 7.5131 | 5.9626 |
| **Lasso Regression** | **-0.0103** | **7.3918** | **5.9008** |
| **Elastic Net** | **-0.0103** | **7.3918** | **5.9008** |
| Decision Tree Regression | -1.1052 | 10.6703 | 8.4150 |
| Random Forest Regression | -0.1066 | 7.7363 | 6.0910 |
| Gradient Boosting Regression | -0.0535 | 7.5485 | 6.0057 |
| Support Vector Regression (SVR) | -0.0535 | 7.5482 | 5.9513 |
| KNN Regression | -0.2242 | 8.1369 | 6.3140 |

**Interpretation:** Lasso and Elastic Net tie for the highest test R² and lowest RMSE/MAE among the ten tested models, but **all test R² values are negative**. Negative test R² means predictions perform worse than a constant test-set mean baseline on the evaluated samples. This is a legitimate result and does not, by itself, imply incorrect code. The available inputs appear to contain little predictive information for the sales target.

### 4.4 Hyperparameter tuning and cross-validation

`GridSearchCV` with five-fold CV was used to tune:

- **Lasso:** best `alpha = 1`; tuned test R² ≈ −0.0103, RMSE ≈ 7.3918, MAE ≈ 5.9008.
- **Elastic Net:** best `alpha = 1`, `l1_ratio = 0.9`; tuned test R² ≈ −0.0103, RMSE ≈ 7.3918, MAE ≈ 5.9008.

Tuning did **not** improve the reported held-out test metrics relative to the original results. Five-fold CV mean R² was approximately **−0.0129** for both, with standard deviation approximately **0.0090**. The notebook includes a residual plot and actual-vs-predicted plot for tuned Lasso. Predictions cluster around 29–30 units, suggesting the model has learned a near-constant prediction rather than explaining the observed variation.

### 4.5 Tree-based feature importance

Random Forest impurity-based feature importance highlights `Marketing_Spend`, `Price` and `Discount` among the largest relative importances. **Importance is model-specific and does not imply causation**; a model can rank features even if its predictive performance is weak.

## 5. Review 1 and 2: churn classification

### 5.1 Dataset and preprocessing

- **5,630** records; `Churn` distribution: **4,682 class 0** (about 83.2%) and **948 class 1** (about 16.8%).
- Missing numerical values occur in variables including `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount` and `DaySinceLastOrder`.
- The notebook uses median imputation for numerical features, mode imputation for categorical features, one-hot encoding and `StandardScaler`.
- `CustomerID` is removed as a non-predictive identifier; no exact duplicate rows were reported.
- Stratified 80/20 split: **4,504 training**, **1,126 test**; **29 encoded features**.

**Methodological caution:** In the current notebook workflow, imputation and dummy encoding are performed before splitting. A stronger leakage-resistant production workflow would fit all learned preprocessing (including imputation) on the training data only, preferably with a `Pipeline`/`ColumnTransformer`.

### 5.2 Ten classification models

**Review 1 / Part A:** Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree and SVM.

**Review 2 / Part B:** Random Forest, AdaBoost, Gradient Boosting, Bagging and MLP (Multilayer Perceptron).

**Key concepts:**

| Algorithm | Core idea | Important consideration |
|---|---|---|
| Logistic Regression | Models binary class probability using a sigmoid/logit relationship | Linear decision boundary in feature space |
| KNN | Classifies using labels of nearest neighbors | Sensitive to feature scaling |
| Gaussian Naive Bayes | Applies Bayes' theorem with conditional independence and Gaussian assumptions | Assumptions may not hold in real customer data |
| Decision Tree | Learns feature-threshold splits | Unrestricted trees may overfit |
| SVM | Finds a large-margin separating boundary; kernels enable nonlinear separation | Scaling and kernel choice matter |
| Random Forest | Aggregates randomized decision trees | Reduces variance, less directly interpretable |
| AdaBoost | Sequentially emphasizes difficult/misclassified observations | Can be sensitive to noise |
| Gradient Boosting | Sequentially minimizes loss using new weak learners | Requires regularization/tuning |
| Bagging | Trains base estimators on bootstrap samples and aggregates votes | Helps reduce variance |
| MLP | Learns nonlinear representations with hidden layers and backpropagation | Scaling, convergence and tuning matter |

**Bagging vs boosting:** Bagging fits base learners largely independently on resampled data and aggregates them. Boosting fits learners sequentially to improve the current ensemble.

### 5.3 Review 1 preliminary model comparison

| Classifier | Accuracy | Weighted F1 |
|---|---:|---:|
| Decision Tree | 0.9663 | 0.9663 |
| SVM | 0.9192 | 0.9125 |
| KNN | 0.8970 | 0.8879 |
| Logistic Regression | 0.8890 | 0.8808 |
| Gaussian Naive Bayes | 0.7043 | 0.7391 |

These are the recorded Part A notebook results. The Review 2 notebook's separate ten-model comparison may use a different training/evaluation workflow; **do not assume all values must be identical between notebooks**.

### 5.4 Review 2 final model comparison and selection

The Review 2 notebook evaluates **all ten classifiers** using **Accuracy, Precision, Recall, Weighted F1 and ROC-AUC** and provides a comparison table. Refer to `review2_classification.ipynb` for the exact row-by-row five-metric values; do not substitute the earlier Part A table for the final comparison.

The highest reported **weighted F1** belongs to **Bagging**:

| Bagging metric | Reported result |
|---|---:|
| Accuracy | **0.9849 (98.49%)** |
| Weighted F1 | **0.9849** |
| ROC-AUC | **0.9978** |

This supports choosing Bagging as the **best reported model on this evaluation**, not a guarantee of equivalent performance on new populations. For churn decisions, inspect class-1 recall, precision and the confusion matrix: weighted averages may conceal weak minority-class detection.

### 5.5 Hyperparameter tuning (Review 2)

`GridSearchCV` with **five-fold cross-validation** was used to tune a **Decision Tree**, chosen as an interpretable model. The recorded weighted F1 changed as follows:

| Decision Tree | Weighted F1 |
|---|---:|
| Before tuning | 0.9646 |
| After tuning | 0.9744 |
| Absolute improvement | **+0.0098** (0.98 percentage points) |

**Distinction to explain to reviewers:** Bagging is the top reported overall classifier, but Decision Tree is the model tuned in the notebook. If the rubric is interpreted as requiring tuning the *final selected best model*, tune Bagging too and document its before/after held-out results. Avoid claiming Bagging was tuned unless the notebook actually demonstrates it.

### 5.6 Confusion matrices and ROC-AUC

For the positive class (`Churn=1`):

- **TP:** correctly predicted churn; **TN:** correctly predicted non-churn.
- **FP:** predicted churn but customer stayed; **FN:** predicted non-churn but customer churned.
- **Precision** emphasizes avoiding false churn alerts; **recall** emphasizes finding actual churners.
- **ROC-AUC** evaluates ranking discrimination over classification thresholds; an AUC near 1 is excellent discrimination on the evaluated sample.

The class imbalance makes churn-class recall and false negatives especially important when discussing business use.

## 6. Review 2: customer clustering

### 6.1 Dataset preparation and goals

The **440-customer Wholesale Customers** dataset is clustered on six spending categories: `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper` and `Delicassen`. The goal is to discover groups of customers with similar purchasing behavior. Scaling is important because K-Means and Ward-linkage hierarchical clustering rely on distances.

### 6.2 K-Means

K-Means iteratively assigns observations to their nearest centroid and recomputes centroids to minimize **within-cluster sum of squares (WCSS / inertia)**. The elbow curve plots inertia against K; it helps inspect diminishing returns as the number of clusters increases. The notebook reports a **five-cluster configuration**. A clear elbow may not exist, so cluster count should also be evaluated for stability, size balance and business interpretability.

### 6.3 Agglomerative Hierarchical Clustering

Agglomerative clustering starts with one cluster per customer and repeatedly merges the nearest clusters according to a **linkage rule**. The **dendrogram** visualizes the hierarchy of merges and their linkage distances. The main final comparison uses **Ward linkage**, which seeks merges with a small increase in within-cluster variance. The notebook also explores other linkage choices.

### 6.4 Evaluation metrics and recorded results

| Metric | Desired direction | K-Means (K=5) | Agglomerative (Ward, 5 clusters) |
|---|---|---:|---:|
| Silhouette Score | Higher | **0.3690** | 0.2399 |
| Davies-Bouldin Index | Lower | **0.9125** | 0.9727 |
| Calinski-Harabasz Index | Higher | **162.4134** | 128.6450 |

**Interpretation:** For the specific five-cluster configuration and Ward-linkage comparison, K-Means scores better on all three reported internal metrics. This is not proof that K-Means is universally better or that five clusters is the uniquely optimal choice.

### 6.5 Cluster sizes and important limitation

| Cluster label | K-Means count | Agglomerative (Ward) count |
|---|---:|---:|
| 0 | 270 | 153 |
| 1 | 10 | 5 |
| 2 | 63 | 104 |
| 3 | 96 | 1 |
| 4 | 1 | 177 |

**Important:** Both algorithms create a **singleton cluster** (one customer). This may represent an extreme spending profile rather than a useful segment. Before operational use, inspect that observation, consider skew-aware transformations such as `log1p`, compare different K values and examine cluster-level spending summaries. Do not automatically discard an outlier without investigating its business significance. Cluster numbers are arbitrary identifiers, not rankings.

### 6.6 PCA and t-SNE visualizations

- **PCA (Principal Component Analysis)** is a linear dimensionality-reduction method. The notebook uses two principal components for plotting the K-Means and Agglomerative cluster labels. The first two components explain approximately **72.46%** of variance in the analyzed representation.
- **t-SNE** is a nonlinear visualization method emphasizing local neighborhood relationships. It provides an additional two-dimensional view of customer groupings.

PCA and t-SNE are visualization aids here. A visually separated 2D plot does not by itself establish cluster quality or causal customer segments.

## 7. Evaluation concepts and formulas

### 7.1 Regression

For actual values `y`, predictions `ŷ`, and sample size `n`:

- **MAE:** `mean(|y − ŷ|)` — average absolute error; **lower is better**.
- **MSE:** `mean((y − ŷ)²)` — squared error; emphasizes larger mistakes.
- **RMSE:** `sqrt(MSE)` — error in the target's original units; **lower is better**.
- **R²:** `1 − Σ(y − ŷ)² / Σ(y − mean(y))²` — relative to a mean baseline; **higher is better** and can be negative.
- **Residual:** `actual − predicted`; residual plots help reveal systematic patterns and model limitations.

### 7.2 Classification

For the positive class (`Churn=1`):

- **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`.
- **Precision:** `TP / (TP + FP)`.
- **Recall / TPR:** `TP / (TP + FN)`.
- **F1:** `2 × Precision × Recall / (Precision + Recall)`.
- **Weighted F1:** support-weighted average of per-class F1 scores.
- **FPR:** `FP / (FP + TN)`.
- **ROC-AUC:** area under the TPR-vs-FPR curve as the threshold changes; larger generally indicates better ranking ability.

### 7.3 Clustering

- **Silhouette:** `(b − a) / max(a, b)`, where `a` is mean within-cluster distance and `b` is the nearest other-cluster mean distance; **higher is better** (usually between −1 and +1).
- **Davies-Bouldin:** average similarity of each cluster to its most similar other cluster; **lower is better**.
- **Calinski-Harabasz:** ratio of between-cluster to within-cluster dispersion (with degrees-of-freedom scaling); **higher is better**.
- **Inertia / WCSS:** sum of squared distances to assigned centroids; decreases as K increases and is not by itself a fair cross-K model-quality score.

### 7.4 Core methodological vocabulary

- **Overfitting:** strong fit to training data but weak generalization; inspect train vs validation/test results.
- **Underfitting:** model too simple to learn the underlying pattern.
- **Parameter:** learned during fitting (e.g., model coefficients).
- **Hyperparameter:** chosen/configured before fitting (e.g., `max_depth`, `n_estimators`, `alpha`).
- **Cross-validation:** rotates validation subsets within the training data; avoids relying on a single validation partition.
- **Data leakage:** test/future information unintentionally influences training or preprocessing; can inflate evaluation scores.
- **`random_state`:** makes randomized operations reproducible; 42 has no special statistical meaning.
- **`fit` / `transform` / `predict`:** learn parameters / apply a learned transformation / generate predictions.

## 8. Main findings, limitations and future work

### Main findings

1. **Regression:** All reported test R² values are negative. Lasso and Elastic Net minimize reported error but do not deliver strong sales prediction. Weak numerical correlations are consistent with this result.
2. **Classification:** Bagging has the best reported overall weighted F1 (0.9849). Decision Tree tuning shows a measurable +0.0098 improvement in weighted F1.
3. **Clustering:** K-Means outperforms Ward-linkage Agglomerative clustering on the three reported internal metrics at five clusters, but singleton groups complicate interpretation.

### Limitations

- Results come from particular datasets and evaluation splits; external generalization has not been established.
- Churn is imbalanced; weighted scores alone are insufficient for operational model selection.
- Classification preprocessing should be moved into a train-fitted pipeline to reduce leakage risk.
- The regression predictors appear insufficient to predict sales quantities reliably.
- Singleton clustering groups and sensitivity to scaling/outliers require investigation.
- Hyperparameter tuning of Decision Tree does not establish a tuning improvement for Bagging itself.
- No production deployment, real-time monitoring or business-impact experiment is claimed.

### Recommended future work

- Add richer demand drivers (seasonality, promotions, inventory, product history) and validate on chronological holdout data for sales forecasting.
- Evaluate churn-class recall, PR-AUC, probability calibration, decision thresholds and business costs; test the final classifier on independent data.
- Tune Bagging if required by the rubric; compare cross-validation performance and document the exact configuration.
- Profile cluster means/medians, inspect outliers and evaluate alternative K values, log transformations and stability.
- Package all preprocessing and models into reproducible pipelines and remove machine-specific paths.

## 9. Rubric alignment and viva guide

### Review 1 technical work

- Regression EDA, preprocessing, date feature engineering, ten models, R²/RMSE/MAE comparison.
- Regression hyperparameter search, five-fold CV, residual and actual-vs-predicted plots, tree-based feature importance.
- Classification Part A: five algorithms, Accuracy, weighted F1, confusion matrices and preliminary comparison.

### Review 2 technical work

| Rubric item | Evidence in repository |
|---|---|
| A1: Five additional classification models | `review2_classification.ipynb` |
| A2: Compare all ten using five metrics | Ten-model comparison in `review2_classification.ipynb` |
| A3: Selection and hyperparameter tuning | Bagging selection; Decision Tree GridSearchCV and before/after F1 (see caveat in §5.5) |
| B1: K-Means and Agglomerative clustering | `review2_clustering.ipynb` |
| B2: Elbow, dendrogram, three metrics | `review2_clustering.ipynb` |
| B3: PCA and t-SNE visualizations | `review2_clustering.ipynb` |
| C: Quality, README, GitHub and dependencies | This README and repository; GitHub commit history must be verified separately |
| D: Presentation and viva | Require live demonstration and oral explanation; cannot be proven by a README |

**Viva questions and concise answers:**

1. **Why three ML tasks?** They illustrate numerical prediction, categorical prediction and unlabeled segmentation.
2. **Why is `Units_Sold` regression?** It is numerical; `Churn` is binary, so it is classification.
3. **Why did regression have negative R²?** Its predictions were worse than a mean baseline on the test set; predictors show weak signal.
4. **Why scale?** To make distance-sensitive and regularized methods less sensitive to raw feature magnitude.
5. **Why fit scaling on train only?** To avoid information leakage from the test set.
6. **Why is accuracy insufficient for churn?** A majority-class-only predictor already scores about 83%; check recall and F1.
7. **Bagging vs boosting?** Bagging aggregates independently fitted bootstrap models; boosting sequentially improves an ensemble.
8. **Why Bagging?** Highest reported weighted F1, accuracy and strong ROC-AUC in the project comparison.
9. **Why tune Decision Tree?** Interpretability and measurable F1 improvement; acknowledge Bagging itself was not the tuned model.
10. **K-Means vs KNN?** K-Means is unsupervised clustering; KNN is supervised neighbor-based prediction.
11. **Why an elbow plot?** To inspect diminishing reductions in within-cluster inertia as K grows.
12. **What is a dendrogram?** A hierarchy showing cluster merges and linkage distances.
13. **Why K-Means over Ward here?** Better silhouette, Davies-Bouldin and Calinski-Harabasz metrics at five clusters.
14. **Why singleton clusters?** Potentially extreme observations; investigate rather than assuming meaningful segments.
15. **PCA vs t-SNE?** PCA is linear and variance-oriented; t-SNE is nonlinear and neighborhood-oriented.
16. **Does feature importance prove causation?** No, it reports model-specific association/importance.

### Suggested two-minute presentation

> Our capstone studies three machine-learning problems: e-commerce sales prediction, customer churn classification and wholesale customer segmentation. For regression, we analyzed 1,000 sales records, engineered date features, trained ten regressors and compared R², RMSE and MAE. The strongest reported regressors were Lasso and Elastic Net, although all test R² values were negative, showing limited predictive signal. For classification, we used 5,630 customer records and compared ten algorithms across Accuracy, Precision, Recall, weighted F1 and ROC-AUC. Bagging achieved the highest reported weighted F1 of 0.9849; separately, tuning a Decision Tree improved its weighted F1 from 0.9646 to 0.9744. For unsupervised learning, we clustered 440 wholesale customers with K-Means and Agglomerative Clustering. We evaluated the results with three clustering metrics and visualized them using PCA and t-SNE. At five clusters, K-Means performed better than Ward linkage on the reported metrics, though singleton clusters require further investigation. Our conclusion is that classification showed strong held-out performance, regression was limited by weak predictors, and clustering identified possible purchasing patterns that need business validation.

---

**Reproducibility note:** Numeric values above reflect recorded notebook outputs and the supplied README. Before final submission, verify that every notebook runs from a fresh kernel, confirm the precise file paths and dependencies, check GitHub history and ensure the presentation matches the saved results.
