# Breast Cancer Patient Outcomes Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-25%20Passed-16877F)

A single-page interactive Streamlit application for exploring patient outcomes across a SEER-style breast cancer cohort (`n = 4,024` raw records; `n = 3,443` cleaned records).

The dashboard reproduces the complete data-cleaning and feature-engineering pipeline from `Breast_Cancer.ipynb`, pairs visual cohort exploration with an eight-step **Statistics & Probability** module, and surfaces automated, rule-based analytical insights.

> **Disclaimer:** This project is for **descriptive and educational analysis only**. All reported rates are unadjusted observational proportions. It is not medical advice and must not be used for clinical decision-making.

---

## Quick Start

### 1. Prerequisites
Place the following files in the project root directory:
- `app.py`
- `statistics_section.py`
- `Breast_Cancer.csv` *(input dataset — 4,024 rows × 16 columns)*
- `requirements.txt`

### 2. Installation & Execution
Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

*(If `streamlit` is not on your system PATH on Windows, run `python -m streamlit run app.py` instead.)*

### 3. Access
Open **http://localhost:8501** in your browser. There is no manual file-upload step—the application automatically ingests, validates, and cleans `Breast_Cancer.csv` on startup.

---

## Architecture & Tech Stack

| Layer | Technology | Role in Application |
|---|---|---|
| **Core Language** | Python (3.10+) | Data pipeline, statistical modelling, and UI orchestration |
| **Web Framework** | Streamlit (`>=1.40`) | Single-page layout, form-based sidebar filtering, state caching, CSV exports |
| **Data Engineering** | pandas (`>=2.0`) | Ingestion, schema validation, deduplication, IQR filtering, feature engineering |
| **Numerical Computing** | NumPy (`>=1.24`) | Empirical survival curves, histogram binning, Monte Carlo sampling |
| **Visualisation** | Plotly (`>=5.18`) | Interactive charts (donuts, bars, histograms, violins, grouped boxplots, heatmaps) |
| **Statistical Inference** | SciPy (`>=1.10`) | Point-biserial correlation, Welch's *t*-test, one-way ANOVA, OLS regression, Chi-square, Wilson CIs |
| **Styling & UI** | Custom HTML / CSS | Hero banner, sticky section navigation, KPI cards, responsive grid system |
| **Quality Assurance** | pytest (`>=7.4`) | 25 automated unit and UI-render tests using synthetic cohorts |

**Performance Design:** `@st.cache_data` caches both the cleaned baseline dataset (keyed on file modification time) and the rendered figure dictionary per filter combination (`max_entries=24`), ensuring near-instant filter updates after initial load.

---

## Project Structure

```text
BreastCancer/
├── app.py                  # Main Streamlit dashboard and data pipeline
├── statistics_section.py   # Section 06: Educational statistics & probability module
├── Breast_Cancer.csv       # Input dataset (SEER-style cohort, 4,024 rows × 16 cols)
├── requirements.txt        # Python package dependencies
├── test_app.py             # Automated pytest suite (25 tests)
├── README.md               # Project documentation
├── .gitignore              # Git exclusion rules
└── .streamlit/
    └── config.toml         # Custom light theme and minimal toolbar configuration
```

To verify your dataset dimensions before launching:

```bash
python -c "import pandas as pd; print(pd.read_csv('Breast_Cancer.csv', encoding='latin1').shape)"
# Expected output: (4024, 16)
```

---

## Dashboard Walkthrough

The interface is organised as a continuous, single-page report with a **sticky navigation bar** for instant section jumping:

| # | Section | Key Contents |
|---|---|---|
| **01** | **Overview** | Cohort KPI tiles (total patients, recorded deaths, crude mortality, median follow-up), outcome donut chart, and mortality gradient by 6th Stage. |
| **02** | **Patient Profile** | Univariate distributions and categorical breakdowns across demographics, tumor characteristics, lymph node involvement, and ER/PR receptor status. |
| **03** | **Outcomes** | Subgroup mortality rates benchmarked against the cohort average (red = above average, teal = at/below average), plus Alive vs. Dead violin and boxplot comparisons. |
| **04** | **Follow-up** | Yearly follow-up distribution, 60-month milestone KPIs, and the cumulative observed survival curve. |
| **05** | **Data Quality** | Pearson correlation heatmap, point-biserial correlation table, notebook reconciliation audit, per-column IQR outlier table, and one-click CSV exports. |
| **06** | **Statistics & Probability** | Guided 8-step statistical analysis dynamically computed on the active cohort view. |
| **07** | **Insights** | Auto-generated findings paired with analytical recommendations, deeper modelling next steps, and a plain-language glossary. |

### Sidebar Controls
- **Filters:** `Race`, `6th Stage`, and `Age Group` (batched inside a form to prevent unnecessary reruns while selecting).
- **Scope:** Active filters dynamically update all charts, KPIs, statistical tests, and insight cards. Data cleaning and notebook reconciliation always evaluate the full underlying file.

---

## Data Pipeline & Notebook Reconciliation

`prepare_dataset()` enforces strict schema validation and reproduces the exact cleaning pipeline from `Breast_Cancer.ipynb`:

1. **Header Normalisation:** Strips column whitespace and standardises known dataset typos (`Reginol Node Positive` → `Regional Node Positive`, `differentiate` → `Differentiate`).
2. **Type & Value Validation:** Coerces numeric fields, strips categorical strings, and validates allowed categories for `Status`, `Estrogen Status`, and `Progesterone Status`.
3. **Grade Standardisation:** Maps raw codes (`1`, `2`, `3`, `anaplastic; Grade IV`) to readable labels (`Grade 1` through `Grade 4 (Anaplastic)`).
4. **Deduplication:** Removes exact duplicate records (`1` row removed).
5. **Simultaneous IQR Outlier Removal:** Evaluates the `1.5 × IQR` rule across `Age`, `Tumor Size`, `Regional Node Examined`, `Regional Node Positive`, and `Survival Months` on the de-duplicated dataset. Any record falling outside the `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` bounds in one or more columns is excluded (`580` rows removed).
6. **Feature Engineering:** Derives `Lymph_Node_Ratio`, `Survival_Years`, `Age_Group`, `Tumor_Category`, `Hormone_Profile`, `Survival_Year_Bracket`, and `Status_Binary`.

### Baseline Reconciliation Table
Section 05 automatically verifies the loaded CSV against the notebook baseline:

| Pipeline Metric | Notebook Baseline | Status |
|---|---:|---|
| Raw records | `4,024` | ✓ Match |
| Missing values | `0` | ✓ Match |
| Alive / Dead in raw file | `3,408` / `616` | ✓ Match |
| Duplicates removed | `1` | ✓ Match |
| Rows after de-duplication | `4,023` | ✓ Match |
| IQR outliers removed | `580` | ✓ Match |
| **Clean rows analysed** | **`3,443`** | **✓ Match** |
| Alive / Dead after cleaning | `3,021` / `422` (12.3% mortality) | ✓ Match |

---

## Statistics & Probability Module (Section 06)

Designed as an interactive, 8-step curriculum applied directly to the filtered patient cohort:

1. **Descriptive Statistics:** Mean, median, standard deviation, min, and max across all continuous clinical variables.
2. **Empirical Probability & Bayes' Rule:** Marginal, joint, and conditional probabilities (`P(Dead)`, `P(IIIC)`, `P(Dead | IIIC)`, and `P(IIIC | Dead)`).
3. **Theoretical Distributions:** Observed tumor size density overlaid with a parametric Normal curve, alongside a Binomial mortality model (`n = 20`).
4. **Sampling & Estimation:** 95% Wilson score confidence interval for cohort mortality and a 500-iteration Central Limit Theorem (CLT) simulation (`n = 30` per sample).
5. **Parametric Hypothesis Testing:** Two-sample Welch's *t*-test (tumor size in Dead vs. Alive) and one-way ANOVA (tumor size across 6th Stage groups).
6. **Linear Regression:** Ordinary Least Squares (OLS) regression modelling positive lymph nodes as a function of tumor size (`r`, slope, `R²`, and *p*-value).
7. **Categorical Association:** Contingency table analysis (`6th Stage × Status`) using Pearson's Chi-square test, Cramér's *V* effect size, and automated small-cell warnings (`expected < 5`).
8. **Analytical Reporting:** Guidelines on effect sizes, confidence intervals, multiple testing, and model limitations.

---

## Methodological Limitations

- **Unadjusted Proportions:** Subgroup mortality rates do not control for confounding variables (e.g., age, stage, or receptor status).
- **Crude Survival Curve:** The 60-month curve represents the empirical proportion of patients without a recorded death by month *t*. It is **not** a Kaplan–Meier estimator and does not account for right-censoring.
- **Small Sample Instability:** Subgroups with `n < 30` (such as `Grade 4 (Anaplastic)`) are flagged with an asterisk (`*`) to warn against over-interpretation.
- **Recommended Next Steps:** Formal survival modelling should incorporate Kaplan–Meier product-limit estimators and multivariable Cox proportional-hazards regression.

---

## Testing

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

The 25-test suite uses synthetic fixtures to verify:
- Data pipeline integrity, typo renaming, deduplication, IQR filtering, and feature engineering.
- Edge-case resilience (empty views, single-row cohorts, single-class outcomes, zero-variance inputs).
- Mathematical accuracy of Wilson intervals, conditional probabilities, statistical tests, and full Streamlit AppTest rendering.

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `'streamlit' is not recognized` | Run via Python module: `python -m streamlit run app.py`. |
| `ModuleNotFoundError: statistics_section` | Ensure `statistics_section.py` is in the exact same directory as `app.py`. |
| `Breast_Cancer.csv was not found` | Place `Breast_Cancer.csv` in the project root next to `app.py`. |
| Reconciliation table shows `Differs` | Press `C` inside the running Streamlit app to **Clear Cache**, then refresh (`Ctrl+F5`). |
| Statistical test displays `Not available` | The active sidebar filters leave fewer than 2 groups or insufficient observations. Click **Reset filters**. |

---

## 🌐 Live Demo

Explore the live interactive dashboard here: **[https://breast-cancer-outcomes.streamlit.app](https://breast-cancer-outcomes.streamlit.app)**


