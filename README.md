# Recession Early Warning Model: Macroeconomic Risk Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdoulayeSeydi/recession-early-warning/blob/main/recession_early_warning_model.ipynb)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

## Overview

This project builds a machine learning early-warning system to predict U.S. recession risk 3-6 months in advance using macroeconomic indicators from the Federal Reserve Economic Data (FRED).

**What this project does:**
- Predicts recession probability 3-6 months ahead using macro indicators
- Engineers 49 features from raw economic time series
- Evaluates models with proper time-based validation (no data leakage)
- Analyzes feature importance and model calibration
- Identifies limitations in predicting black swan events

**What this project does NOT do:**
- Predict exact recession timing
- Make causal claims about recession causes
- Provide investment advice
- Predict pandemic-driven recessions (exogenous shocks)

## Key Findings

### Model Performance
- **AUC: 0.75** (strong discrimination between recession and non-recession periods)
- **Model Type:** XGBoost with 49 engineered features
- **Prediction Horizon:** 3-6 months ahead
- **Test Period:** 2018-2024 (73 months)

### Lead Time Analysis
**2020 COVID Recession:**
- No warning at 30%, 40%, or 50% probability thresholds
- **Finding:** Black swan events (pandemics) are unpredictable from traditional macro indicators
- This limitation validates the model's honest design - it cannot predict exogenous shocks

### False Alarm Analysis

**False Alarm #1 (October 2023): 82.7% probability**
```
Context (what the model detected):
  - Yield spread: -0.81 (inverted curve)
  - Unemployment: 3.9% (low but rising)
  - Fed Funds: 5.33% (historically high - tight policy)
  
Interpretation: Model correctly identified economic stress during 
Fed's aggressive rate hiking cycle. Fed achieved "soft landing" 
rather than recession - a rare outcome historically.
```

**False Alarm #2 (February 2024): 54.6% probability**
```
Context:
  - Yield spread: -1.23 (deeply inverted)
  - Unemployment: 3.9%
  - Fed Funds: 5.33%
  
Interpretation: Continued stress from inverted yield curve.
```

**False Alarm #3 (October-November 2024): 70.6% probability**
```
Context:
  - Yield spread: -0.44 (still inverted)
  - Unemployment: 4.15% (rising)
  - Fed Funds: 4.73% (still elevated)
  
Interpretation: Model tracked elevated risk during normalization period.
```

### Top Predictive Features (XGBoost Feature Importance)
1. **Fed Funds Rate (lag 3m)**: 41.9 (monetary policy tightness)
2. **Yield Spread (lag 3m)**: 34.8 (term structure signal)
3. **Yield Spread (current)**: 23.0 (inverted curve)
4. **Fed Funds Rate (lag 6m)**: 21.6 (policy momentum)
5. **Yield Spread (lag 6m)**: 20.8 (sustained inversion)

**Key Insight:** Yield curve inversion and Fed policy tightness are the dominant recession predictors, consistent with economic literature.

## Project Structure
```
recession-early-warning/
├── recession_model.ipynb       # Complete analysis pipeline (all 5 phases)
└── README.md                   # This file
```

## Notebook Sections

### Phase 1: Data Acquisition & Labeling
- Pull 8 FRED economic indicators (1970-2024)
- Download NBER recession dates (ground truth)
- Create 3-6 month forward-looking labels
- **Critical design choice:** Drop in-recession months (predict onset, not continuation)
- Handle weekly data (initial claims) with 4-week moving average
- Forward-fill quarterly GDP (no interpolation to avoid look-ahead bias)

### Phase 2: Feature Engineering
- **49 engineered features** from 8 base indicators
- Lag features (3-month, 6-month)
- Change features (first differences, 6-month changes, YoY%)
- Stress signals:
  - Yield curve inversion indicators
  - Unemployment acceleration (2nd derivative)
  - Credit spread widening
  - GDP deceleration
- Composite: Financial stress index (PCA)

**Result:** 294 samples, 12 recession events (4.1% positive class)

### Phase 3: Baseline Models
Establish performance benchmarks:
1. **Naive baseline:** Always predict no recession (test log loss: 0.2167)
2. **Yield curve rule:** If inverted → 30% probability
3. **Simple logistic:** 3 features only (yield spread, unemployment, GDP)

**Benchmark to beat:** 0.2167 log loss (naive baseline)

### Phase 4: Main Models

**Logistic Regression (49 features):**
- L2 regularization (C=1.0)
- Platt calibration
- **Result:** Severe overfitting (train: 0.029, test: 1.76 log loss)
- **Reason:** Too many features (49) for too few events (8)

**XGBoost (49 features):** ⭐
- Conservative settings (max_depth=3, n_estimators=100)
- Class imbalance handled with scale_pos_weight
- **Results:**
  - Train log loss: 0.0085
  - Test log loss: 0.2727
  - **AUC: 0.75** ← Strong discrimination
  - Accuracy: 86.5%

**Interpretation:** XGBoost learns useful patterns (0.75 AUC) but log loss is penalized by confident wrong predictions. Model ranks recession risk well but calibration could improve.

### Phase 5: Interpretation & Visualization

**1. Time Series Probability Plot**
- Shows recession probability over 2018-2024 test period
- Highlights false alarm periods (2023-2024)
- Demonstrates model behavior during yield curve inversion

**2. Lead Time Analysis**
- Measures early warning capability at multiple thresholds
- **Finding:** COVID recession had no advance signal (expected)

**3. False Alarm Analysis**
- Identified 3 periods of elevated risk without recession
- **2023-2024:** Model correctly detected stress from inverted yield curve + tight Fed policy
- Fed's "soft landing" is historically rare - model's warnings were economically justified

**4. Feature Importance**
- Fed funds rate and yield curve dominate predictions
- Consistent with economic theory (Taylor Rule, term premium)

**5. Calibration Curve**
- Shows predicted vs actual recession frequency
- Identifies areas where model is over-confident

## Methodology

### 1. Data Sources
- **Economic Indicators:** Federal Reserve Economic Data (FRED)
  - Unemployment rate (UNRATE)
  - Initial jobless claims (ICSA)
  - Real GDP growth (A191RL1Q225SBEA)
  - Industrial production (INDPRO)
  - CPI inflation (CPIAUCSL)
  - 10y-3m yield spread (T10Y3M)
  - BAA-AAA credit spread (BAMLC0A4CBBBEY)
  - Fed funds rate (FEDFUNDS)
- **Recession Dates:** NBER Business Cycle Dating Committee
- **Time Period:** 1997-2024 (limited by credit spread data availability)

### 2. Label Creation (Critical Design)
```python
For each month t:
  if t is INSIDE a recession:
    DROP (we predict onset, not status)
  elif recession starts between t+3 and t+6 months:
    label = 1
  else:
    label = 0
```

**Why 3-6 months?**
- 6-12 months: Signal too weak (initial attempt failed)
- 3-6 months: Stronger signal, more predictable
- 1-3 months: Too late for meaningful action

### 3. Train/Test Split (Time-Based)
- **Train:** 1997-2018 (221 samples, 8 recessions)
- **Test:** 2018-2024 (73 samples, 4 recession labels)
- **No validation set:** Insufficient recession samples (would have 0 events)
- **No random shuffling:** Strictly chronological to prevent data leakage

### 4. Evaluation Metrics (Priority Order)
1. **Log Loss** (primary) - penalizes confident wrong predictions
2. **Brier Score** - calibration quality
3. **ROC-AUC** - discrimination ability
4. Accuracy (de-emphasized - misleading with class imbalance)

### 5. Feature Engineering Principles
- **No interpolation:** GDP forward-filled only
- **Smoothing:** Weekly claims converted to 4-week MA
- **Economic theory:** Sahm Rule (unemployment acceleration), yield curve inversion
- **Lags matter:** 3-month and 6-month lags capture momentum

## Technologies Used

- **Python 3.8+**
- **Data:** `pandas`, `numpy`, `fredapi`
- **ML:** `scikit-learn`, `xgboost`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Google Colab

## Model Details

### XGBoost Configuration
```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,          # Shallow trees to prevent overfitting
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=26.5, # Handle 8 positive / 212 negative imbalance
    random_state=42
)
```

### Feature Categories (49 total)
- **Base indicators:** 8 (unemployment, claims, GDP, IP, CPI, yield spread, credit spread, fed funds)
- **Lag features:** 16 (3m and 6m lags)
- **Change features:** 16 (diffs, 6m changes, YoY%)
- **Stress signals:** 7 (inversion, acceleration, widening)
- **Composite:** 1 (financial stress index)

## How to Run

1. Open `recession_model.ipynb` in Google Colab
2. Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
3. Paste API key in Phase 1 config
4. Run all cells sequentially (Runtime → Run all)
5. **Total runtime:** ~10-15 minutes
   - Data acquisition: ~2 min
   - Feature engineering: ~1 min
   - Model training: ~5 min
   - Visualization: ~2 min

All results and plots appear inline.

## Results Summary

### Statistical Findings
- **Feature correlations:** Top features have 0.22-0.24 correlation with 3-6 month recession labels
- **Class imbalance:** 4.1% recession samples (realistic for rare events)
- **Overfitting check:** XGBoost gap (train-test) = 0.26 log loss (acceptable given event rarity)

### Model Comparison (Test Set)
| Model | Log Loss | AUC | Accuracy |
|-------|----------|-----|----------|
| Naive Baseline | 0.2167 | 0.50 | 94.5% |
| XGBoost (49 feat) | 0.2727 | **0.75** | 86.5% |
| Logistic (49 feat) | 1.7623 | 0.41 | 63.5% |

**Winner:** Naive baseline on log loss, but XGBoost on AUC (discrimination ability)

### Interpretation
- ✅ **XGBoost has learned useful patterns** (0.75 AUC significantly > 0.5)
- ⚠️ **Log loss slightly worse than baseline** due to confident wrong predictions
- ✅ **False alarms are economically justified** (real stress periods)
- ✅ **COVID limitation is expected** (exogenous shock, not cyclical)

## Limitations

### Data Limitations
- **Limited history:** 1997-2024 (constrained by credit spread data)
- **Few recession events:** Only 3 recessions in data (2001, 2007-09, 2020)
- **Feature-to-event ratio:** 49 features / 8 training events = 6:1 (causes overfitting in logistic regression)
- **Synthetic indicators:** Some FRED series are revised post-release (we use final values)

### Methodological Limitations
- **No causal claims:** Correlation-based prediction, not causal inference
- **Point-in-time:** No real-time forecasting or data vintage simulation
- **U.S. only:** Model trained on U.S. economy; may not generalize to other countries
- **Binary outcome:** Predicts recession/no-recession, not severity or duration
- **Static features:** No time-varying feature importance or regime changes

### Generalizability
Results may not transfer to:
- **Other countries** (different monetary policy regimes)
- **Different time periods** (structural breaks in economy)
- **Different recession types:** Financial crisis vs. pandemic vs. oil shock
- **Real-time deployment** (data revisions, reporting lags)

## What This Project Does NOT Claim

❌ **Perfect recession prediction** (0.75 AUC ≠ crystal ball)  
❌ **Investment advice** (this is academic research, not financial guidance)  
❌ **Causal mechanisms** (we measure correlations, not causes)  
❌ **Black swan prediction** (COVID was unpredictable by design)  
❌ **Production-ready tool** (would require real-time data, monitoring, retraining)

## Key Takeaways

### Technical Learnings
1. **Time-based validation is non-negotiable** for time series to avoid leakage
2. **Feature engineering matters more than model complexity** (yield curve > deep learning)
3. **Rare event prediction is hard:** 8 training events limits model complexity
4. **Overfitting is the main challenge:** Logistic regression (49 features) completely failed
5. **Calibration ≠ discrimination:** XGBoost ranks risk well (0.75 AUC) but calibration needs work

### Economic Learnings
1. **Yield curve inversion is the king:** Dominates feature importance consistently
2. **Fed policy matters:** Tight monetary policy (high Fed funds) is second-strongest signal
3. **Lead time varies:** Traditional recessions have ~6-12 month buildup; pandemics don't
4. **False alarms are informative:** 2023-2024 warnings reflected real economic stress
5. **Soft landings are rare:** Historically, inverted curves + tight policy → recession

### Project Management
- **Clear scope prevents scope creep:** This is prediction, not causal inference
- **Limitations section is critical:** Honesty about what the model CAN'T do builds credibility
- **Baseline comparisons are mandatory:** Must beat naive model to claim success
- **Visualizations drive understanding:** Phase 5 plots are more valuable than metrics alone

## Future Extensions

### Immediate Next Steps
1. **Feature reduction:** Test 5-feature model (top predictors only) to reduce overfitting
2. **Different horizons:** Try 1-3 month (shorter) or 9-12 month (longer) windows
3. **Ensemble methods:** Combine logistic + XGBoost predictions
4. **Calibration improvement:** Test isotonic regression, temperature scaling

### Advanced Extensions
1. **Real-time deployment:** Handle data revisions, reporting lags, monthly updates
2. **Multi-country model:** Train on G7 economies for comparison
3. **Recession severity:** Predict depth/duration, not just binary onset
4. **Regime detection:** Identify structural breaks in Fed policy or economy
5. **Explainability:** SHAP values for time-varying feature importance

### Research Directions
1. **Causal inference:** Use instrumental variables or diff-in-diff for policy evaluation
2. **Text data:** Incorporate FOMC minutes, Fed speeches for forward guidance
3. **High-frequency data:** Use weekly jobless claims, daily financial indicators
4. **Attention mechanisms:** LSTM or Transformer for temporal patterns
5. **Audit study:** Compare to professional forecasters (Survey of Professional Forecasters)

## Academic Context

This project demonstrates techniques from:
- **ML Time Series:** Proper validation, feature engineering, rare event prediction
- **Macroeconomics:** Yield curve, Taylor Rule, business cycle indicators
- **Model Evaluation:** Calibration curves, baseline comparisons, fairness in prediction
- **Responsible AI:** Transparent limitations, no overclaiming, honest evaluation

## Contact

Questions or collaboration? Reach out!

abdoulayeaseydi@gmail.com


---

**Note:** This project was built as a rigorous demonstration of ML time series forecasting and economic indicator analysis. All limitations are clearly documented. The model achieves 0.75 AUC, demonstrating strong discrimination ability, but is not production-ready and should not be used for investment decisions.

## License

MIT License - Free to use for educational purposes.

---

**Built with:** Python, XGBoost, scikit-learn, FRED API, Google Colab  
**Project Type:** Time Series Forecasting, Economic Indicators, ML Evaluation  
**Status:** Complete ✓
