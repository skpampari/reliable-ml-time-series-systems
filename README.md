# Adaptive & Explainable Machine Learning for Time-Dependent Decision Systems

## Overview
This repository provides a reproducible machine learning pipeline for time-dependent data, with a focus on model reliability, explainability, and decision-aware evaluation.

While initially developed as an LSTM-based forecasting system, the project has evolved into a research-oriented framework for studying how machine learning models behave under temporal constraints, distribution shifts, and real-world deployment conditions.

The goal is not only predictive performance, but also:
- understanding model behavior over time
- evaluating reliability under changing data distributions
- enabling explainable and decision-aware modeling

This work is positioned toward applications in real-world systems such as transportation, infrastructure, and other high-stakes environments where robustness and interpretability are critical.


## Why This Matters (Research Context)

Modern machine learning systems often achieve high accuracy in static settings but fail in real-world environments due to:
- evolving data distributions (concept drift)
- lack of interpretability
- absence of uncertainty awareness

This project investigates how to build machine learning systems that are:
- reliable over time
- explainable in their predictions
- adaptable to changing environments

These challenges are critical in domains such as transportation safety, operations, and infrastructure systems, where decisions must be both accurate and interpretable.

## Systems for AI Relevance
This project aligns with systems-for-AI research because it focuses on building reproducible and reliable ML pipelines, including:
- modular pipeline design (data, training, evaluation)
- leakage control and correct evaluation
- reproducibility controls (fixed seeds, deterministic execution)
- baseline comparisons and rigorous validation

These system-level practices are critical for scalable AI systems, which require reliability, reproducibility, and correct evaluation in real-world deployments.

## Research Scope & Intent
This project intentionally focuses on **univariate time series forecasting** (using closing prices only)
to study temporal dependency modeling with LSTM networks under realistic evaluation constraints.

## Research Direction (Ongoing Work)

This project is being extended toward:

- Explainable AI (XAI): Understanding model predictions using feature attribution and interpretability methods
- Uncertainty Modeling: Estimating confidence in predictions using probabilistic and approximate Bayesian techniques
- Adaptive Systems: Studying how models behave under distribution shifts and evolving data
- Decision-Aware Evaluation: Moving beyond RMSE to evaluate real-world decision impact

The long-term goal is to develop adaptive machine learning systems that can support reliable decision-making in dynamic environments.

The goal is not to propose a novel deep learning architecture, but to:
- build a leakage-free and reproducible forecasting pipeline,
- evaluate predictive performance using RMSE,
- establish a baseline for future extensions such as:
  - feature enrichment (volume, technical indicators),
  - baseline model comparisons,
  - uncertainty-aware forecasting,
  - decision-oriented evaluation for financial analytics.


## Methodology
### Data Collection and Preprocessing
- Data Source: CSV file `data/sp500.csv` containing historical S&P 500 data with 'Date' and 'Close' columns.
- Preprocessing Steps:
  - Filter to 'Close' prices.
  - Scale data to [0,1] using MinMaxScaler.
  - Create sequences: 60-day window to predict the next price.
- Challenges Addressed: Data variability (missing values, outliers, fluctuations) via scaling and sequence preparation; Hyperparameter tuning for accuracy; Overfitting/underfitting risks.

### Model Architecture
- LSTM Network: Stacked layers (50 units each), Dense output for regression.
- Optimizer: Adam; Loss: Mean Squared Error (MSE).
- Training: 5 epochs (expandable), batch size 32, on 80% train split.

### Evaluation
- Metrics:
  - Root Mean Squared Error (RMSE)
  - Directional Accuracy (planned)
  - Decision-aware evaluation metrics (planned)
  - Final Evaluation: Assess accuracy on test set, compare predictions with actual data, identify strengths/weaknesses (e.g., trend capture but volatility issues).

### Key Notes
- Leakage-free scaling (scaler fit on training data only).
- Time-ordered split (no shuffling).
- Fixed seeds for reproducibility.
- Outputs saved to `results/plots/`.
  
## Limitations

- The current implementation uses a univariate input (closing prices only).
- Only simple baseline models (naive forecast) are included; stronger statistical baselines (e.g., moving average, ARIMA) are not yet evaluated..
- Market regime shifts and exogenous variables are not explicitly modeled.
These limitations are intentional and will be addressed in future iterations.
    
## Next Planned Improvements (Systems & AI)
- Add baseline models (naive forecast, moving average, ARIMA)
- Add explainability methods (e.g., SHAP, feature importance analysis)
- Add walk-forward validation (rolling window) for robust evaluation
- Add uncertainty estimation (e.g., Monte Carlo dropout)
- Add decision-oriented evaluation (directional accuracy, cost-aware metrics)
- Add pipeline automation and reproducible deployment (Docker, CI)
- Add benchmarking and performance analysis for training and inference
- Future work includes Bayesian approaches for uncertainty estimation in time-series predictions.


## Installation
1. Clone the repo: `git clone https://github.com/sk/advanced-stock-price-forecasting-lstm.git`
2. Install dependencies: `pip install -r requirements.txt`
   - Contents: numpy, pandas, matplotlib, scikit-learn, tensorflow
3. See `requirements.txt` for pinned dependencies.
     
## Reproducibility Checklist
- Fixed random seeds (Python, NumPy, TensorFlow)
- Train/test split = 80/20 (time-ordered)
- Scaler fit on training portion only (leakage-free)
- Results saved to `results/plots/`

## Usage / Execution Steps
1. **Run Notebook**: Open `notebooks/stock-market-analysis-prediction-using-lstm.ipynb` in Jupyter: `jupyter notebook notebooks/stock-market-analysis-prediction-using-lstm.ipynb`—executes full tutorial (data fetch, EDA, LSTM training, prediction).
2. **Run Scripts**: For modular run:
   - `python src/preprocess.py` (fetches/preprocesses data).
   - `python src/model.py` (builds/trains LSTM).
   - `python src/evaluate.py` (predicts/evaluates/plots).
   - `python src/main.py` (full pipeline).
3. **Outputs**: See `results/` for predictions.csv and plots/actual_vs_predicted.png.

## Results
- Sample RMSE:
| Setting | Value |
|---|---|
| Seed | 42 |
| Split | 80/20 time-ordered |
| Sequence length | 60 |
| Epochs | 5 |
| Batch size | 32 |
| Test RMSE | ~84 |

  Note: RMSE is reported in the original price scale after inverse transformation. Because the S&P 500 index spans a wide numeric range over decades, this metric primarily reflects trend-level
  accuracy; future work will include normalized errors and baseline comparisons for improved interpretability.

- Visualization: Actual vs. predicted curves showing trend alignment.
- Interpretation: RMSE indicates how well the model predicts future S&P 500 prices; lower RMSE means better accuracy.
- This next-step prediction is shown for demonstration and is not a trading strategy.
- Baseline comparison:
  - Naive (t−1) RMSE: reported alongside LSTM for reference.


## References
- Fischer, T., & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions.* European Journal of Operational Research, 270(2), 654–669.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.
- Bao, J., Yue, J., & Rao, L. (2017). *A deep learning framework for financial time series using stacked autoencoders and LSTM networks.* PLoS ONE, 12(7), e0180944.

## License
MIT License.

## Contact
Sai Kumar Pampari - skpampari2022@gmail.com - Open to collaborations.



