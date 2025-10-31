#  StaySharpe — A Quantitative Trading Model for Optimal Buy/Sell Decisions

> ⚡ *Developed for the “Stay Sharpe” Quant Trading Hackathon*
> 🏆 *Awarded 2nd Place for performance on unseen test data.*

---

##  Overview

**StaySharpe** is a quantitative trading model that predicts whether to **Buy (1)**, **Hold (0)**, or **Sell (-1)** an asset based on its short-term price movements.
The project combines **technical indicators**, **machine learning**, and **risk-adjusted evaluation** to maximize returns while minimizing drawdown and overfitting.

---

## Problem Statement

Traditional price-based trading strategies often rely on fixed technical rules that fail under varying market regimes.
The goal of this project was to create an **adaptive ML-based agent** that:

* Learns patterns from OHLCV (Open-High-Low-Close-Volume) data
* Predicts optimal trade actions (Buy/Hold/Sell)
* Maximizes **Sharpe ratio** and minimizes **maximum drawdown**
* Avoids **forward bias** and **overfitting**

---

##  Dataset

* **Input Format:** OHLCV time-series data (1 unit = 1 trading timestep)
* **Data Used:** Historical stock price data provided for the hackathon
* **Structure:**

  | Column | Description                                  |
  | :----- | :------------------------------------------- |
  | open   | Opening price of the asset                   |
  | high   | Highest price of the asset during the period |
  | low    | Lowest price of the asset during the period  |
  | close  | Closing price of the asset                   |
  | volume | Traded volume during the period              |

---

## ⚙️ Model Architecture

The strategy uses a **Random Forest Classifier** trained on technical indicators computed from OHLCV data.

###  Key Features Extracted

* **RSI (Relative Strength Index)** — momentum indicator
* **MACD (Moving Average Convergence Divergence)** — trend strength
* **Bollinger Bands** — volatility measure
* **OBV (On-Balance Volume)** — volume-based flow confirmation

All indicators are computed using the `pandas_ta` library.

---

##  Labeling Logic

Each sample is labeled using **forward returns** over a lookahead period `h`:

[
return = \frac{close_{t+h}}{close_t} - 1
]

* **Buy (1):** if return > `upper_threshold`
* **Sell (-1):** if return < `lower_threshold`
* **Hold (0):** otherwise

This ensures the model’s outputs are directly aligned with profit-based thresholds.

---

##  Training Setup

To prevent data leakage and overfitting, the dataset was split **chronologically**:

| Split      | Portion | Purpose                 |
| :--------- | :------ | :---------------------- |
| Train      | 64%     | Model training          |
| Validation | 16%     | Hyperparameter tuning   |
| Test       | 20%     | Final unseen evaluation |

###  Hyperparameter Optimization

Grid Search was performed over:

* Lookahead horizon `h ∈ {10, 30, 50}`
* Threshold `∈ {0.002, 0.006, 0.010}` (coarse)
* Then refined with `h ∈ {20, 30, 40}`, threshold `∈ {0.001, 0.002, 0.003}` (fine)

The best combination achieved the highest **Sharpe ratio** and lowest **max drawdown** on the validation set.

---

## 🏆 Final Model Parameters

| Parameter         | Value                  | Description                   |
| :---------------- | :--------------------- | :---------------------------- |
| Model             | RandomForestClassifier | Robust non-linear classifier  |
| `h`               | 20                     | Lookahead period for labeling |
| `upper_threshold` | 0.003                  | Buy signal threshold          |
| `lower_threshold` | -0.003                 | Sell signal threshold         |
| `execution_delay` | 1                      | Avoids forward bias           |
| Features          | RSI, MACD, BBANDS, OBV | Derived via `pandas_ta`       |
| Validation Metric | Sharpe Ratio           | Risk-adjusted return measure  |

---

##  Evaluation Metrics

* **Sharpe Ratio:** Measures risk-adjusted performance
* **Max Drawdown:** Largest equity loss from peak to trough
* **Signal Distribution:** Ensures balanced buy/hold/sell activity
* **Classification Report:** For alignment between predicted and actual market direction

---

##  How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/StaySharpe.git
   cd StaySharpe
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook

   ```bash
   jupyter notebook Training.ipynb
   ```

4. Run all cells, the notebook will train the model, tune hyperparameters, and backtest the final strategy.

---

##  Acknowledgements

* **Event:** *Stay Sharpe Quant Hackathon*
* **Organized by:** Ctrl+Alpha : The Official Finance Group of the International Institute of Information Technology, Hyderabad (IIIT-H).
* **Achievement:** *2nd Place - Best Performing Model on Unseen Dataset*

---

## 📬 Connect

👤 **Avnish Uba**
🔗 [LinkedIn Profile](https://linkedin.com/in/AvnishUba)


