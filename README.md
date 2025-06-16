# ✨ maynard

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

**Want to know what will happen to the economy next week, next month, or next quarter?**  
``maynard`` helps you stay ahead — it takes raw macroeconomic data and turns it into clear, actionable nowcasts – available in minutes, not weeks

Based on years of experience in machine learning and econometrics, we’ve build `maynard` — a modern, high-frequency forecasting tool for the XXI century economy

> Named after John Maynard Keynes, an economist who revolutionized macroeconomics in XX century. Maynard AI aims at revolutionizing economic forecasting for the modern era.

---

## 🚀 Why use maynard?

* **Real-time Insights** – Instant prediction updates driven by economic news, based on machine learning + econometrics. 
* **Explainable AI along the way (global and local)** – Aligned with responsible AI principles, the tool eliminates the ‘black box’ effect — not only tells you what’s gonna happen, but also provides insights into the why behind the predictions. 
* **Battle-tested during COVID-19 pandemic** – By using machine learning algorithms and real-time data, `maynard` reacts fast to economic shocks. Our models started showing signs of weakening US growth in mid-March 2020.
* **Proven in production** – it's the monthly model factory behind [InsightsNow.app](https://insightsnow.mini.pw.edu.pl/pages/dashboard)

  Ready to see it in action?
    - 🎞️ [Jump to the demo →](https://www.youtube.com/watch?v=RfoxH-lfU7k) for tour of the tool
    - 💻  Try the live instance → [https://insightsnow.mini.pw.edu.pl](https://insightsnow.mini.pw.edu.pl/pages/dashboard) and explore the dashboard yourself

## 🤓 Some technical details for the geek in you

The package is built around a [Kedro](https://kedro.org)-based pipeline and wrapped as a Python package.
It handles everything — from raw data to cleaned inputs, model training, and forecast outputs — with full control over each step.

### 🧱 Architecture Overview

```text
+---------------------+
|   Input data        |  ← CSV, Excel, FRED exports
+---------------------+
         |
         v
+---------------------+
|  maynard pipeline   |  ← Kedro nodes: clean → select → model → explain
+---------------------+
         |
         v
+---------------------+
|  Excel Report       |  ← Results ready for dashboard or manual review
+---------------------+
```

### 🔮 What’s under the hood?

`maynard` is designed for impact. This nowcasting tool is about information efficiency — using just the right amount of data to deliver fast, reliable, and explainable forecasts. Here’s what it actually does:
* **📅 Uses real-time-like data**  
    It Replicates the actual real time data vintage -- the data as it available at each forecast origin date – thus permits sidestepping the inherent risk of information leak;
* **⏳ Tackles messy and mixed-frequency data**  
    Things like monthly predictors for quarterly GDP, or missing values due to publication lags. These are treated as what they really are: missing data problems, and `maynard` solves them smartly. It applies growth rates, log changes, or differences — whatever best captures the signal — and uses splines to fill gaps.
* **🎯 Keeps only the useful variables**  
    `maynard` picks out a small set of features that matter most — so it's models are easier to understand and more stable.  
    Fewer variables mean fewer parameters, less overfitting, and better generalization. Feature selection, combined with statistical diagnostics — tests for stationarity and variance — helps strike a balance between oversimplification (which may result in model misspecification) and excessive complexity (which may lead to instability).

  In essence, `maynard` reduces dimensionality much like a well-designed map simplifies a landscape: not so much as to lose crucial details, but enough to reveal essential features. 
* **⚖️ Gives each variable a smart weight**  
    Each new incoming data point — whether it is a volatile survey result or solid disposable income data — is assigned a specific weight based on its historical reliability and informational value. These weights are inferred by machine learning models and are reflected by model coefficients and Shapley values. As the data changes both model’s parameters and Shapley values are also revised. These dynamically updated weights reflect both the timeliness and noisiness of each indicator. 
* **📊 Shows how accurate the forecast is**  
    Every forecast comes with accuracy metrics like R², RMSE, and MAPE — so you can judge how the model performs over time. It also provides confidence bands based on backtesting errors, showing the likely range of outcomes and benchmark predictions of baseline models (ARIMA and VAR).
* **🧮 Tells you what’s driving the forecast**  
    You don’t just get a number — you get context. Local explanations (for each forecast) and global ones (across time) tell you which features are important and what are the underlying reasons for each forecast. No more black boxes — people want to know why
---

## 🏁 Getting Started

#### 📦 Install the package (in editable mode)

```bash
pip install -e .
```

---

#### ⚙️ Set things up

You can control how the pipeline works by editing two config files:

* `parameters.yaml` — sets the target variable, forecast/reference dates, and backtest range
* `catalog.yaml` — defines data sources and where outputs are saved

You’ll find example configs in `/conf/base/`
Sample input data is available in `/data/0_source/`

---

#### 🧙🏻‍♂️ Run the pipeline

**Run the full pipeline from start to finish:**

```bash
maynard run
```

**Run just one step (e.g. transform time series):**

```bash
maynard run --from-nodes transform_time_series_node
```

**Run a selected part of the pipeline (a few steps in a row):**

```bash
maynard run --from-nodes transform_time_series_node --to-nodes estimate_ml_models_node,estimate_arima_node,estimate_var_node
```

---

#### 🎨 Visualize the pipeline

To explore how everything connects:

```bash
maynard viz
```

---

## 🫟 Coming Soon

* PyPI release for `maynard`
* Ready-to-use templates for typical nowcasting setups

---

## 🙋 Need help?

Open an issue or reach out at [https://github.com/jowike/maynard/issues](https://github.com/jowike/maynard/issues)

---
