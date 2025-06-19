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
* **Battle-tested during COVID-19 pandemic** – By using machine learning algorithms and real-time data, `maynard` reacts fast to economic shocks. Our models started showing signs of weakening US growth in April 2020.
* **Proven in production** – `maynard` is already being used in practice. It's pipeline is the forecasting core behind [InsightsNow.app](https://github.com/jowike/InsightsNow) – a live dashboard that shows up-to-date predictions for key macroeconomic indicators. Essentially, it runs in the background as a monthly model factory.

  To get a better feel for how it works:
    - 🎞️ [jump to the demo →](https://www.youtube.com/watch?v=RfoxH-lfU7k) for a guided tour of the tool. It walks through the dashboard and reveals what kinds of questions it can help answer. 
    - 💻  try the live instance → [https://insightsnow.mini.pw.edu.pl](https://insightsnow.mini.pw.edu.pl/pages/dashboard) and explore the dashboard yourself

Together, they show how our package  moves from concept to something that is ready to use — explainable, proven, and grounded in real-world needs.

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

### 📦 1. Install the package

First, install `maynard` in editable mode:

```bash
pip install -e .
```

---

### ⚙️ 2. Configure your run

Control how the pipeline behaves by editing:

* `parameters.yaml` — choose the target variable, forecast dates, and backtest settings
* `catalog.yaml` — tell the pipeline where to find your input data and where to save outputs

You'll find example configs in `/conf/base/`
Sample input data is available in `/data/0_source/`

---

### 🧙🏻‍♂️ 3. Make your first forecast

First, set things up by creating a new project (in this example, we’ll call it MaynardAI — but feel free to use any name you like):

```bash
maynard init MaynardAI
cd MaynardAI
```

Then you're ready to go. 

#### 🧭 Visualize the pipeline

To see how everything connects, launch the interactive blueprint of your data and ML workflows:

```bash
maynard viz
```

---

#### 🚀 Run the full pipeline

To run everything end-to-end — from raw data all the way to the final forecast — just type:

```bash
maynard run
```

---

#### 🎯 Want to run only part of the workflow?

You can totally run just a piece of it — here’s how:

* **Run specific steps only**

  Want to skip the prep and jump straight to model estimation? Here you go:

  ```bash
  maynard run --nodes estimate_ml_models,estimate_arima,estimate_var
  ```

* **Start partway through and let `maynard` take it from there**

  Want to jump in halfway? Start from any node — let it be data transformation — and run the rest:

  ```bash
  maynard run --from-nodes transform_time_series
  ```

> ⚠️ Heads up: you can only use one of `--nodes`, `--from-nodes`, or `--to-nodes` at a time — they don’t work together.

---

### 💡 **Quick tip: dry run**

Want to check what’s going to happen before running the full pipeline?

Run tests on any part of the project using:

```bash
pytest -s tests/  # or run a specific file, e.g.
pytest -s tests/test_run_command_with_nodes.py
```

This will print out detailed logs — super handy for making sure everything’s set up correctly before a full run.


🙈 If anything goes wrong, no worries — we’ve got you . Just let us know – we’re here to help. 

Otherwise: hats off 🎩 — you just pulled off what most real-world forecasters only *aspire* to! 🎉🎊

---

## 🔜 Coming Soon

* PyPI release for `maynard`
* Ready-to-use templates for typical nowcasting setups

---

## 🙋 Need help?

Open an issue or reach out at [https://github.com/jowike/maynard/issues](https://github.com/jowike/maynard/issues)

---
