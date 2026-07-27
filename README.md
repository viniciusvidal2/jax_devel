# JAX-Devel: Time Series Prediction & Architecture Evaluation

This repository contains JAX and Flax-based implementations of neural architectures designed for time series forecasting, specifically sequence-to-one regression on stock price data (e.g., Apple stock, AAPL).

## Repository Overview

- **[prepare_dataset.py](file:///home/vini/Desktop/jax_devel/prepare_dataset.py)**: Handles data loading, date filtering, min-max windowed normalization, and builds data loaders for training.
- **[gpt_style_model.py](file:///home/vini/Desktop/jax_devel/gpt_style_model.py)**: Implements a custom Flax GPT-style transformer regressor that processes sequences and uses masked self-attention pooling.
- **[lstm_model.py](file:///home/vini/Desktop/jax_devel/lstm_model.py)**: Implements a stacked LSTM sequence-to-one regressor.
- **[train_gpt_optax.py](file:///home/vini/Desktop/jax_devel/train_gpt_optax.py)**: Code to train the GPT regressor with Adam optimizer (Optax), cosine learning rate decay, and loss-stability early stopping.
- **[evaluate_gpt_optax.py](file:///home/vini/Desktop/jax_devel/evaluate_gpt_optax.py)**: Script to restore saved checkpoints (Orbax), run sequential evaluation on a test window, calculate performance metrics, and generate diagnostic plots.
- **[configs/config_loader.py](file:///home/vini/Desktop/jax_devel/configs/config_loader.py)**: Utility to load shared workspace YAML settings from `configs/workspace_config.yaml`.
- **[references/](file:///home/vini/Desktop/jax_devel/references/)**: Contains educational reference notebooks translated to Python scripts, outlining LLM architectures (`L2.py`) and data loaders using Google Grain (`L3.py`).

---

## Method: Time Series Prediction

To predict future values in a sequence (e.g., predicting the next day's price given a sliding window of historical prices):
1. **Window-based Min-Max Normalization**: Each input sequence window of length $W$ is extracted and normalized to the range $[0, 1]$ using the formula:
   $$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min} + 1e-8}$$
   This scale-invariant representation ensures the model focuses on local trends/patterns rather than absolute values.
2. **Target Normalization**: The target value (the step immediately following the window) is normalized using the same window minimum and maximum.
3. **Sequence-to-One Architecture**:
   - **GPT-style Regressor**: Embeds window positions and projects scalars into high-dimensional tokens. Applies causal masked self-attention blocks to allow tokens to attend only to current and past timesteps, followed by basic attention pooling and dense projection to a single scalar prediction.
   - **LSTM Regressor**: Feeds the sequence step-by-step through stacked `OptimizedLSTMCell`s and projects the final hidden state into a single scalar prediction.
4. **Denormalization**: Predictions are converted back to the original scale during evaluation using:
   $$x_{real} = x_{normalized} \times (x_{max} - x_{min} + 1e-8) + x_{min}$$

---

## Architectural Capabilities & Impact Evaluation

### GPT-style Attention vs. Stacked LSTMs
- **Multi-Head Attention**: Captures long-range dependencies efficiently and handles complex inter-step relationships through parallel attention channels, though it lacks direct sequence ordering without positional embeddings.
- **LSTM Recurrence**: Naturally sequential and keeps state over time, but is susceptible to forgetting long historical context and cannot be parallelized as easily during training.

### Quantitative Evaluation Metrics
We study the quality of predictions using:
- **RMSE** (Root Mean Squared Error) & **MSE** (Mean Squared Error): Measures prediction variance and heavily penalizes large errors.
- **MAE** (Mean Absolute Error) & **Median AE**: Offers a robust indicator of typical prediction error.
- **MAPE** (Mean Absolute Percentage Error): Evaluates error relative to the target scale.
- **R²** (Coefficient of Determination): Measures how much variance in the price curve is explained by the model.

---

## Installation & Setup

We recommend using Python 3.12 and setting up the environment using Conda.

### Option 1: Using Conda (Recommended)

1. Make sure you have Anaconda or Miniconda installed.
2. Create and activate the conda environment using the [environment.yml](file:///home/vini/Desktop/jax_devel/environment.yml) file:
   ```bash
   conda env create -f environment.yml
   conda activate jax
   ```

### Option 2: Using Pip

If you prefer using standard virtual environments:
1. Create and activate your virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install the required dependencies from [requirements.txt](file:///home/vini/Desktop/jax_devel/requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
