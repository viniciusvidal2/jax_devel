#!/usr/bin/env python
"""Evaluate a saved GPT-style checkpoint and visualize one-step-ahead tracking.

The script:
- restores the Orbax checkpoint saved by train_gpt_optax.py
- iterates through the dataset in order with prepare_dataset.create_dataloader
- plots predictions against the actual stock curve over time
- shows several error-metric views for quick inspection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from orbax import checkpoint as ocp
import pandas as pd

from configs.config_loader import get_section, load_workspace_config
from gpt_style_model import GPTConfig, build_gpt_style_model
from prepare_dataset import Dataset, create_dataloader, load_dataset_with_dates


@dataclass(frozen=True)
class EvalConfig:
    file_path: str = "datasets/AAPL.csv"
    date_start: str = "2022-01-01"
    date_end: str = "2022-12-31"
    price_column: str = "Adj Close"

    window_size: int = 100
    batch_size: int = 64

    embed_dim: int = 260
    num_heads: int = 20
    ff_dim: int = 512
    num_blocks: int = 6
    dropout_rate: float = 0.4
    checkpoint_path: str = "gpt_model_checkpoint.orbax"

    rolling_window: int = 20
    save_dir: str = "plots"
    show: bool = True


def load_eval_config() -> EvalConfig:
    """
    Load the evaluation configuration from workspace YAML file.

    Returns:
        EvalConfig: Loaded configuration dataclass instance.
    """
    config = load_workspace_config()
    dataset_loading = get_section(config, "dataset_loading")
    gpt_model = get_section(config, "gpt_model")
    evaluation = get_section(config, "evaluation")

    evaluation_date_range = dataset_loading.get(
        "evaluation_date_range", ["2022-01-01", "2022-12-31"])
    if not isinstance(evaluation_date_range, list) or len(evaluation_date_range) != 2:
        raise ValueError(
            "dataset_loading.evaluation_date_range must be a 2-item list.")

    return EvalConfig(
        file_path=dataset_loading.get("file_path", "datasets/AAPL.csv"),
        date_start=evaluation_date_range[0],
        date_end=evaluation_date_range[1],
        price_column=dataset_loading.get("price_column", "Adj Close"),
        window_size=gpt_model.get("window_size", 20),
        batch_size=evaluation.get("batch_size", 64),
        embed_dim=gpt_model.get("embed_dim", 260),
        num_heads=gpt_model.get("num_heads", 20),
        ff_dim=gpt_model.get("ff_dim", 512),
        num_blocks=gpt_model.get("num_blocks", 6),
        dropout_rate=gpt_model.get("dropout_rate", 0.4),
        rolling_window=evaluation.get("rolling_window", 20),
        save_dir=evaluation.get("save_dir", "plots"),
        show=evaluation.get("show", True),
        checkpoint_path=evaluation.get(
            "checkpoint_path", "gpt_model_checkpoint.orbax"),
    )


def load_restored_params(config: EvalConfig) -> tuple[GPTStyleRegressor, dict]:
    """
    Build the GPT model and restore its parameters from the saved checkpoint.

    Parameters:
        config (EvalConfig): Evaluation configuration parameters.

    Returns:
        tuple[GPTStyleRegressor, dict]: Model instance and its restored parameters dictionary.
    """
    model = build_gpt_style_model(
        GPTConfig(
            window_size=config.window_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
        )
    )

    checkpoint_path = Path.cwd() / config.checkpoint_path
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(checkpoint_path)
    
    return model, restored_params


def recover_real_scale(values: jnp.ndarray, x_min: jnp.ndarray, x_max: jnp.ndarray) -> jnp.ndarray:
    """
    Invert the per-window normalization used by Dataset.__getitem__.

    Parameters:
        values (jnp.ndarray): Normalized prediction or actual values.
        x_min (jnp.ndarray): The minimum value of the corresponding window.
        x_max (jnp.ndarray): The maximum value of the corresponding window.

    Returns:
        jnp.ndarray: Values scaled back to the original range.
    """
    scale = x_max - x_min + 1e-8
    return values * scale + x_min


def run_ordered_predictions(config: EvalConfig) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Execute predictions on the dataset sequentially and map them back to dates.

    Parameters:
        config (EvalConfig): Evaluation configuration parameters.

    Returns:
        tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]: target_dates, actual_values, predicted_values.
    """
    loaded = load_dataset_with_dates(
        file_path=config.file_path,
        date_filter=(config.date_start, config.date_end),
        price_column=config.price_column,
    )
    if loaded is None:
        raise SystemExit(1)

    dates, values = loaded
    dataset = Dataset(data=values, window_size=config.window_size)
    target_dates = pd.to_datetime(dates[config.window_size:])

    model, params = load_restored_params(config)

    loader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=0,
        num_epochs=1,
        drop_last=False,
    )

    predictions = []
    actuals = []

    for batch in loader:
        x_batch = jnp.asarray(batch["x"], dtype=jnp.float32)
        y_batch = jnp.asarray(batch["y"], dtype=jnp.float32)
        x_min = jnp.asarray(batch["x_min"], dtype=jnp.float32)
        x_max = jnp.asarray(batch["x_max"], dtype=jnp.float32)
        batch_predictions = model.apply({"params": params}, x_batch)

        predicted_real = recover_real_scale(batch_predictions, x_min, x_max)
        actual_real = recover_real_scale(y_batch, x_min, x_max)

        predictions.append(np.asarray(jax.device_get(predicted_real)))
        actuals.append(np.asarray(jax.device_get(actual_real)))

    predicted_values = np.concatenate(predictions, axis=0)
    actual_values = np.concatenate(actuals, axis=0)

    if len(predicted_values) != len(target_dates):
        raise RuntimeError(
            "Prediction count does not match target date count: "
            f"{len(predicted_values)} vs {len(target_dates)}"
        )

    return target_dates, actual_values, predicted_values


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """
    Calculate evaluation metrics comparing actual and predicted values.

    Parameters:
        actual (np.ndarray): Real target values.
        predicted (np.ndarray): Predicted values.

    Returns:
        dict[str, float]: Dictionary of metric names mapping to their calculated values.
    """
    error = predicted - actual
    abs_error = np.abs(error)
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    median_ae = float(np.median(abs_error))
    max_ae = float(np.max(abs_error))

    epsilon = 1e-8
    mape = float(
        np.mean(abs_error / np.maximum(np.abs(actual), epsilon)) * 100.0)

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "max_ae": max_ae,
        "mape": mape,
        "r2": r2,
    }


def plot_results(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: dict[str, float],
    config: EvalConfig,
) -> None:
    """
    Plot actual vs predicted price curves, errors over time, and residual distributions.

    Parameters:
        dates (pd.DatetimeIndex): Target dates for plotting.
        actual (np.ndarray): Actual stock prices.
        predicted (np.ndarray): Predicted stock prices.
        metrics (dict[str, float]): Computed metrics to display in the plot.
        config (EvalConfig): Evaluation configuration parameters.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    error = predicted - actual
    abs_error = np.abs(error)
    rolling_mae = pd.Series(abs_error).rolling(
        window=max(1, config.rolling_window),
        min_periods=1,
    ).mean()

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    fig.suptitle("GPT-style one-step forecast evaluation",
                 fontsize=16, fontweight="bold")

    axes[0, 0].plot(dates, actual, label="Actual",
                    linewidth=2.0, color="#1f77b4")
    axes[0, 0].plot(dates, predicted, label="Predicted",
                    linewidth=1.8, color="#d62728", alpha=0.9)
    axes[0, 0].set_title("Actual vs predicted price curve")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel(config.price_column)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(dates, abs_error, label="Absolute error",
                    color="#ff7f0e", alpha=0.7)
    axes[0, 1].plot(dates, rolling_mae,
                    label=f"Rolling MAE ({config.rolling_window})", color="#2ca02c", linewidth=2.0)
    axes[0, 1].set_title("Error over time")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Error")
    axes[0, 1].legend(loc="best")

    axes[1, 0].hist(error, bins=35, color="#9467bd",
                    alpha=0.85, edgecolor="white")
    axes[1, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Residual distribution")
    axes[1, 0].set_xlabel("Prediction - actual")
    axes[1, 0].set_ylabel("Count")
    metric_text = "\n".join(
        [
            f"RMSE: {metrics['rmse']:.4f}",
            f"MAE: {metrics['mae']:.4f}",
            f"Median AE: {metrics['median_ae']:.4f}",
            f"Max AE: {metrics['max_ae']:.4f}",
            f"MAPE: {metrics['mape']:.2f}%",
            f"R²: {metrics['r2']:.4f}",
        ]
    )
    axes[1, 0].text(
        0.98,
        0.98,
        metric_text,
        transform=axes[1, 0].transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4",
              "facecolor": "white", "alpha": 0.85},
    )

    min_val = min(float(np.min(actual)), float(np.min(predicted)))
    max_val = max(float(np.max(actual)), float(np.max(predicted)))
    axes[1, 1].scatter(actual, predicted, s=14, alpha=0.6,
                       color="#17becf", edgecolors="none")
    axes[1, 1].plot([min_val, max_val], [min_val, max_val],
                    linestyle="--", color="black", linewidth=1.2)
    axes[1, 1].set_title("Predicted vs actual")
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")
    axes[1, 1].set_aspect("equal", adjustable="box")

    output_path = save_dir / "gpt_checkpoint_evaluation.png"
    fig.savefig(output_path, dpi=180)
    print(f"Saved figure to {output_path}")

    plt.figure(figsize=(18, 5))
    plt.plot(dates, actual, label="Actual", linewidth=2.1, color="#1f77b4")
    plt.plot(dates, predicted, label="Predicted",
             linewidth=1.6, color="#d62728", alpha=0.9)
    plt.title("One-step-ahead tracking across the full date range")
    plt.xlabel("Date")
    plt.ylabel(config.price_column)
    plt.legend(loc="best")
    plt.tight_layout()
    series_path = save_dir / "gpt_checkpoint_series.png"
    plt.savefig(series_path, dpi=180)
    print(f"Saved figure to {series_path}")

    if config.show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    """
    Main entry point to run ordered predictions, compute metrics, and plot results.
    """
    config = load_eval_config()
    dates, actual, predicted = run_ordered_predictions(config)
    metrics = compute_metrics(actual, predicted)

    print("Evaluation metrics:")
    for name, value in metrics.items():
        if name == "mape":
            print(f"  {name}: {value:.2f}%")
        else:
            print(f"  {name}: {value:.6f}")

    plot_results(dates, actual, predicted, metrics, config)


if __name__ == "__main__":
    main()
