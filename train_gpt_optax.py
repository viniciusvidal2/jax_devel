#!/usr/bin/env python
"""Train GPT-style regressor with Optax Adam + cosine decay.

This script uses:
- dataset utilities from prepare_dataset.py
- model definitions from gpt_style_model.py

It supports stability-based early stopping on a moving loss average.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint as ocp

from config_loader import get_section, load_workspace_config
from gpt_style_model import GPTConfig, build_gpt_style_model
from prepare_dataset import Dataset, create_dataloader, load_data


@dataclass(frozen=True)
class TrainConfig:
    file_path: str = "AAPL.csv"
    date_start: str = "1990-01-01"
    date_end: str = "2021-12-31"
    price_column: str = "Adj Close"

    window_size: int = 20
    batch_size: int = 64

    embed_dim: int = 260
    num_heads: int = 20
    ff_dim: int = 512
    num_blocks: int = 4
    dropout_rate: float = 0.4

    max_steps: int = 15_000
    log_every: int = 150
    seed: int = 42

    learning_rate: float = 1e-3
    min_lr_scale: float = 0.1
    decay_steps: int = 5_000
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Stability/plateau stopping.
    stability_window: int = 100
    min_steps_before_stability: int = 500
    min_relative_improvement: float = 1e-3
    plateau_patience_windows: int = 5
    checkpoint_path: str = "gpt_model_checkpoint.orbax"


def load_train_config() -> TrainConfig:
    config = load_workspace_config()
    dataset_loading = get_section(config, "dataset_loading")
    gpt_model = get_section(config, "gpt_model")
    training = get_section(config, "training")

    training_date_range = dataset_loading.get(
        "training_date_range", ["1990-01-01", "2021-12-31"])
    if not isinstance(training_date_range, list) or len(training_date_range) != 2:
        raise ValueError(
            "dataset_loading.training_date_range must be a 2-item list.")

    return TrainConfig(
        file_path=dataset_loading.get("file_path", "AAPL.csv"),
        date_start=training_date_range[0],
        date_end=training_date_range[1],
        price_column=dataset_loading.get("price_column", "Adj Close"),
        window_size=gpt_model.get("window_size", 20),
        batch_size=training.get("batch_size", 64),
        embed_dim=gpt_model.get("embed_dim", 260),
        num_heads=gpt_model.get("num_heads", 20),
        ff_dim=gpt_model.get("ff_dim", 512),
        num_blocks=gpt_model.get("num_blocks", 6),
        dropout_rate=gpt_model.get("dropout_rate", 0.4),
        max_steps=training.get("max_steps", 15_000),
        log_every=training.get("log_every", 150),
        seed=training.get("seed", 42),
        learning_rate=training.get("learning_rate", 1e-3),
        min_lr_scale=training.get("min_lr_scale", 0.1),
        decay_steps=training.get("decay_steps", 5_000),
        adam_b1=training.get("adam_b1", 0.9),
        adam_b2=training.get("adam_b2", 0.999),
        adam_eps=training.get("adam_eps", 1e-8),
        grad_clip_norm=training.get("grad_clip_norm", 1.0),
        stability_window=training.get("stability_window", 100),
        min_steps_before_stability=training.get(
            "min_steps_before_stability", 500),
        min_relative_improvement=training.get(
            "min_relative_improvement", 1e-3),
        plateau_patience_windows=training.get("plateau_patience_windows", 5),
        checkpoint_path=training.get(
            "checkpoint_path", "gpt_model_checkpoint.orbax"),
    )


def make_optimizer(config: TrainConfig) -> tuple[optax.GradientTransformation, optax.Schedule]:
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(1, config.decay_steps),
        alpha=config.min_lr_scale,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(
            learning_rate=lr_schedule,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
        ),
    )
    return optimizer, lr_schedule


def should_stop_for_stability(
    loss_history: deque[float],
    best_window_loss: float,
    no_improve_windows: int,
    config: TrainConfig,
) -> tuple[bool, float, int, float]:
    current_window_loss = float(sum(loss_history) / len(loss_history))
    relative_improvement = (best_window_loss - current_window_loss) / max(
        abs(best_window_loss), 1e-8
    )

    if current_window_loss < best_window_loss:
        best_window_loss = current_window_loss

    if relative_improvement < config.min_relative_improvement:
        no_improve_windows += 1
    else:
        no_improve_windows = 0

    should_stop = no_improve_windows >= config.plateau_patience_windows
    return should_stop, best_window_loss, no_improve_windows, current_window_loss


def train(config: TrainConfig) -> None:
    values = load_data(
        file_path=config.file_path,
        date_filter=(config.date_start, config.date_end),
        price_column=config.price_column,
    )
    if values is None:
        raise SystemExit(1)

    dataset = Dataset(data=values, window_size=config.window_size)
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

    init_x = jnp.ones((1, config.window_size), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(config.seed), init_x)
    params = variables["params"]

    optimizer, lr_schedule = make_optimizer(config)
    opt_state = optimizer.init(params)

    @jax.jit(device=jax.devices("cpu")[0])
    def train_step(
        current_params: dict,
        current_opt_state: optax.OptState,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, jnp.ndarray]:
        def loss_fn(p: dict) -> jnp.ndarray:
            preds = model.apply({"params": p}, x_batch)
            return jnp.mean((preds - y_batch) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    step = 0
    loss_history: deque[float] = deque(maxlen=config.stability_window)
    best_window_loss = float("inf")
    no_improve_windows = 0

    while step < config.max_steps:
        loader = create_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + step,
            num_epochs=1,
            drop_last=True,
        )

        for batch in loader:
            x_batch = jnp.asarray(batch["x"], dtype=jnp.float32)
            y_batch = jnp.asarray(batch["y"], dtype=jnp.float32)

            params, opt_state, loss = train_step(
                params, opt_state, x_batch, y_batch)

            step += 1
            loss_value = float(loss)
            loss_history.append(loss_value)

            if step % config.log_every == 0:
                current_lr = float(lr_schedule(step))
                print(
                    f"step={step} loss={loss_value:.6f} lr={current_lr:.8f} "
                    f"history_avg={sum(loss_history)/len(loss_history):.6f}"
                )

            if (
                step >= config.min_steps_before_stability
                and len(loss_history) == config.stability_window
            ):
                stop, best_window_loss, no_improve_windows, window_avg = should_stop_for_stability(
                    loss_history,
                    best_window_loss,
                    no_improve_windows,
                    config,
                )
                if stop:
                    print(
                        "Stopping early on stable loss: "
                        f"step={step}, window_avg={window_avg:.6f}, "
                        f"best_window_avg={best_window_loss:.6f}, "
                        f"no_improve_windows={no_improve_windows}"
                    )
                    # Save the model parameters before exiting with Orbax
                    checkpoint_path = Path.cwd() / config.checkpoint_path
                    checkpointer = ocp.StandardCheckpointer()
                    checkpointer.save(checkpoint_path, params, force=True)
                    print(f"Model parameters saved to {checkpoint_path}.")
                    return

            if step >= config.max_steps:
                break

    print(f"Reached max_steps={config.max_steps}.")


def main() -> None:
    config = load_train_config()
    train(config)


if __name__ == "__main__":
    main()
