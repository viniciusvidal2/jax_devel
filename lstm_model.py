#!/usr/bin/env python
"""Flax LSTM regressor for windowed price sequences.

The model accepts inputs shaped as [batch, window_size] and predicts a single
next value for each window.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from prepare_dataset import Dataset, load_data


class LSTMRegressor(nn.Module):
    """Sequence-to-one regression model built with stacked LSTM cells."""

    hidden_size: int
    num_layers: int = 1
    dropout_rate: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(
                "Expected inputs shaped as [batch, window_size] or [window_size].")

        x = x[..., None]

        for layer_index in range(self.num_layers):
            cell = nn.OptimizedLSTMCell(
                features=self.hidden_size, name=f"lstm_cell_{layer_index}")
            initial_carry = cell.initialize_carry(
                jax.random.PRNGKey(0), (x.shape[0],))

            outputs = []
            carry = initial_carry
            for timestep in range(x.shape[1]):
                carry, y = cell(carry, x[:, timestep, :])
                outputs.append(y)

            x = jnp.stack(outputs, axis=1)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not self.train)

        features = x[:, -1, :]
        features = nn.Dense(self.hidden_size)(features)
        features = nn.relu(features)
        prediction = nn.Dense(1)(features)
        return prediction.squeeze(-1)


@dataclass(frozen=True)
class LSTMConfig:
    window_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.0


def build_lstm_model(config: LSTMConfig) -> LSTMRegressor:
    return LSTMRegressor(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    )


def initialize_model(window_size: int = 10, hidden_size: int = 64, num_layers: int = 2):
    model = build_lstm_model(LSTMConfig(
        window_size=window_size, hidden_size=hidden_size, num_layers=num_layers))
    dummy_input = jnp.ones((1, window_size), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    output = model.apply(params, dummy_input)
    return model, params, output


def main() -> None:
    file_path = "AAPL.csv"
    date_range = ("2017-01-01", "2022-12-31")
    price_column = "Adj Close"
    window_size = 10

    data = load_data(file_path, date_filter=date_range,
                     price_column=price_column)
    if data is None:
        raise SystemExit(1)

    dataset = Dataset(data=data, window_size=window_size)
    sample = dataset[0]

    model = build_lstm_model(LSTMConfig(
        window_size=window_size, hidden_size=64, num_layers=2))
    params = model.init(jax.random.PRNGKey(0), sample["x"][None, :])
    prediction = model.apply(params, sample["x"][None, :])

    print(f"Dataset length: {len(dataset)}")
    print(f"Input window shape: {sample['x'].shape}")
    print(f"Target value: {sample['y']}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction[0]}")


if __name__ == "__main__":
    main()
