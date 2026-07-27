#!/usr/bin/env python
"""Flax GPT-style transformer regressor for windowed price sequences.

The model accepts inputs shaped as [batch, window_size] and predicts a single
next value for each window.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from prepare_dataset import Dataset, load_data


def causal_mask(seq_len: int) -> jnp.ndarray:
    """
    Generate a causal mask for self-attention.

    Parameters:
        seq_len (int): The sequence length.

    Returns:
        jnp.ndarray: A 4D boolean mask array shaped [1, 1, seq_len, seq_len].
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        Apply a transformer block with multi-head attention and feed-forward network.

        Parameters:
            x (jnp.ndarray): Input sequence tensor.
            mask (jnp.ndarray): Attention mask tensor.

        Returns:
            jnp.ndarray: The transformed sequence tensor.
        """
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not self.train,
        )(x, x, mask=mask)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.ff_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)
        x = nn.Dense(self.embed_dim)(x)
        x = residual + x
        return x


class GPTStyleRegressor(nn.Module):
    """GPT-style transformer for scalar sequence regression."""

    window_size: int
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_blocks: int = 4
    dropout_rate: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the GPT-style regressor model on input sequences.

        Parameters:
            x (jnp.ndarray): Input sequence tensor, shaped [batch, window_size] or [window_size].

        Returns:
            jnp.ndarray: Predicted scalar values for each window, shaped [batch] or scalar.
        """
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(
                "Expected inputs shaped as [batch, window_size] or [window_size].")

        if x.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window_size={self.window_size}, got {x.shape[1]}.")

        positions = jnp.arange(self.window_size)[None, :]
        token_projection = nn.Dense(self.embed_dim, name="token_projection")
        position_embedding = nn.Embed(
            num_embeddings=self.window_size, features=self.embed_dim, name="position_embedding")

        x = x[..., None]
        x = token_projection(x) + position_embedding(positions)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)

        mask = causal_mask(self.window_size)
        for block_index in range(self.num_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                train=self.train,
                name=f"transformer_block_{block_index}",
            )(x, mask=mask)

        x = nn.LayerNorm()(x)
        attn_scores = nn.Dense(1, name="pooling_score")(x)
        attn_weights = nn.softmax(attn_scores, axis=1)
        x = jnp.sum(attn_weights * x, axis=1)  # Basic attention pooling
        x = nn.Dense(self.ff_dim)(x)
        x = nn.gelu(x)
        prediction = nn.Dense(1)(x)
        return prediction.squeeze(-1)


@dataclass(frozen=True)
class GPTConfig:
    window_size: int = 10
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_blocks: int = 4
    dropout_rate: float = 0.0


def build_gpt_style_model(config: GPTConfig) -> GPTStyleRegressor:
    """
    Build a GPT-style regressor model from configuration.

    Parameters:
        config (GPTConfig): Configuration parameters for the model.

    Returns:
        GPTStyleRegressor: Instantiated regressor model.
    """
    return GPTStyleRegressor(
        window_size=config.window_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
    )


def initialize_model(
    window_size: int = 10,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 4,
) -> tuple[GPTStyleRegressor, dict, jnp.ndarray]:
    """
    Initialize model parameters and return instance, parameters, and a dummy prediction.

    Parameters:
        window_size (int): Size of input sliding window.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward dimension.
        num_blocks (int): Number of transformer blocks.

    Returns:
        tuple[GPTStyleRegressor, dict, jnp.ndarray]: Model, initialized parameters, and output on dummy input.
    """
    model = build_gpt_style_model(
        GPTConfig(
            window_size=window_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
        )
    )
    # Print model characteristics
    print(f"Model characteristics:")
    print(f"  Window size: {window_size}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Feed-forward dimension: {ff_dim}")
    print(f"  Number of transformer blocks: {num_blocks}")
    print("-"*100)
    print(model)
    print("-"*100)
    dummy_input = jnp.ones((1, window_size), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    output = model.apply(params, dummy_input)
    return model, params, output


def main() -> None:
    """
    Execute sample data loading and prediction using the GPT-style model.
    """
    file_path = "datasets/AAPL.csv"
    date_range = ("2017-01-01", "2022-12-31")
    price_column = "Adj Close"
    window_size = 100

    data = load_data(file_path, date_filter=date_range,
                     price_column=price_column)
    if data is None:
        raise SystemExit(1)

    dataset = Dataset(data=data, window_size=window_size)
    sample = dataset[0]

    model = build_gpt_style_model(GPTConfig(
        window_size=window_size, embed_dim=128, num_heads=4, ff_dim=256, num_blocks=4))
    params = model.init(jax.random.PRNGKey(0), sample["x"][None, :])
    prediction = model.apply(params, sample["x"][None, :])

    print(f"Dataset length: {len(dataset)}")
    print(f"Input window shape: {sample['x'].shape}")
    print(f"Target value: {sample['y']}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction[0]}")


if __name__ == "__main__":
    main()
