from dataclasses import dataclass

import numpy as np
import pandas as pd
import jax.numpy as jnp


def load_dataset_with_dates(
    file_path: str,
    date_filter: tuple = ('2017-01-01', '2022-12-31'),
    input_columns: list[str] | tuple[str, ...] = ('Adj Close', 'Volume'),
    target_columns: list[str] | tuple[str, ...] = ('Adj Close',),
) -> tuple[list, np.ndarray, np.ndarray] | None:
    """
    Load filtered dates, input features, and target features from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.
        date_filter (tuple): A tuple containing the start and end dates for filtering the DataFrame.
        input_columns (list[str] | tuple[str, ...]): The names of the input columns to extract.
        target_columns (list[str] | tuple[str, ...]): The names of the target columns to extract.

    Returns:
        tuple[list, np.ndarray, np.ndarray] | None: Filtered dates, input arrays, and target arrays, or None if load fails.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Parsing error: Could not parse {file_path}.")
        return None

    print(f"Loaded dataset from {file_path}:")
    print(df.head())

    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    df = df[(df['Date'] >= start_date) & (
        df['Date'] <= end_date)].sort_values('Date')

    input_cols = list(input_columns)
    target_cols = list(target_columns)

    missing_inputs = [col for col in input_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]

    if missing_inputs or missing_targets:
        if missing_inputs:
            print(f"Input columns not found: {missing_inputs}")
        if missing_targets:
            print(f"Target columns not found: {missing_targets}")
        return None

    inputs = df[input_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)

    return df['Date'].tolist(), inputs, targets


def load_data(
    file_path: str,
    date_filter: tuple = ('2017-01-01', '2022-12-31'),
    input_columns: list[str] | tuple[str, ...] = ('Adj Close', 'Volume'),
    target_columns: list[str] | tuple[str, ...] = ('Adj Close',),
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load input and target feature series from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.
        date_filter (tuple): A tuple containing the start and end dates for filtering.
        input_columns (list[str] | tuple[str, ...]): The names of the input columns to extract.
        target_columns (list[str] | tuple[str, ...]): The names of the target columns to extract.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Inputs and targets arrays, or None if load fails.
    """
    loaded = load_dataset_with_dates(
        file_path, date_filter=date_filter, input_columns=input_columns, target_columns=target_columns)
    if loaded is None:
        return None

    _, inputs, targets = loaded
    return inputs, targets


@dataclass
class Dataset:
    """
    A class to represent a dataset.

    Attributes:
        input_data (np.ndarray | list): The input feature array.
        target_data (np.ndarray | list): The target feature array.
        window_size (int): The size of the sliding window for the dataset.
    """
    input_data: np.ndarray | list
    target_data: np.ndarray | list
    window_size: int = 10

    def __post_init__(self) -> None:
        """
        Initialize the dataset, converting input and target data to JAX arrays and verifying dimensions.
        """
        self.input_data = jnp.asarray(self.input_data, dtype=jnp.float32)
        if self.input_data.ndim == 1:
            self.input_data = self.input_data[:, None]

        self.target_data = jnp.asarray(self.target_data, dtype=jnp.float32)
        if self.target_data.ndim == 1:
            self.target_data = self.target_data[:, None]

        self.window_size = int(self.window_size)
        if self.window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if self.window_size >= len(self.input_data):
            raise ValueError(
                "window_size must be less than the length of the data.")
        if len(self.input_data) != len(self.target_data):
            raise ValueError(
                "input_data and target_data must have the same length.")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples available in the dataset.
        """
        return len(self.input_data) - self.window_size

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing normalized input 'x', target 'y', and min/max ranges for inputs and targets.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")
        # Prepare the input window plus the target value
        x_raw = self.input_data[idx:idx + self.window_size]
        x_min = jnp.min(x_raw, axis=0)
        x_max = jnp.max(x_raw, axis=0)
        # Normalize input window per feature column to [0, 1]
        x = (x_raw - x_min) / (x_max - x_min + 1e-8)

        # Target value
        y_raw = self.target_data[idx + self.window_size]
        y_raw_window = self.target_data[idx:idx + self.window_size]
        y_min = jnp.min(y_raw_window, axis=0)
        y_max = jnp.max(y_raw_window, axis=0)
        y = (y_raw - y_min) / (y_max - y_min + 1e-8)

        if y.shape[0] == 1:
            y = y[0]
            y_min = y_min[0]
            y_max = y_max[0]

        return {
            "x": x,
            "y": y,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1,
    drop_last: bool = True,
):
    """
    Create a DataLoader for the given dataset.

    Parameters:
        dataset (Dataset): The dataset to create the DataLoader for.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data before creating batches.
        seed (int): The random seed for shuffling.
        num_epochs (int): Number of epochs the sampler should iterate through.
        drop_last (bool): Whether to drop the final incomplete batch.

    Returns:
        iterator: An iterator yielding dictionaries with batched 'x', 'y', and min/max arrays.
    """
    num_records = len(dataset)
    if num_records == 0:
        return

    rng = np.random.default_rng(seed)

    for _ in range(num_epochs):
        indices = np.arange(num_records)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, num_records, batch_size):
            batch_indices = indices[start:start + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                continue

            samples = [dataset[int(index)] for index in batch_indices]
            yield {
                'x': jnp.stack([sample['x'] for sample in samples], axis=0),
                'y': jnp.stack([sample['y'] for sample in samples], axis=0),
                'x_min': jnp.stack([sample['x_min'] for sample in samples], axis=0),
                'x_max': jnp.stack([sample['x_max'] for sample in samples], axis=0),
                'y_min': jnp.stack([sample['y_min'] for sample in samples], axis=0),
                'y_max': jnp.stack([sample['y_max'] for sample in samples], axis=0),
            }


def main() -> None:
    """Run the entire pipeline for loading the dataset and preparing it for use."""
    file_path = 'datasets/AAPL.csv'
    input_columns = ['Adj Close', 'Volume']
    target_columns = ['Adj Close']
    date_range = ('1990-01-01', '2021-12-31')

    loaded = load_data(file_path, date_filter=date_range,
                       input_columns=input_columns, target_columns=target_columns)
    sliding_window_size = 10

    if loaded is not None:
        inputs, targets = loaded
        dataset = Dataset(input_data=inputs, target_data=targets, window_size=sliding_window_size)
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample input shape (x): {sample['x'].shape}")
        print(f"Sample target (y): {sample['y']}")
        print(f"Sample x_min: {sample['x_min']}")
        print(f"Sample x_max: {sample['x_max']}")
        print(f"Sample y_min: {sample['y_min']}")
        print(f"Sample y_max: {sample['y_max']}")

        loader = create_dataloader(
            dataset, batch_size=32, shuffle=False, seed=42)

        print("Loader batches:")
        for i, batch in enumerate(loader):
            print(f"Batch {i} x shape: {batch['x'].shape}, y shape: {batch['y'].shape}")
            if i > 2:
                break


if __name__ == "__main__":
    main()
