from dataclasses import dataclass

import numpy as np
import pandas as pd
import jax.numpy as jnp


def load_dataset_with_dates(
    file_path: str,
    date_filter: tuple = ('2017-01-01', '2022-12-31'),
    price_column: str = 'Adj Close',
) -> tuple[list, list] | None:
    """
    Load filtered dates and prices from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    date_filter (tuple): A tuple containing the start and end dates for filtering the DataFrame.
    price_column (str): The name of the column to extract from the DataFrame.

    Returns:
    tuple[list, list]: The filtered dates and the selected price column values.
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

    if price_column not in df.columns:
        print(f"Column not found: {price_column}")
        return None

    return df['Date'].tolist(), df[price_column].tolist()


def load_data(file_path: str, date_filter: tuple = ('2017-01-01', '2022-12-31'), price_column: str = 'Adj Close') -> list:
    """Load only the filtered price series from a CSV file."""
    loaded = load_dataset_with_dates(
        file_path, date_filter=date_filter, price_column=price_column)
    if loaded is None:
        return None

    _, prices = loaded
    return prices


@dataclass
class Dataset:
    """
    A class to represent a dataset.

    Attributes:
    data (list): The data loaded from the CSV file.
    window_size (int): The size of the sliding window for the dataset.
    """
    data: list
    window_size: int = 10

    def __post_init__(self):
        # Convert the list to a JAX array for further processing
        self.data = jnp.asarray(self.data, dtype=jnp.float32)
        self.window_size = int(self.window_size)
        if self.window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if self.window_size >= len(self.data):
            raise ValueError(
                "window_size must be less than the length of the data.")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        """
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): The index of the sample to retrieve.

        Returns:
        dict: A dictionary containing the input window 'x' and the target value 'y'.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")
        # Prepare the input window plus the target value
        x_raw = self.data[idx:idx + self.window_size]
        x_min = jnp.min(x_raw)
        x_max = jnp.max(x_raw)
        # Normalize the input window to [0, 1]
        x = (x_raw - x_min) / (x_max - x_min + 1e-8)
        # The target value is the next value in the series
        y = (self.data[idx + self.window_size] - x_min) / \
            (x_max - x_min + 1e-8)
        return {"x": x, "y": y, "x_min": x_min, "x_max": x_max}


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
    iterator: An iterator yielding dictionaries with batched 'x' and 'y' arrays.
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
            }


def main() -> None:
    """Run the entire pipeline for loading the dataset and preparing it for use."""
    # Reading the dataset from a CSV file
    file_path = 'AAPL.csv'  # Replace with your actual file path
    price_column = 'Adj Close'  # Replace with the actual column name if different
    # Replace with your desired date range
    date_range = ('1990-01-01', '2021-12-31')
    data = load_data(file_path, date_filter=date_range,
                     price_column=price_column)
    sliding_window_size = 10  # You can adjust this value as needed

    if data is not None:
        dataset = Dataset(data=data, window_size=sliding_window_size)
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample input (x): {sample['x']}")
        print(f"Sample target (y): {sample['y']}")
        print(f"Sample x_min: {sample['x_min']}")
        print(f"Sample x_max: {sample['x_max']}")

        loader = create_dataloader(
            dataset, batch_size=32, shuffle=False, seed=42)

        # Iterate through the loader and print the first few batches
        print("Loader batches:")
        for i, batch in enumerate(loader):
            print(f"Batch {i}: {batch}")
            if i > 2:
                break


if __name__ == "__main__":
    main()
