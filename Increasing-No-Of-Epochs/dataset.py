import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(client_id, num_clients=2, random_state=42):
    df = pd.read_csv("dataset.csv")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Use the same deterministic shuffle for every client so the slices are
    # stable and non-overlapping across runs.
    indices = np.arange(len(X))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split for clients
    size = len(X) // num_clients
    start = client_id * size
    end = len(X) if client_id == num_clients - 1 else start + size

    X_client = X[start:end]
    y_client = y[start:end]

    return train_test_split(X_client, y_client, test_size=0.2, random_state=random_state)
