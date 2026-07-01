import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(client_id, num_clients=2):
    df = pd.read_csv("dataset.csv")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl") 

    # Split for federated clients
    size = len(X) // num_clients
    start = client_id * size
    end = start + size

    X_client = X[start:end]
    y_client = y[start:end]

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test