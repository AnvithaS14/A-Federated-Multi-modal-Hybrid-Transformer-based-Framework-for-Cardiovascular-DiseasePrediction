import flwr as fl
import torch
import argparse
from model import CVDModel
from dataset import load_data

# -------------------------------
# Client Class
# -------------------------------
class Client(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.cid = int(cid)

        # Load data
        X_train, X_test, y_train, y_test = load_data(self.cid, num_clients)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Model
        self.model = CVDModel(self.X_train.shape[1])

        # Loss + Optimizer
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    # -------------------------------
    # Get model parameters
    # -------------------------------
    def get_parameters(self, config):
        return [val.detach().numpy() for val in self.model.state_dict().values()]

    # -------------------------------
    # Set model parameters
    # -------------------------------
    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        state_dict = dict(zip(keys, [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict)

    # -------------------------------
    # Training (Local)
    # -------------------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for _ in range(5):  # epochs
            self.optimizer.zero_grad()
            pred = self.model(self.X_train)
            loss = self.loss_fn(pred, self.y_train)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config), len(self.X_train), {}

    # -------------------------------
    # Evaluation
    # -------------------------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.X_test)
            loss = self.loss_fn(pred, self.y_test).item()
            accuracy = ((pred > 0.5) == self.y_test).float().mean().item()

        return loss, len(self.X_test), {"accuracy": accuracy}

    # -------------------------------
    # Required for new Flower API
    # -------------------------------
    def to_client(self):
        return fl.client.NumPyClient.to_client(self)


# -------------------------------
# Start Client
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a CVD federated client.")
    parser.add_argument("cid", type=int, help="Client ID, starting from 0")
    parser.add_argument("--num-clients", type=int, default=2)
    args = parser.parse_args()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=Client(args.cid, args.num_clients).to_client()
    )
