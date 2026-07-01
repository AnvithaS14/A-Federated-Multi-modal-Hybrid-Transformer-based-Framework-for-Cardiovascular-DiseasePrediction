import flwr as fl
import torch
from model import CVDModel
from dataset import load_data

DEVICE = torch.device("cpu")
INPUT_DIM = 13

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.cid = int(cid)
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(self.cid, num_clients)

        print(f"Client {self.cid} feature size:", self.X_train.shape[1])

        self.model = CVDModel(INPUT_DIM).to(DEVICE)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01)
        #self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(),lr=0.01)
        #self.optimizer = torch.optim.Adagrad(self.model.parameters(),lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        X = torch.tensor(self.X_train, dtype=torch.float32)
        y = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)

        self.model.train()
        for _ in range(3):
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == y).float().mean().item()

        return self.get_parameters(config), len(X), {
            "train_accuracy": acc * 100,
            "train_loss": loss.item()
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        X = torch.tensor(self.X_test, dtype=torch.float32)
        y = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            loss = self.criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == y).float().mean().item()

        return loss, len(X), {
            "val_accuracy": acc * 100
        }


def main():
    import sys
    cid = sys.argv[1]
    num_clients = int(sys.argv[2])

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(cid, num_clients)
    )


if __name__ == "__main__":
    main()