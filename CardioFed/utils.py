import torch
import torch.nn as nn
import torch.optim as optim


def train_local_model(model, X, y, epochs=10):

    X = torch.tensor(X).float()

    y = torch.tensor(y).float().view(-1, 1)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

    return model.state_dict(), loss.item()


def evaluate_model(model, X, y):

    X = torch.tensor(X).float()

    y = torch.tensor(y).float().view(-1, 1)

    model.eval()

    with torch.no_grad():

        predictions = model(X)

        predictions = (predictions > 0.5).float()

        accuracy = (predictions == y).float().mean()

    return accuracy.item()