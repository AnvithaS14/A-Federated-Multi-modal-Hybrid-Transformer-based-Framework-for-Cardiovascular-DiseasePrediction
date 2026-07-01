# =========================
# server.py
# =========================

import flwr as fl
import torch
from model import CVDModel

# -------------------------
# Metrics Aggregation
# -------------------------
def fit_metrics_aggregation(metrics):
    total = sum(n for n, _ in metrics)

    acc = sum(
        n * m["train_accuracy"]
        for n, m in metrics
    ) / total

    loss = sum(
        n * m["train_loss"]
        for n, m in metrics
    ) / total

    return {
        "train_accuracy": acc,
        "train_loss": loss,
    }


def eval_metrics_aggregation(metrics):
    total = sum(n for n, _ in metrics)

    acc = sum(
        n * m["val_accuracy"]
        for n, m in metrics
    ) / total

    return {
        "val_accuracy": acc
    }


# -------------------------
# Initial Global Model
# -------------------------
def get_initial_parameters():
    model = CVDModel(input_dim=13)

    return [
        val.cpu().numpy()
        for val in model.state_dict().values()
    ]


initial_parameters = fl.common.ndarrays_to_parameters(
    get_initial_parameters()
)

# ======================================================
# GLOBAL OPTIMIZER
# ======================================================

# -------------------------
# FedAvgM (CURRENT)
# -------------------------
strategy = fl.server.strategy.FedAvgM(
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,

    fit_metrics_aggregation_fn=fit_metrics_aggregation,
    evaluate_metrics_aggregation_fn=eval_metrics_aggregation,

    server_momentum=0.9,

    initial_parameters=initial_parameters
)

# ======================================================
# OTHER GLOBAL ALGORITHMS
# Uncomment ONLY ONE at a time
# ======================================================

# -------------------------
# FedAvg
# -------------------------
# strategy = fl.server.strategy.FedAvg(
#     min_fit_clients=3,
#     min_available_clients=3,
#     min_evaluate_clients=3,
#
#     fit_metrics_aggregation_fn=fit_metrics_aggregation,
#     evaluate_metrics_aggregation_fn=eval_metrics_aggregation,
#
#     initial_parameters=initial_parameters
# )

# -------------------------
# FedAdam
# -------------------------
# strategy = fl.server.strategy.FedAdam(
#     min_fit_clients=3,
#     min_available_clients=3,
#     min_evaluate_clients=3,
#
#     fit_metrics_aggregation_fn=fit_metrics_aggregation,
#     evaluate_metrics_aggregation_fn=eval_metrics_aggregation,
#
#     eta=0.01,
#     eta_l=0.01,
#     beta_1=0.9,
#     beta_2=0.99,
#     tau=1e-9,
#
#     initial_parameters=initial_parameters
# )

# -------------------------
# Start Server
# -------------------------
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)