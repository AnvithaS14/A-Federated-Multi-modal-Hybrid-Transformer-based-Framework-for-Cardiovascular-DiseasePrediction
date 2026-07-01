import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_data
from model import CVDModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model):
    return [value.detach().clone() for value in model.state_dict().values()]


def set_parameters(model, parameters):
    keys = model.state_dict().keys()
    state_dict = {key: value.detach().clone() for key, value in zip(keys, parameters)}
    model.load_state_dict(state_dict)


def fedavg(client_parameters, client_sizes):
    total_examples = sum(client_sizes)
    averaged = []
    for layer_values in zip(*client_parameters):
        layer_average = sum(
            value * (size / total_examples)
            for value, size in zip(layer_values, client_sizes)
        )
        averaged.append(layer_average)
    return averaged


def aggregate_parameters(
    algorithm, global_parameters, client_parameters, client_sizes, server_state
):
    averaged_parameters = fedavg(client_parameters, client_sizes)

    if algorithm == "fedavg":
        return averaged_parameters, server_state

    if algorithm == "fedavgm":
        if server_state is None:
            server_state = {
                "momentum": [torch.zeros_like(value) for value in global_parameters]
            }

        updated_momentum = []
        updated_parameters = []
        for global_value, averaged_value, momentum_value in zip(
            global_parameters, averaged_parameters, server_state["momentum"]
        ):
            delta = averaged_value - global_value
            new_momentum = 0.9 * momentum_value + delta
            updated_momentum.append(new_momentum)
            updated_parameters.append(global_value + new_momentum)

        return updated_parameters, {"momentum": updated_momentum}

    if algorithm == "fedadam":
        beta_1 = 0.9
        beta_2 = 0.99
        server_lr = 0.1
        tau = 1e-3

        if server_state is None:
            server_state = {
                "m": [torch.zeros_like(value) for value in global_parameters],
                "v": [torch.zeros_like(value) for value in global_parameters],
            }

        updated_m = []
        updated_v = []
        updated_parameters = []
        for global_value, averaged_value, m_value, v_value in zip(
            global_parameters,
            averaged_parameters,
            server_state["m"],
            server_state["v"],
        ):
            delta = averaged_value - global_value
            new_m = beta_1 * m_value + (1 - beta_1) * delta
            new_v = beta_2 * v_value + (1 - beta_2) * torch.square(delta)
            update = server_lr * new_m / (torch.sqrt(new_v) + tau)

            updated_m.append(new_m)
            updated_v.append(new_v)
            updated_parameters.append(global_value + update)

        return updated_parameters, {"m": updated_m, "v": updated_v}

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_optimizer(name, parameters, local_lr):
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    return optimizers[name](parameters, lr=local_lr)


def build_clients(num_clients, seed, local_lr, optimizer_name):
    clients = []
    for client_id in range(num_clients):
        X_train, X_test, y_train, y_test = load_data(client_id, num_clients, seed)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        model = CVDModel(X_train.shape[1])
        optimizer = create_optimizer(optimizer_name, model.parameters(), local_lr)
        clients.append(
            {
                "model": model,
                "optimizer": optimizer,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )
    return clients


def train_local(client, parameters, local_epochs):
    set_parameters(client["model"], parameters)
    client["model"].train()
    loss_fn = torch.nn.BCELoss()

    for _ in range(local_epochs):
        client["optimizer"].zero_grad()
        pred = client["model"](client["X_train"])
        loss = loss_fn(pred, client["y_train"])
        loss.backward()
        client["optimizer"].step()

    return get_parameters(client["model"]), len(client["X_train"])


def evaluate_global(global_model, parameters, clients):
    set_parameters(global_model, parameters)
    global_model.eval()
    loss_fn = torch.nn.BCELoss()

    total_examples = 0
    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for client in clients:
            pred = global_model(client["X_test"])
            y_test = client["y_test"]
            loss = loss_fn(pred, y_test).item()
            correct = ((pred > 0.5) == y_test).float().sum().item()
            examples = len(y_test)

            total_loss += loss * examples
            total_correct += correct
            total_examples += examples

    return total_loss / total_examples, total_correct / total_examples


def run_training(
    max_rounds, num_clients, local_epochs, local_lr, seed, optimizer_name, algorithm
):
    set_seed(seed)
    clients = build_clients(num_clients, seed, local_lr, optimizer_name)
    global_model = CVDModel(clients[0]["X_train"].shape[1])
    global_parameters = get_parameters(global_model)
    server_state = None
    round_history = []

    for round_num in range(1, max_rounds + 1):
        local_parameters = []
        local_sizes = []

        for client in clients:
            params, size = train_local(client, global_parameters, local_epochs)
            local_parameters.append(params)
            local_sizes.append(size)

        global_parameters, server_state = aggregate_parameters(
            algorithm,
            global_parameters,
            local_parameters,
            local_sizes,
            server_state,
        )
        loss, accuracy = evaluate_global(global_model, global_parameters, clients)
        round_history.append(
            {
                "round": round_num,
                "loss": loss,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
            }
        )

    return round_history


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(round_history, output_path):
    rounds = [row["round"] for row in round_history]
    accuracy = [row["accuracy_percent"] for row in round_history]
    best = max(round_history, key=lambda row: row["accuracy"])
    tick_step = max(1, len(rounds) // 10)

    plt.figure(figsize=(9, 5.5))
    plt.plot(rounds, accuracy, marker="o", color="#2563eb", linewidth=2)
    plt.scatter(
        [best["round"]],
        [best["accuracy_percent"]],
        color="#dc2626",
        s=80,
        label=f"Best: round {best['round']} ({best['accuracy_percent']:.2f}%)",
    )
    plt.title("Accuracy vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Global Test Accuracy (%)")
    plt.xticks(rounds[::tick_step])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_loss(round_history, output_path):
    rounds = [row["round"] for row in round_history]
    loss = [row["loss"] for row in round_history]
    tick_step = max(1, len(rounds) // 10)

    plt.figure(figsize=(9, 5.5))
    plt.plot(rounds, loss, marker="o", color="#16a34a", linewidth=2)
    plt.title("Loss vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Global Test Loss")
    plt.xticks(rounds[::tick_step])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_optimizer_comparison(comparison_rows, output_path, algorithm):
    labels = [row["local_optimizer"] for row in comparison_rows]
    accuracy = [row["best_accuracy_percent"] for row in comparison_rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracy, color=["#2563eb", "#16a34a", "#f97316", "#7c3aed"])
    plt.title(f"Optimizer Comparison with {algorithm}")
    plt.xlabel("Local Optimizer")
    plt.ylabel("Best Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, accuracy):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def summarize_run(round_history, args, optimizer_name):
    best = max(round_history, key=lambda row: row["accuracy"])
    return {
        "algorithm": args.algorithm,
        "local_optimizer": optimizer_name,
        "num_clients": args.num_clients,
        "local_epochs": args.local_epochs,
        "learning_rate": args.local_lr,
        "seed": args.seed,
        "best_iteration": best["round"],
        "best_accuracy": best["accuracy"],
        "best_accuracy_percent": best["accuracy_percent"],
        "loss_at_best_iteration": best["loss"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep FedAvg iterations for CVD prediction."
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--local-lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "rmsprop", "adagrad"],
        default="adam",
    )
    parser.add_argument(
        "--algorithm",
        choices=["FedAvg", "FedAvgM", "FedAdam"],
        default="FedAvg",
        help="Global aggregation algorithm.",
    )
    parser.add_argument(
        "--compare-optimizers",
        action="store_true",
        help="Compare Adam, SGD, RMSprop, and Adagrad with FedAvg.",
    )
    args = parser.parse_args()

    if args.compare_optimizers:
        comparison_rows = []
        for optimizer_name in ["adam", "sgd", "rmsprop", "adagrad"]:
            round_history = run_training(
                max_rounds=args.max_rounds,
                num_clients=args.num_clients,
                local_epochs=args.local_epochs,
                local_lr=args.local_lr,
                seed=args.seed,
                optimizer_name=optimizer_name,
                algorithm=args.algorithm.lower(),
            )
            comparison_rows.append(summarize_run(round_history, args, optimizer_name))

        comparison_csv = f"{args.algorithm.lower()}_optimizer_comparison.csv"
        comparison_graph = f"{args.algorithm.lower()}_optimizer_comparison_graph.png"
        write_csv(
            comparison_csv,
            comparison_rows,
            [
                "algorithm",
                "local_optimizer",
                "num_clients",
                "local_epochs",
                "learning_rate",
                "seed",
                "best_iteration",
                "best_accuracy",
                "best_accuracy_percent",
                "loss_at_best_iteration",
            ],
        )
        plot_optimizer_comparison(comparison_rows, comparison_graph, args.algorithm)

        best = max(comparison_rows, key=lambda row: row["best_accuracy"])
        print(f"{args.algorithm} optimizer comparison complete")
        print(f"Best optimizer: {best['local_optimizer']}")
        print(f"Best iteration: {best['best_iteration']}")
        print(f"Best accuracy: {best['best_accuracy_percent']:.2f}%")
        print(f"Saved: {comparison_csv}, {comparison_graph}")
        return

    round_history = run_training(
        max_rounds=args.max_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        seed=args.seed,
        optimizer_name=args.optimizer,
        algorithm=args.algorithm.lower(),
    )

    sweep_rows = [
        {
            "iterations_used": row["round"],
            "final_accuracy": row["accuracy"],
            "final_accuracy_percent": row["accuracy_percent"],
            "final_loss": row["loss"],
        }
        for row in round_history
    ]
    best = max(round_history, key=lambda row: row["accuracy"])
    summary_rows = [summarize_run(round_history, args, args.optimizer)]

    write_csv(
        "round_sweep_results.csv",
        sweep_rows,
        ["iterations_used", "final_accuracy", "final_accuracy_percent", "final_loss"],
    )
    write_csv(
        "round_sweep_summary.csv",
        summary_rows,
        [
            "algorithm",
            "local_optimizer",
            "num_clients",
            "local_epochs",
            "learning_rate",
            "seed",
            "best_iteration",
            "best_accuracy",
            "best_accuracy_percent",
            "loss_at_best_iteration",
        ],
    )

    with open("accuracy.txt", "w") as f:
        for row in round_history:
            f.write(f"{row['accuracy']}\n")

    plot_accuracy(round_history, "accuracy_graph.png")
    plot_loss(round_history, "loss_graph.png")

    print(f"{args.algorithm} iteration sweep complete")
    print(f"Best iteration: {best['round']}")
    print(f"Best accuracy: {best['accuracy_percent']:.2f}%")
    print("Saved: round_sweep_results.csv, round_sweep_summary.csv")
    print("Saved: accuracy_graph.png, loss_graph.png, accuracy.txt")


if __name__ == "__main__":
    main()
