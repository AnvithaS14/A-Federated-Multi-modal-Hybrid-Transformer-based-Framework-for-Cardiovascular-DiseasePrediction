import argparse

import flwr as fl


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(
        num_examples * metric["accuracy"] for num_examples, metric in metrics
    ) / total_examples
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Start the Flower FedAvg server.")
    parser.add_argument("--rounds", type=int, default=24, help="Number of FL rounds")
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"SERVER STARTING: FedAvg for {args.rounds} rounds")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("SERVER ERROR:", e)
