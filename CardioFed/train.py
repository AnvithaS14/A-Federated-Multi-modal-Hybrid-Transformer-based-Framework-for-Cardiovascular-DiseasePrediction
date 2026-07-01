import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from client1.ecg_to_csv import convert_to_csv

from models.fcnn import FCNN
from models.mlp import MLP

from server.fed import fedavg

from utils import train_local_model
from utils import evaluate_model


# ---------------------------------------------------
# CLIENT 1 : ECG IMAGE -> CNN -> CSV
# ---------------------------------------------------

client1_dataframe = convert_to_csv("client1/ecg_client")


# ---------------------------------------------------
# CLIENT 2 : TABULAR CSV
# ---------------------------------------------------

client2_dataframe = pd.read_csv("client2/data.csv")


# ---------------------------------------------------
# SPLIT FEATURES AND LABELS
# ---------------------------------------------------

def split_features_labels(dataframe):

    X = dataframe.iloc[:, :-1].values

    y = dataframe.iloc[:, -1].values

    return X, y


X1, y1 = split_features_labels(client1_dataframe)

X2, y2 = split_features_labels(client2_dataframe)


# ---------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------

scaler1 = StandardScaler()

X1 = scaler1.fit_transform(X1)


scaler2 = StandardScaler()

X2 = scaler2.fit_transform(X2)


# ---------------------------------------------------
# SAME INPUT SIZE
# ---------------------------------------------------

input_size = X1.shape[1]


# ---------------------------------------------------
# INITIALIZE GLOBAL MODEL
# ---------------------------------------------------

global_model = FCNN(input_size)

global_weights = global_model.state_dict()


# ---------------------------------------------------
# TRAINING PARAMETERS
# ---------------------------------------------------

rounds = 15

loss_history = []

accuracy_history = []


# ---------------------------------------------------
# FEDERATED LEARNING
# ---------------------------------------------------

for round_number in range(rounds):

    print(f"\n========== ROUND {round_number + 1} ==========")

    # ---------------------------------------------
    # CLIENT 1 : FCNN
    # ---------------------------------------------

    client1_model = FCNN(input_size)

    client1_model.load_state_dict(global_weights)

    client1_weights, client1_loss = train_local_model(
        client1_model,
        X1,
        y1,
        epochs=10
    )

    # ---------------------------------------------
    # CLIENT 2 : MLP
    # ---------------------------------------------

    client2_model = MLP(input_size)

    client2_model.load_state_dict(global_weights)

    client2_weights, client2_loss = train_local_model(
        client2_model,
        X2,
        y2,
        epochs=10
    )

    # ---------------------------------------------
    # FEDAVG AGGREGATION
    # ---------------------------------------------

    aggregated_weights = fedavg([
        client1_weights,
        client2_weights
    ])

    global_weights = aggregated_weights

    global_model.load_state_dict(global_weights)

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------

    accuracy_client1 = evaluate_model(
        global_model,
        X1,
        y1
    )

    accuracy_client2 = evaluate_model(
        global_model,
        X2,
        y2
    )

    average_accuracy = (
        accuracy_client1 +
        accuracy_client2
    ) / 2

    average_loss = (
        client1_loss +
        client2_loss
    ) / 2

    loss_history.append(average_loss)

    accuracy_history.append(average_accuracy)

    print(f"Loss : {average_loss:.4f}")

    print(f"Accuracy : {average_accuracy:.4f}")




plt.figure(figsize=(10, 5)) 
round_numbers = list(range(1, rounds + 1)) 
plt.plot(round_numbers, loss_history, marker='o', label="Loss") 
plt.plot(round_numbers, accuracy_history, marker='o', label="Accuracy") 
plt.xticks(round_numbers) 
plt.xlabel("Epochs") 
plt.ylabel("Value") 
plt.title(" CardioFed - Performance Analysis") 
plt.legend() 
plt.show()