import matplotlib.pyplot as plt

# Read accuracy values
with open("accuracy.txt", "r") as f:
    acc = [float(line.strip()) for line in f.readlines()]

rounds = list(range(1, len(acc)+1))

plt.figure(figsize=(8,5))
plt.plot(rounds, acc, marker='o', color='blue')
plt.title("Federated Learning Accuracy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("accuracy_graph.png")
plt.show()