def fedavg(weights_list):

    avg_weights = {}

    for key in weights_list[0].keys():

        avg_weights[key] = sum(
            [weights[key] for weights in weights_list]
        ) / len(weights_list)

    return avg_weights