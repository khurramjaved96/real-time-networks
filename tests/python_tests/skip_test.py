"""
A test to test a "skip" connection network.

THERE IS CURRENTLY A DISCREPANCY BETWEEN THIS AND THE C++ IMPLEMENTATION.

This is due to the C++ implementation using partial gradients instead
of unrolling the computation graph back fully when you have a split in the
DAG of your NN that results in two paths to the output of different lengths.
"""
import torch
from torch import nn


if __name__ == "__main__":
    list_of_synapses = []
    list_of_synapses.append((1,2, 0.2))
    list_of_synapses.append((2,3, 0.1))
    list_of_synapses.append((2,4, 0.4))
    list_of_synapses.append((3,4, 0.3))


    list_of_state_values = []
    for a in range(1, 4):
        list_of_state_values.append(a)

    for a in range(0, 10):
        list_of_state_values.append(0)
    # for a in list_of_state_values:
    #     print(a)
    # quit()

    list_of_weights = {}
    state = torch.zeros(4, requires_grad=False)

    for a in list_of_synapses:
        dict_name = (a[0], a[1])
        list_of_weights[dict_name] = nn.Parameter(torch.zeros(1) + a[2])

    print(list_of_weights.values())
    sum_of_output = 0
    for i in range(0, 12):
        # state = state.detach()
        state[0] = list_of_state_values[i]
        temp_state = torch.zeros_like(state)
        for inner in list_of_synapses:
            dict_name = (inner[0], inner[1])
            w = list_of_weights[dict_name]
            state_from = inner[0]
            state_to = inner[1]
            temp_state[state_to-1] += state[state_from-1]*w[0]
        sum_of_output += temp_state[-1]
        print(f"Time = {i}, input = {state[0].item():.6}, output = {temp_state[-1].item():.6}")

        state = torch.nn.functional.relu(temp_state)

        state[-1] = temp_state[-1]

        state[-1].backward(retain_graph=True)
        for a in list_of_weights:
            print(a, list_of_weights[a].grad)

        print()

    # sum_of_output.backward()
