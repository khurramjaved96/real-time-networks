import torch
from argparse import ArgumentParser
from torch import nn

def create_base_nn():
    list_of_synapses = []
    list_of_synapses.append((1,4, 0.2))
    list_of_synapses.append((1,6, 0.5))
    list_of_synapses.append((2,4, -0.2))
    list_of_synapses.append((2,5, 0.7))
    list_of_synapses.append((3,4, 0.65))
    list_of_synapses.append((3,5, 0.1))
    # list_of_synapses.append((3,7, -0.1))
    # list_of_synapses.append((4,6, 0.2))
    # list_of_synapses.append((4,5, -0.2))
    list_of_synapses.append((4,7, -0.1))
    list_of_synapses.append((5,7, 0.2))
    list_of_synapses.append((6,7, 0.2))

    list_of_weights = {}

    # The +1 is for adding/removing a neuron.
    state = torch.zeros(7 + 1, requires_grad=False)

    for a in list_of_synapses:
        dict_name = (a[0], a[1])
        list_of_weights[dict_name] = nn.Parameter(torch.zeros(1) + a[2])
        # list_of_weights[dict_name][0] = a[2]

    print(list_of_weights.values())
    return list_of_weights, list_of_synapses

def create_dataset():

    list_of_state_values = []
    for a in range(0, 100):
        list_of_state_values.append([a*0.01, 10 - (a*0.1), a])

    for a in range(0, 100):
        list_of_state_values.append([0, 0, 0])

    return list_of_state_values

def parse_args():
    parser = ArgumentParser(description="Add delete test")
    parser.add_argument('--test', type=str, default='add_neuron', help='Type of test to run (add_neuron | delete_neuron | add_synapse | delete_synapse)')

    return parser

def get_changes(test_type : str = "add_neuron"):
    weights_to_add = {}
    synapses_to_add = []

    synapses_to_delete = []
    states_to_reset = []
    if test_type == "add_neuron":
        synapses_to_add = [
            (1, 8, 0.01), (2, 8, 0.01), (3, 8, 0.01),
            (8, 7, 1.0)
        ]
        for i, o, w in synapses_to_add:
            weights_to_add[(i, o)] = nn.Parameter(torch.zeros(1) + w)
    elif test_type == "delete_neuron":
        synapses_to_delete = [
            (1, 6, 0.5), (6, 7, 0.2)
        ]
        states_to_reset = [5]
    elif test_type == "add_synapse":
        synapses_to_add = [(1, 5, 0.6)]
        for i, o, w in synapses_to_add:
            weights_to_add[(i, o)] = nn.Parameter(torch.zeros(1) + w)
    elif test_type == "delete_synapse":
        synapses_to_delete = [(2, 5)]
    else:
        raise NotImplementedError()

    return synapses_to_add, weights_to_add, synapses_to_delete, states_to_reset


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    test_type = args.test

    synapses_to_add, weights_to_add, synapses_to_delete, states_to_reset = get_changes(test_type)

    list_of_state_values = create_dataset()
    list_of_weights, list_of_synapses = create_base_nn()

    out_neuron_idx = 6

    state = torch.zeros(8)
    sum_of_output = 0
    for i in range(0, 200):
        # state = state.detach()
        if i == 24:
            if synapses_to_add:
                list_of_synapses += synapses_to_add
                for b, w in weights_to_add.items():
                    list_of_weights[b] = w
            if synapses_to_delete:
                for syn in synapses_to_delete:
                    list_of_synapses.remove(syn)
                    del list_of_weights[syn[:-1]]
                for del_s in states_to_reset:
                    state[del_s] = 0

        for inp in range(0, 3):
            state[inp] = list_of_state_values[i][inp]

        temp_state = torch.zeros_like(state)
        for inner in list_of_synapses:
            dict_name = (inner[0], inner[1])
            w = list_of_weights[dict_name]
            state_from = inner[0]
            state_to = inner[1]
            temp_state[state_to-1] += state[state_from-1]*w[0]
        sum_of_output += temp_state[out_neuron_idx]
        print(f"Time = {i}, output = {temp_state[out_neuron_idx].item():.6}")

        state = torch.nn.functional.relu(temp_state)

        state[out_neuron_idx] = temp_state[out_neuron_idx]

        state[out_neuron_idx].backward(retain_graph=True)
        for a in list_of_weights:
            print(a, list_of_weights[a].grad)

        print()

    # sum_of_output.backward()
