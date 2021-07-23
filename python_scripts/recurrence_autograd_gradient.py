import torch
import random
from torch import nn

list_of_nodes = []
list_of_nodes.append((1,3, 0.9))
list_of_nodes.append((2,3, -0.8))
list_of_nodes.append((3,3, 0.6))
list_of_nodes.append((3,4, 0.7))


list_of_state_values = []

# for a in range(0, 100):
#     list_of_state_values.append([0, 0])

for a in range(0, 100):
    list_of_state_values.append([ 10 - (a*0.1), a*0.01])

# for a in range(0, 100):
#     list_of_state_values.append([0, 0])


list_of_weights = {}
state = torch.zeros(4)

for a in list_of_nodes:
    dict_name = (a[0], a[1])
    list_of_weights[dict_name] = nn.Parameter(torch.zeros(1) + a[2])
    # list_of_weights[dict_name][0] = a[2]

print(list_of_weights.values())
opti = torch.optim.SGD(list_of_weights.values(), lr=1)
sum_of_output = 0
sum_of_state =0;
for a in range(0, 100):
    for inp in range(0, 2):
        state[inp] = list_of_state_values[a][inp]
    temp_state = torch.zeros_like(state)
    for inner in list_of_nodes:
        dict_name = (inner[0], inner[1])
        w = list_of_weights[dict_name]
        state_from = inner[0]
        state_to = inner[1]
        # print(state_to -1, state_from -1)
        temp_state[state_to-1] += state[state_from-1]*w[0]
    sum_of_output += state[3]
    # print("Time = ", a, state[3])
    state = temp_state

    state_temp = torch.nn.functional.relu(state)
    state_temp[3] = state[3]
    sum_of_state += state[2]
    # print(a, list_of_state_/values[a])
    print(state, "Counter", a, "State = ", state[3].item())
    state = state_temp


opti.zero_grad();
sum_of_output.backward()
print(sum_of_state)
for a in list_of_weights:
    print(a, list_of_weights[a].grad)