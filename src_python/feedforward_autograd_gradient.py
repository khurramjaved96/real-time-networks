import torch
import random
from torch import nn

list_of_nodes = []
list_of_nodes.append((1,4, 0.2))
list_of_nodes.append((1,6, 0.5))
list_of_nodes.append((2,4, -0.2))
list_of_nodes.append((2,5, 0.7))
list_of_nodes.append((3,4, 0.65))
list_of_nodes.append((3,5, 0.1))
list_of_nodes.append((3,7, -0.1))
list_of_nodes.append((4,6, 0.2))
list_of_nodes.append((4,7, -0.1))
list_of_nodes.append((4,5, -0.2))
list_of_nodes.append((5,7, 0.2))
list_of_nodes.append((6,7, 0.2))

list_of_state_values = []
for a in range(0, 100):
    list_of_state_values.append([a*0.01, 10 - (a*0.1), a])
# for a in range(0, 100):
#     list_of_state_values.append([0, 0, 0])
# for a in list_of_state_values:
#     print(a)
# quit()

list_of_weights = {}
state = torch.zeros(7)

for a in list_of_nodes:
    dict_name = (a[0], a[1])
    list_of_weights[dict_name] = nn.Parameter(torch.zeros(1) + a[2])
    # list_of_weights[dict_name][0] = a[2]

print(list_of_weights.values())
opti = torch.optim.SGD(list_of_weights.values(), lr=1)
sum_of_output = 0
sum_of_6th = 0;
for a in range(0, 100):
    print(list_of_state_values[a])
    for inp in range(0, 3):
        state[inp] = list_of_state_values[a][inp]
    temp_state = torch.zeros_like(state)

    for inner in list_of_nodes:
        dict_name = (inner[0], inner[1])
        w = list_of_weights[dict_name]
        state_from = inner[0]
        state_to = inner[1]
        temp_state[state_to-1] += state[state_from-1]*w[0]

    sum_of_output += state[6]
    print("Time = ", a, state[6])
    state = temp_state

    state_temp = torch.nn.functional.relu(state)

    state_temp[6] = state[6]
    state = state_temp
    if a < 100:
        sum_of_6th += state[5];



sum_of_output.backward()
for a in list_of_weights:
    print(a, list_of_weights[a].grad)
print(sum_of_6th)
print([float(x.grad) for x in list_of_weights.values()])
