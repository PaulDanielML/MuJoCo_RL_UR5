import torch
from collections import defaultdict

file = "Data/14_09_20_13_56_total_of_16968_transitions.pt"

data = torch.load(file)
output_data = defaultdict(list)
idx_list = []

for i, d in enumerate(data["rewards"]):
    if d == 1:
        idx_list.append(i)

for index in idx_list:
    output_data["states"].append(data["states"][index])
    output_data["actions"].append(data["actions"][index])
    output_data["rewards"].append(data["rewards"][index])


new_file = file[:-3] + "_positives_" + str(len(output_data["states"])) + ".pt"

print(f"Saved transitions with a reward of 1 to {new_file}.")
torch.save(output_data, new_file)
