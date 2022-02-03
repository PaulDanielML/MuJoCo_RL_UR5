import torch
import time
from collections import defaultdict

final_data = defaultdict(list)
total_length = 0

try:
    for i in range(1, 1000):
        path = f"Data/grasping_data_{i}.pt"
        new_data = torch.load(path)
        # print(type(new_data['states']))
        # print(len(new_data['states']))
        # print(*new_data['rewards'])
        # print(*new_data['actions'])
        final_data["states"] += new_data["states"]
        final_data["actions"] += new_data["actions"]
        final_data["rewards"] += new_data["rewards"]
        total_length += len(new_data["actions"])
        print(f"loaded file {path}.")
except Exception as e:
    # print(e)
    print(f"Tried to find file {path}, but did not find it.")
finally:
    print("Let's save what we have.")
    current_time = time.strftime("%d_%m_%y_%H_%M", time.localtime())
    output_file = f"Data/{current_time}_total_of_{total_length}_transitions.pt"
    torch.save(final_data, output_file)
    print(f"Successfully saved to {output_file}.")


# print(*final_data['rewards'])
