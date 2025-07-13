import numpy as np
import torch 
from models.NanoBind_pair import NanoBind_pair
from utils.dataloader import anli_aff
from torch.utils.data import DataLoader
 
def predicting(model, device, loader1, loader2):
    model.eval()
    
    comparison_matrix = np.zeros((len(loader1.dataset), len(loader2.dataset)), dtype=int)

    for i, data1 in enumerate(loader1):
        ab1, ag1 = data1

        for j, data2 in enumerate(loader2):
            ab2, ag2 = data2
            with torch.no_grad():
                prob = model(ab1, ag1, ab2, ag2,device).item()

                comparison_matrix[i, j] = 1 if prob > 0.5 else 0
                
    return comparison_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NanoBind_pair().to(device)

model_path = './output/checkpoint/NanoBind_pair_anli.model'
weights = torch.load(model_path, map_location=device)
model.load_state_dict(weights)

Dataset1 = anli_aff('./data/affinity/test_anli.csv')
Dataset2 = anli_aff('./data/affinity/sorted_file.csv')
loader1 = DataLoader(Dataset1, batch_size=1, shuffle=False)
loader2 = DataLoader(Dataset2, batch_size=1, shuffle=False)

comparison_matrix = predicting(model, device, loader1, loader2)

# 定义区间
intervals = [
"[1.9999000000000003e-11, 3.0000000000000006e-11]",
"[3.0000000000000006e-11, 5.400000000000001e-11]",
"[5.400000000000001e-11, 7.383e-11]",
"[7.383e-11, 8.400000000000001e-11]",
"[8.400000000000001e-11, 1e-10]",
"[1e-10, 1.3e-10]",
"[1.3e-10, 1.4e-10]",
"[1.4e-10, 1.8e-10]",
"[1.8e-10, 5.4e-10]",
"[5.4e-10, 6.27e-10]",
"[6.27e-10, 6.3e-10]",
"[6.3e-10, 6.9e-10]",
"[6.9e-10, 7e-10]",
"[7e-10, 8.1e-10]",
"[8.1e-10, 1e-09]",
"[1e-09, 1.2000000000000002e-09]",
"[1.2000000000000002e-09, 1.4000000000000001e-09]",
"[1.4000000000000001e-09, 1.6200000000000002e-09]",
"[1.6200000000000002e-09, 2.2000000000000003e-09]",
"[2.2000000000000003e-09, 2.36e-09]",
"[2.36e-09, 2.9000000000000003e-09]",
"[2.9000000000000003e-09, 3.0000000000000004e-09]",
"[3.0000000000000004e-09, 3.2e-09]",
"[3.2e-09, 3.5000000000000003e-09]",
"[3.5000000000000003e-09, 3.5700000000000003e-09]",
"[3.5700000000000003e-09, 3.9900000000000005e-09]",
"[3.9900000000000005e-09, 5.3e-09]",
"[5.3e-09, 9.09999999999999e-09]",
"[9.09999999999999e-09, 9.5e-09]",
"[9.5e-09, 9.600000000000002e-09]",
"[9.600000000000002e-09, 1e-08]",
"[1e-08, 1.12e-08]",
"[1.12e-08, 1.54e-08]",
"[1.54e-08, 1.6e-08]",
"[1.6e-08, 2e-08]",
"[2e-08, 2.3e-08]",
"[2.3e-08, 2.5e-08]",
"[2.5e-08, 2.6e-08]",
"[2.6e-08, 2.8e-08]",
"[2.8e-08, 3.6e-08]",
"[3.6e-08, 4.39999999999999e-08]",
"[4.39999999999999e-08, 4.4e-08]",
"[4.4e-08, 4.7e-08]",
"[4.7e-08, 5.2e-08]",
"[5.2e-08, 6e-08]",
"[6e-08, 7.2e-08]",
"[7.2e-08, 9.4e-08]",
"[9.4e-08, 1e-07]",
"[1e-07, 1.16e-07]",
"[1.16e-07, 1.57e-07]",
"[1.57e-07, 1.66e-07]",
"[1.66e-07, 2.76e-07]",
"[2.76e-07, 4.51e-07]",
"[4.51e-07, 8.5e-07]",
"[8.5e-07, 5.7e-06]"
]

def find_best_positions(matrix, intervals):
    num_rows = len(matrix)
    results = []
    for row_idx in range(num_rows):
        row = matrix[row_idx]
        prefix = [0] * (len(intervals)+2)
        for i in range(1, len(intervals)+2):
            prefix[i] = prefix[i-1] + row[i-1]
        suffix = [0] * (len(intervals)+2)
        for i in range(len(intervals), -1, -1):
            suffix[i] = suffix[i+1] + (1 if row[i] == 0 else 0)

        max_sum = -1
        best_positions = []
        for i in range(len(intervals)+2):
            current_sum = prefix[i] + suffix[i]
            if current_sum > max_sum:
                max_sum = current_sum
                best_positions = [i]
            elif current_sum == max_sum:
                best_positions.append(i)

        descriptions = []
        for pos in best_positions:
            if pos == 0:
                desc = "< 1.9999000000000003e-11 M"
            elif pos == len(intervals)+1:
                desc = "> 5.7e-06 M"
            else:
                desc = f"{intervals[pos-1]} M"
            descriptions.append(desc)

        results.append({
            'max_sum': max_sum,
            'descriptions': descriptions
        })
    return results
 
results = find_best_positions(comparison_matrix, intervals)
for idx, res in enumerate(results):
    print(f"The predicted Kd value of Pair {idx+1}:")
    for desc in res['descriptions']:
        print(f"{desc}")
    print()