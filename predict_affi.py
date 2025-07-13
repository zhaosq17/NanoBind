import numpy as np
import torch 
from models.NanoBind_pair import NanoBind_pair
from utils.dataloader import anli_aff
from torch.utils.data import DataLoader
import argparse
from Bio import SeqIO

def get_args():
    parser = argparse.ArgumentParser(description='Demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nanobody_fasta', dest='nanobody_fasta', help='Path to the nanobody FASTA file')    
    parser.add_argument('--antigen_fasta', dest='antigen_fasta', help='Path to the antigen FASTA file')   
    return parser.parse_args()

def read_fasta(fasta_path):
    # 读取FASTA文件并返回第一个序列
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError("No sequences found in the FASTA file.")
    return str(records[0].seq)

args = get_args()

def predicting(model, device, loader2):
    model.eval()
    comparison_matrix = np.zeros((1, len(loader2.dataset)), dtype=int)

    ab1 = read_fasta(args.nanobody_fasta)
    ag1 = read_fasta(args.antigen_fasta)
    for j, data2 in enumerate(loader2):
        ab2, ag2 = data2
        with torch.no_grad():
            prob = model(ab1, ag1, ab2, ag2, device).item()
            comparison_matrix[0, j] = 1 if prob > 0.2 else 0 
                
    return comparison_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
model = NanoBind_pair().to(device)

model_path = './output/checkpoint/NanoBind_pair_random.model'
weights = torch.load(model_path, map_location=device)
model.load_state_dict(weights)

Dataset2 = anli_aff('./data/affinity/qujian.csv')
loader2 = DataLoader(Dataset2, batch_size=1, shuffle=False)

# 生成比较矩阵
comparison_matrix = predicting(model, device, loader2)

# 定义区间
intervals = [
    "[7.1699990000000006e-12, 2e-11]",
    "[2e-11, 3e-11]",
    "[3e-11, 5.4e-11]",
    "[5.4e-11, 7.382999999999999e-11]",
    "[7.382999999999999e-11, 8.4e-11]",
    "[8.4e-11, 9.999999999999999e-11]",
    "[9.999999999999999e-11, 1.3000000000000002e-10]",
    "[1.3000000000000002e-10, 1.4000000000000003e-10]",
    "[1.4000000000000003e-10, 1.8e-10]",
    "[1.8e-10, 5.400000000000001e-10]",
    "[5.400000000000001e-10, 6.269999999999999e-10]",
    "[6.269999999999999e-10, 6.3e-10]",
    "[6.3e-10, 6.9e-10]",
    "[6.9e-10, 7e-10]",
    "[7e-10, 8.1e-10]",
    "[8.1e-10, 1e-09]",
    "[1e-09, 1.2e-09]",
    "[1.2e-09, 1.4e-09]",
    "[1.4e-09, 1.4999999999999998e-09]",
    "[1.4999999999999998e-09, 1.58e-09]",
    "[1.58e-09, 1.6199999999999998e-09]",
    "[1.6199999999999998e-09, 2.2000000000000003e-09]",
    "[2.2000000000000003e-09, 2.36e-09]",
    "[2.36e-09, 2.9e-09]",
    "[2.9e-09, 2.9999999999999996e-09]",
    "[2.9999999999999996e-09, 3.2000000000000005e-09]",
    "[3.2000000000000005e-09, 3.5e-09]",
    "[3.5e-09, 3.57e-09]",
    "[3.57e-09, 3.9900000000000005e-09]",
    "[3.9900000000000005e-09, 5.3e-09]",
    "[5.3e-09, 9.099999999999987e-09]",
    "[9.099999999999987e-09, 9.5e-09]",
    "[9.5e-09, 9.6e-09]",
    "[9.6e-09, 1e-08]",
    "[1e-08, 1.12e-08]",
    "[1.12e-08, 1.5400000000000002e-08]",
    "[1.5400000000000002e-08, 1.6e-08]",
    "[1.6e-08, 2e-08]",
    "[2e-08, 2.3e-08]",
    "[2.3e-08, 2.5e-08]",
    "[2.5e-08, 2.6e-08]",
    "[2.6e-08, 2.8000000000000012e-08]",
    "[2.8000000000000012e-08, 3.6000000000000005e-08]",
    "[3.6000000000000005e-08, 4.4e-08]",
    "[4.4e-08, 4.4000000000000004e-08]",
    "[4.4000000000000004e-08, 4.7000000000000004e-08]",
    "[4.7000000000000004e-08, 5.2e-08]",
    "[5.2e-08, 6.000000000000002e-08]",
    "[6.000000000000002e-08, 7.2e-08]",
    "[7.2e-08, 9.400000000000001e-08]",
    "[9.400000000000001e-08, 1e-07]",
    "[1e-07, 1.16e-07]",
    "[1.16e-07, 1.5700000000000004e-07]",
    "[1.5700000000000004e-07, 1.66e-07]",
    "[1.66e-07, 2.760000000000001e-07]",
    "[2.760000000000001e-07, 4.51e-07]",
    "[4.51e-07, 8.5e-07]",
    "[8.5e-07, 1.013e-06]",
    "[1.013e-06, 5.7e-06]",
    "[5.7e-06, 6.799999999999999e-06]"
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
                desc = "< 7.1699990000000006e-12 M"
            elif pos == len(intervals)+1:
                desc = "> 6.799999999999999e-06 M"
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
    print("Predicted value of Kd (dissociation constant):")
    for desc in res['descriptions']:
        print(f"{desc}")
