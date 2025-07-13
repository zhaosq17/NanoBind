import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from models.NanoBind_site import NanoBind_site
from Bio import SeqIO
import numpy as np

output_path = './output/predictions_NanoBind_site.csv'

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

def predicting(model, device):
    model.eval()

    with torch.no_grad():
        seqs_nanobody = read_fasta(args.nanobody_fasta)
        seqs_antigen = read_fasta(args.antigen_fasta)

        BSite_output = model(seqs_nanobody, seqs_antigen, device).cpu().numpy()
                
    return BSite_output

args = get_args()

# 装载训练好的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NanoBind_site(pretrained_model=r'./models/esm2_t6_8M_UR50D', hidden_size=320, finetune=0).to(device)

model_dir = './output/checkpoint/'
model_name = 'NanoBind_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_good.model'
model_path = model_dir + model_name
weights = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(weights)

p = predicting(model, device)
p = np.array(p, dtype=np.float64) 

seqs_antigen = read_fasta(args.antigen_fasta)

assert len(seqs_antigen) == len(p[0]), "序列长度与得分数组长度不一致"

binding_sites = [i + 1 for i, score in enumerate(p[0]) if score > 0.5]

print("预测的结合位点氨基酸及其位置：")
for position in binding_sites:
    amino_acid = seqs_antigen[position - 1] 
    print(f"site:{position}, amino acid:{amino_acid}")
    
    
