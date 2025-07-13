import torch 
import argparse
from torch.nn import functional as F
from models.NanoBind_pair import NanoBind_pair
from Bio import SeqIO

def get_args():
    parser = argparse.ArgumentParser(description='Demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nanobody_fasta1', dest='nanobody_fasta1', help='Path to the nanobody1 FASTA file')    
    parser.add_argument('--antigen_fasta1', dest='antigen_fasta1', help='Path to the antigen1 FASTA file')   
    parser.add_argument('--nanobody_fasta2', dest='nanobody_fasta2', help='Path to the nanobody2 FASTA file')    
    parser.add_argument('--antigen_fasta2', dest='antigen_fasta2', help='Path to the antigen2 FASTA file') 
    
    return parser.parse_args()

def read_fasta(fasta_path):
    # 读取FASTA文件并返回第一个序列
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError("No sequences found in the FASTA file.")
    return str(records[0].seq)
args = get_args()

def predicting(model, device):
    model.eval()
    total_preds = torch.Tensor()

    with torch.no_grad():
            ab1 = read_fasta(args.nanobody_fasta1)
            ag1 = read_fasta(args.antigen_fasta1)
            ab2 = read_fasta(args.nanobody_fasta2)
            ag2 = read_fasta(args.antigen_fasta2)
            #Calculate output
            p=model(ab1,ag1,ab2,ag2,device)            
            total_preds = torch.cat((total_preds, p.cpu()), 0)

    return total_preds.numpy().flatten()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=NanoBind_pair()
model = model.to(device)
model_path = './output/checkpoint/' + 'NanoBind_pair_random.model'
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

p = predicting(model, device)

for item in p:
    if item>0.2:
        pred=1
        print('The predicted Kd value of Pair 1 is larger.')
    else:
        pred=0
        print('The predicted Kd value of Pair 2 is larger.')

