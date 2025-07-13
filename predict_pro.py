import torch
from Bio import SeqIO
import argparse
import warnings
warnings.filterwarnings("ignore")
from models.NanoBind_pro import NanoBind_pro
import torch

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

def predicting(model, device, Model_type):
    model.eval()
    total_preds = torch.Tensor()

    with torch.no_grad():
            seqs_nanobody = read_fasta(args.nanobody_fasta)
            seqs_antigen = read_fasta(args.antigen_fasta)

            p = model(seqs_nanobody,seqs_antigen,device)
            total_preds = torch.cat((total_preds, p.cpu()), 0)
            
    return total_preds.numpy().flatten()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NanoBind_pro(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
model_dir = './output/checkpoint/'
model_name = 'NanoBind_pro(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_good.model'
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

#Test
p = predicting(model, device, Model_type=3)

for item in p:
    if item>0.5:
        pred=1
    else:
        pred=0
print(pred)