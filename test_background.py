import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import logging
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
from Bio import SeqIO
from models.NanoBind_seq import NanoBind_seq
from models.NanoBind_pro import NanoBind_pro

ESM2_MODEL = 'esm2_t6_8M_UR50D'

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--size', dest='size', type=str, default='100w',
                        help='size',metavar='E')

    return parser.parse_args()

def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds_ave = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        # for data in tqdm(loader):
        for data in loader:
            #Get input
            seqs_nanobody = data[0]
            seqs_antigen = data[1]

            #Calculate output
            p = model(seqs_nanobody,seqs_antigen,device)
            
            total_preds_ave = torch.cat((total_preds_ave, p.cpu()), 0)
            
            g = data[2]
            total_labels = torch.cat((total_labels, g), 0)

    return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten()
   
#装载训练好的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class seqData_background(Dataset):
    def __init__(self,antigen_seq):
        super(seqData_background,self).__init__()
        
        self.seq_data = list()
        
        #Load sequence data
        nb_backgound = list()
        for fa in SeqIO.parse('./data/INDI/INDI_{}_nanobody.fasta'.format(DATA_SIZE),'fasta'):
            seq = ''.join(list(fa.seq))
            nb_backgound.append(seq)

        seq2 = antigen_seq
        for n,item in enumerate(nb_backgound):
            seq1 = item
            if len(seq1)>150:
                continue

            self.seq_data.append([seq1,seq2,0])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label

args = get_args()
DATA_SIZE = args.size

print('##########################在GST background数据上测试NanoBind_seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
#装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0).to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model_seq = NanoBind_seq(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0).to(device)
model_dir = './output/checkpoint/'
model_name = 'NanoBind_seq({})_SabdabData_finetune1_TF0_good.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu'))
model_seq.load_state_dict(weights)
 
#装载测试数据background
testDataset = seqData_background(antigen_seq='MGPLPRTVELFYDVLSPYSWLGFEILCRYQNIWNINLQLRPSLITGIMKDSGNKPPGLLPRKGLYMANDLKLLRHHLQIPIHFPKDFLSVMLEKGSLSAMRFLTAVNLEHPEMLEKASRELWMRVWSRNEDITEPQSILAAAEKAGMSAEQAQGLLEKIATPKVKNQLKETTEAACRYGAFGLPITVAHVDGQTHMLFGSDRMELLAHLLGEKWMGPIPPAVNARL')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
 
g_seq, p_seq = predicting(model_seq, device, test_loader, Model_type=3)
count_seq = np.sum(p_seq > 0.2263)
print('NanoBind_seq模型预测得分>0.2263的样本数量: {}'.format(count_seq))
 
results_seq = np.column_stack((g_seq, p_seq))
np.savetxt('./output/results_NanoBind_seq(SabdabData)_{}_{}.txt'.format('background({})_GST'.format(DATA_SIZE),ESM2_MODEL), 
           results_seq, fmt='%s', delimiter='\t')


print('##########################在GST background数据上测试NanoBind_pro(SabdabData)_{}模型：'.format(ESM2_MODEL))
#装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t12_35M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t30_150M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t33_650M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t36_3B_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model_nano = NanoBind_pro(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t48_15B_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)

model_dir = './output/checkpoint/'
model_name = 'NanoBind_pro({_SabdabData_finetune1_TF0_good.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu'))
model_nano.load_state_dict(weights)

#装载测试数据background
testDataset = seqData_background(antigen_seq='MGPLPRTVELFYDVLSPYSWLGFEILCRYQNIWNINLQLRPSLITGIMKDSGNKPPGLLPRKGLYMANDLKLLRHHLQIPIHFPKDFLSVMLEKGSLSAMRFLTAVNLEHPEMLEKASRELWMRVWSRNEDITEPQSILAAAEKAGMSAEQAQGLLEKIATPKVKNQLKETTEAACRYGAFGLPITVAHVDGQTHMLFGSDRMELLAHLLGEKWMGPIPPAVNARL')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)

g_pro, p_pro = predicting(model_nano, device, test_loader, Model_type=3)
count_pro = np.sum(p_pro > 0.5630)
print('NanoBind_pro模型预测得分>0.5630的样本数量: {}'.format(count_pro))

results_pro = np.column_stack((g_pro, p_pro))
np.savetxt('./output/results_NanoBind_pro(SabdabData)_{}_{}.txt'.format('background({})_GST'.format(DATA_SIZE),ESM2_MODEL), 
           results_pro, fmt='%s', delimiter='\t')

assert len(p_seq) == len(p_pro), "两个模型的预测结果长度不一致"
 
# 计算同时满足最大分数的样本数量
both_conditions = (p_seq > 0.2263) & (p_pro > 0.5630)
count_both = np.sum(both_conditions)
print('同时满足NanoBind_seq和NanoBind_pro的样本数量: {}'.format(count_both))


