import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import logging
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
from models.NanoBind_seq import NanoBind_seq
from models.NanoBind_pro import NanoBind_pro
from utils.evaluate import evaluate

ESM2_MODEL = 'esm2_t6_8M_UR50D'

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
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class seqData_GST(Dataset):
    def __init__(self,):
        super(seqData_GST,self).__init__()
        
        self.seq_data = list()
        
        # Load pair data
        data = pd.read_csv('./data/case_study/GST.csv').values.tolist()       
        # Antigen: GST
        seq2 = 'MGPLPRTVELFYDVLSPYSWLGFEILCRYQNIWNINLQLRPSLITGIMKDSGNKPPGLLPRKGLYMANDLKLLRHHLQIPIHFPKDFLSVMLEKGSLSAMRFLTAVNLEHPEMLEKASRELWMRVWSRNEDITEPQSILAAAEKAGMSAEQAQGLLEKIATPKVKNQLKETTEAACRYGAFGLPITVAHVDGQTHMLFGSDRMELLAHLLGEKWMGPIPPAVNARL'
        
        for n,item in enumerate(data):   
            seq1 = item[0]
            elisa = item[1]

            if '/' in elisa:
                continue
            if 'No binding' in elisa:
                elisa = 0
            else:
                elisa = float(elisa)
                        
            if len(seq1)>800 or len(seq2)>800:
                continue
            
            self.seq_data.append([seq1,seq2,elisa])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label
print(len(seqData_GST()))

print('##########################在GST数据上测试NanoBind_seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
# 装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0).to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = NanoBind_seq(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0).to(device)

model_dir = './output/checkpoint/'
model_name = 'NanoBind_seq({})_SabdabData_finetune1_TF0_good.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
model.load_state_dict(weights)

testDataset = seqData_GST()
test_loader = DataLoader(testDataset,batch_size=16, shuffle=False)

# Test
g,p1 = predicting(model, device, test_loader,Model_type=3)
np.save('./output/results_NanoBind_seq(SabdabData)_{}_{}.npy'.format('GST',ESM2_MODEL),[g,p1])
max_p1 = np.max(p1)
print("Maximum value of p1: {:.4f}".format(max_p1))

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR ,mcc= evaluate((g>0)+0,p1,thresh=0.3)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f},MCC={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR,mcc))

# Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))



print('##########################在GST数据上测试NanoBind_pro(SabdabData)_{}模型：'.format(ESM2_MODEL))
# 装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t12_35M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t30_150M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t33_650M_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t36_3B_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = NanoBind_pro(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0,
                        Model_BSite_path='./output/checkpoint/NanoBind_site(esm2_t48_15B_UR50D)_SabdabData_finetune1_TF0_good.model').to(device)

model_dir = './output/checkpoint/'
model_name = 'NanoBind_pro({})_SabdabData_finetune1_TF0_good.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
model.load_state_dict(weights)

testDataset = seqData_GST()
test_loader = DataLoader(testDataset,batch_size=16, shuffle=False)

#Test
g,p1 = predicting(model, device, test_loader,Model_type=3)
np.save('./output/results_NanoBind_pro(SabdabData)_{}_{}.npy'.format('GST',ESM2_MODEL),[g,p1])
max_p1 = np.max(p1)
print("Maximum value of p1: {:.4f}".format(max_p1))

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR ,mcc= evaluate((g>0)+0,p1,thresh=0.5)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f},MCC={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR,mcc))

#Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))
