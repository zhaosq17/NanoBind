import torch
import numpy as np
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from utils.evaluate import  evaluate_site 
from utils.dataloader import infaData_Sabdab,collate_fn_infaData 
from models.NanoBind_site import *

ESM2_MODEL = 'esm2_t6_8M_UR50D'

def test(model, device, loader):
    model.eval()
    total_BSite2 = list()
    total_labels = list()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:

            seqs_nanobody = data[0]
            seqs_antigen = data[1]
            BSite_antigen = data[3]

            BSite_output = model(seqs_nanobody, seqs_antigen, device).cpu().numpy().tolist()

            for n in range(len(seqs_antigen)):
                len_seq = len(seqs_antigen[n])
                if len_seq > len(BSite_output[n]):
                    len_seq = len(BSite_output[n])
                total_BSite2.append(BSite_output[n][:len_seq])
                total_labels.append(BSite_antigen[n][:len_seq])

    return total_labels, total_BSite2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('##########################在nai数据上测试NanoBind_site {} 模型：'.format(ESM2_MODEL))

if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0).to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = NanoBind_site(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0).to(device)

model_dir = './output/checkpoint/'
model_name = 'NanoBind_site({})_SabdabData_finetune1_TF0_good.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

testDataset = infaData_Sabdab(pair_path='./data/Sabdab/NAI_test_pos.csv')
test_loader = DataLoader(testDataset, batch_size=32, shuffle=False,collate_fn=collate_fn_infaData)
print(len(testDataset))

#test
true_labels, pred_scores = test(model, device, test_loader)

accuracy, precision, recall, F1_score, auroc, auprc ,mcc= evaluate_site(true_labels, pred_scores, thresh=0.5)
print(f"F1={F1_score:.4f},Accuracy={accuracy:.4f},Precision={precision:.4f},Recall={recall:.4f},AUROC={auroc:.4f},AUPRC={auprc:.4f},MCC={mcc:.4f}")

threshold_range = np.arange(0, 1.1, 0.1)
best_f1 = 0
best_threshold = 0

for threshold in threshold_range:
    _, _, _, f1, _, _ ,_= evaluate_site(true_labels,pred_scores, thresh=threshold)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
a, pe, re, f1, rc, pr ,mc= evaluate_site(true_labels,pred_scores, thresh=best_threshold)

print(f"Best Threshold: {best_threshold:.2f}, Best_F1={best_f1:.4f},Accuracy={a:.4f},Precision={pe:.4f},Recall={re:.4f},AUROC={rc:.4f},AUPRC={pr:.4f},mcc={mc:.4f}")