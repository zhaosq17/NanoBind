import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import warnings
warnings.filterwarnings("ignore")
from models.NanoBind_seq import NanoBind_seq
from utils.dataloader import seqData_NBAT_Test,seqData_Sabdab
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
            g = data[2]
            if Model_type == 0:
                predictions = model(seqs_nanobody,seqs_antigen,device)

                total_preds_ave = torch.cat((total_preds_ave, predictions.cpu()), 0)
            elif Model_type == 1:
                p = model(seqs_nanobody,seqs_antigen,device)
                
                total_preds_ave = torch.cat((total_preds_ave, p.cpu()), 0)

            total_labels = torch.cat((total_labels, g), 0)
                       
    if Model_type == 1:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten()
    else:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('##########################在nai数据上测试NanoBind_seq(SabdabData){}模型：'.format(ESM2_MODEL))

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
weights = torch.load(model_path,map_location=torch.device('cpu')) # 
model.load_state_dict(weights)

testDataset  = seqData_NBAT_Test(seq_path='./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta',
                                pair_path = './data/Nanobody_Antigen-main/all_pair_data.pair.tsv')
test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)
print(len(testDataset))

#Test
g,p= predicting(model, device, test_loader,Model_type=1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR,mcc = evaluate(g,p,thresh=0.3)
print("{}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f},mcc{:.4f}".format(
    'Ensemble',Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR,mcc))

threshold_range = np.arange(0, 1.1, 0.1)

best_f1 = 0
best_threshold = 0

for threshold in threshold_range:
    _, _, _, f1, _, _, _, _, _ ,_= evaluate(g, p, thresh=threshold)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
pe, re, a, f1, t, to, top, rc, pr,mc = evaluate(g, p, thresh=best_threshold)

print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.4f},precision:{pe:.4f},recall:{re:.4f},accuracy:{a:.4f},auroc:{rc:.4f},auprc:{pr:.4f},mcc:{mc:.4f}")







