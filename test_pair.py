import numpy as np
import torch 
from models.NanoBind_pair import NanoBind_pair
from utils.dataloader import seq_affinity
from utils.evaluate import evaluate_aff
from torch.utils.data import DataLoader

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        # for data in tqdm(loader):
        for data in loader:
            #Get input
            ab1 = data[0]
            ag1 = data[1]
            ab2 = data[2]
            ag2 = data[3]

            #Calculate output
            p=model(ab1,ag1,ab2,ag2,device)            
            total_preds = torch.cat((total_preds, p.cpu()), 0)
            
            g = data[4]
            total_labels = torch.cat((total_labels, g), 0)

    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=NanoBind_pair()
model = model.to(device)
model_path = './output/checkpoint/' + 'NanoBind_pair_random.model'
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

testDataset = seq_affinity('./data/affinity/test_random.csv')
test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)

g,p = predicting(model, device, test_loader)
precision,recall,accuracy,F1_score,AUC_ROC,AUC_PR,mcc = evaluate_aff(g,p,thresh=0.5)
print(f"Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f},F1={F1_score:.4f}, AUROC={AUC_ROC:.4f}, AUPRC={AUC_PR:.4f},MCC={mcc:.4f}")

threshold_range = np.arange(0, 1.1, 0.1)
best_f1 = 0
best_threshold = 0

for threshold in threshold_range:
    _, _, _, f1, _, _, _= evaluate_aff(g, p, thresh=threshold)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
pe, re, a, f1, rc, pr,mc = evaluate_aff(g, p, thresh=best_threshold)
print(f"Best Threshold={best_threshold:.2f},Best F1={best_f1:.4f},accuracy={a:.4f},precision={pe:.4f},recall={re:.4f},auroc={rc:.4f},auprc={pr:.4f},mcc={mc:.4f}")
