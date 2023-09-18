import numpy as np
import torch
from DigitFive import load_mnist, load_mnist_m, load_usps, load_svhn, load_syn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image0, train_label0, test_image0, test_label0 = load_mnist_m('./')
target_insts = test_image0[:30, :].astype(np.float32)
target_labels = test_label0[:30].ravel().astype(np.int64)
# 测试model_available
mdan = torch.load("mnist_mdan_43.160.pkl")
mdan.eval()

target_insts = torch.tensor(target_insts, requires_grad=False).to(device)
target_labels = torch.tensor(target_labels).to(device)
preds_labels = torch.max(mdan.inference(target_insts), 1)[1].data.squeeze_()
pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
print("Prediction accuracy {}". format(pred_acc))

