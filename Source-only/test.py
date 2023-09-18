from model import CNN, Classifier
import numpy as np
import os
from os import path
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from DigitFive import digit5_dataset_read

base_path = './'
test_d = 'syn'

if __name__ == '__main__':
    torch.cuda.init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256

    classifier = torch.load(path.join(base_path, "models", "syn_classifier_combined_SGD_0.812.pkl"))
    encoder = torch.load(path.join(base_path, "models", "syn_encoder_combined_SGD_0.812.pkl"))

    all_correct_num = 0
    all_sample_num = 0
    encoder.eval()

    train_loader, val_loader, test_loader = digit5_dataset_read(base_path, test_d, batch_size)

    acc_sum = 0
    for i in range(5):
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = classifier(encoder(test_x.float()).detach())
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc_sum += all_correct_num / all_sample_num
    acc = acc_sum / 5
    print('accuracy: {:.3f}'.format(acc), flush=True)
