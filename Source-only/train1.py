from model import CNN, Classifier
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from DigitFive import digit5_dataset_read
from itertools import cycle


base_path = './'
train_d = 'mnistm'
opt = 'SGD'

if __name__ == '__main__':
    torch.cuda.init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_loader, val_loader, test_loader= digit5_dataset_read(base_path, train_d, batch_size, exp = False)
    # train_loader, val_loader, test_loader = digit5_dataset_read(base_path, train_d, batch_size, exp = True)
    encoder = CNN().to(device)
    classifier = Classifier().to(device)
    
    if opt == 'SGD':
        opt = SGD(encoder.parameters(), lr=1e-1, momentum = 0.9)
    elif opt == 'Adam':
        opt=torch.optim.Adam(encoder.parameters(), lr=1e-3, betas=(0.9,0.999))
    elif opt == 'Adadelta':
        opt = optim.Adadelta(encoder.parameters(), lr=1)

    
    
    loss_fn = CrossEntropyLoss()
    all_epoch =40
    prev_acc = 0
    for current_epoch in range(all_epoch):
        encoder.train()
        classifier.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            opt.zero_grad()
            feature = encoder(train_x.float())
            predict_y = classifier(feature.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            opt.step()

        all_correct_num = 0
        all_sample_num = 0
        encoder.eval()
        
        for idx, (test_x, test_label) in enumerate(val_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = classifier(encoder(test_x.float()).detach())
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        if (acc - prev_acc) < 1e-4:
            break
        torch.save(encoder, './models/{}_encoder_combined_SGD_{:.3f}.pkl'.format(train_d, acc))
        torch.save(classifier, './models/{}_classifier_combined_SGD_{:.3f}.pkl'.format(train_d, acc))
        prev_acc = acc
    print("Model finished training")
    