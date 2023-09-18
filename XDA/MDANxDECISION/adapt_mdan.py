#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import MDANet
# from model_original import MDANet as MDANet_og
from utils import get_logger
from utils import multi_data_loader
from DigitFive import load_mnist, load_svhn, load_syn
from DigitFive import digit5_dataset_read
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def resize32(domain, args):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset, test_dataset = digit5_dataset_read(domain, transform, exp = False, index=False, twice=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.mdan_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.mdan_batch_size, shuffle=True, num_workers=4)

    train_image = []
    train_label = []
    test_image = []
    test_label = []

    for data, labels in train_loader:
        train_image.append(data.numpy())
        train_label.append(labels.numpy())
    train_image = np.concatenate(train_image, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    for data, labels in test_loader:
        test_image.append(data.numpy())
        test_label.append(labels.numpy())
    test_image = np.concatenate(test_image, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    return train_image, train_label, test_image, test_label


def merge_data(args):
    data_insts = []
    data_labels= []
    testy_insts = []
    testy_labels= []
    idx = 0
    for domain in args.mdan_s:
        if domain == args.t:
            t_idx = idx
        if domain == "mnist":
            train_image, train_label, test_image, test_label = load_mnist('../../')
        elif domain == "mnistm":
            train_image, train_label, test_image, test_label = resize32('mnistm', args)
        elif domain == "svhn":
            train_image, train_label, test_image, test_label = load_svhn('../../')
        elif domain == "syn":
            train_image, train_label, test_image, test_label = load_syn('../../')
        elif domain == "usps":
            train_image, train_label, test_image, test_label = resize32('usps', args)
        data_insts.append(train_image)
        data_labels.append(train_label)
        testy_insts.append(test_image)
        testy_labels.append(test_label)
        idx += 1
    return data_insts, data_labels, testy_insts, testy_labels, t_idx

def train_mdan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(args.name)

    data_name = args.mdan_s
    num_data_sets = len(args.mdan_s)

    data_insts, data_labels, testy_insts, testy_labels , t_idx = merge_data(args)

    logger.info("Data sets: {}".format(data_name))

    # The confusion matrix stores the prediction accuracy between the source and the target tasks. The row index the source
    # task and the column index the target task.
    results = {}
    # logger.info("Training fraction = {}, number of actual training data instances = {}".format(args.frac, num_trains))
    logger.info("-" * 100)
    # input_dim = data_insts[0].shape[0]

    if args.model == "mdan":
        configs = {
        # "input_dim": 32, "hidden_layers": [1000, 500, 100], "num_classes": 2,
        "num_epochs": args.epoch, "batch_size": args.mdan_batch_size, "lr": 1e-1, "mu": args.mu, "num_domains":
                    num_data_sets - 1, "mode": args.mode, "gamma": 10.0, "verbose": args.verbose}
        num_epochs = configs["num_epochs"]
        batch_size = configs["batch_size"]
        num_domains = configs["num_domains"]
        lr = configs["lr"]
        mu = configs["mu"]
        gamma = configs["gamma"]
        mode = configs["mode"]
        logger.info("Training with domain adaptation using PyTorch madnNet: ")
        logger.info("Hyperparameter setting = {}.".format(configs))

        i = t_idx
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j].astype(np.float32))
                source_labels.append(data_labels[j].astype(np.int64))

        # Build target instances.
        target_insts = data_insts[i].astype(np.float32)

        # Build test instances.
        test_insts = testy_insts[i].astype(np.float32)
        test_labels = testy_labels[i].astype(np.int64)

        # Train DannNet.
        mdan = MDANet(configs).to(device)
        optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
        mdan.train()
        # Training phase.
        time_start = time.time()
        for t in range(num_epochs):
            running_loss = 0.0
            train_loader = multi_data_loader(source_insts, source_labels, batch_size)
            for xs, ys in train_loader:
                slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                for j in range(num_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                ridx = np.random.choice(target_insts.shape[0], batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                optimizer.zero_grad()
                
                logprobs, sdomains, tdomains = mdan(xs, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                        F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}".format(t, running_loss))
        time_end = time.time()
        # Test on other domains.
        # torch.save(mdan, './{}_mdan_{:.3f}.pkl'.format(data_name[i], running_loss))
        mdan.eval()
        test_insts = torch.tensor(test_insts, requires_grad=False).to(device)
        test_labels = torch.tensor(test_labels).cpu()
        all_outputs = mdan.inference(test_insts)
        preds_labels = torch.max(all_outputs, 1)[1].cpu().data.squeeze_()
        pred_acc = torch.sum(preds_labels == test_labels).item() / float(test_insts.size(0))
        logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".
                    format(data_name[i], pred_acc, time_end - time_start))
        return all_outputs
    else:
        raise ValueError("No support for the following model: {}.".format(args.model))

