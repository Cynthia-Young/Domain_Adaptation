#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn1", nn.BatchNorm2d(64))
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn2", nn.BatchNorm2d(64))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn3", nn.BatchNorm2d(128))
        encoder.add_module("relu3", nn.ReLU())
        self.encoder = encoder

        linear = nn.Sequential()
        linear.add_module("fc1", nn.Linear(8192, 3072))
        linear.add_module("bn4", nn.BatchNorm1d(3072))
        linear.add_module("relu4", nn.ReLU())
        linear.add_module("dropout", nn.Dropout())
        linear.add_module("fc2", nn.Linear(3072, 2048))
        linear.add_module("bn5", nn.BatchNorm1d(2048))
        linear.add_module("relu5", nn.ReLU())
        self.linear = linear

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        print("Encoder Output Shape:", feature.shape)
        feature = feature.view(batch_size, 8192)
        feature = self.linear(feature)
        print("Linear Output Shape:", feature.shape)
        return feature


class Classifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(Classifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(2048, 10))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):
        x = self.linear(x)
        print("Classifier Output Shape:", x.shape)
        return x


# class GradientReversalLayer(torch.autograd.Function):
#     """
#     Implement the gradient reversal layer for the convenience of domain adaptation neural network.
#     The forward part is the identity function while the backward part is the negative function.
#     """
#     def forward(self, inputs):
#         return inputs

#     def backward(self, grad_output):
#         grad_input = grad_output.clone()
#         grad_input = -grad_input
#         return grad_input

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x)

class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(MDANet, self).__init__()
        self.num_domains = configs["num_domains"]

        # Feature extraction layers from the CNN class.
        self.encoder = CNN().encoder
        self.linear = CNN().linear

        # Parameters of the final classifier layer.
        self.classifier = Classifier(data_parallel=False).linear

        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(2048, 2) for _ in range(self.num_domains)])

        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]

    def forward(self, sinputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        # Feature extraction using the encoder.
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            batch_size_s = sh_relu[i].size(0)
            # print(sh_relu[i].shape)
            sh_relu[i] = self.encoder(sh_relu[i])
            # print(sh_relu[i].shape)
            sh_relu[i] = sh_relu[i].view(batch_size_s, 8192)

        batch_size_t = th_relu.size(0)
        th_relu = self.encoder(th_relu)
        th_relu = th_relu.view(batch_size_t, 8192)


        # print("sh_relu Encoder Output Shape:", sh_relu[0].shape)
        # print("th_relu Encoder Output Shape:", th_relu[0].shape)

        
        for i in range(self.num_domains):
            sh_relu[i] = self.linear(sh_relu[i])
        th_relu = self.linear(th_relu)
        # print("sh_relu Linear Output Shape:", sh_relu[0].shape)
        # print("th_relu Linear Output Shape:", th_relu[0].shape)

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.classifier(sh_relu[i]), dim=1))

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        # Feature extraction using the encoder.
        batch_size = inputs.size(0)
        h_relu = self.encoder(inputs)
        h_relu = h_relu.view(batch_size, 8192)
        h_relu = self.linear(h_relu)


        # Classification probability.
        logprobs = F.log_softmax(self.classifier(h_relu), dim=1)
        return logprobs

