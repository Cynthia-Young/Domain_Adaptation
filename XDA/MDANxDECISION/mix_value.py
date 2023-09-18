import torch
import torch.nn.functional as F
import argparse
import os
import os.path as osp
import numpy as np
import torch
import random
from adapt_decision import train_target
from adapt_mdan import train_mdan
from DigitFive import load_mnist, load_mnist_m, load_usps, load_svhn, load_syn



if __name__ == "__main__":
    #DECISION PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # parser.add_argument('--t', type=int, default=0, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=40, help="max DECIISON iterations") #40
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='digit', choices=['office', 'office-home', 'office-caltech', 'digit'])
    parser.add_argument('--lr', type=float, default=1*1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='cnn', help="vgg16, resnet50, res101, cnn")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/adapt_free')
    parser.add_argument('--output_src', type=str, default='ckps/source_free')
    
    #MDAN PARSER
    parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="digit")
    parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                        type=float, default=1.0)
    parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                                "not show training progress.", type=bool, default=True)
    parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                        type=str, default="mdan")

    parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                        type=float, default=1e-2)
    parser.add_argument("-e", "--epoch", help="Number of MDAN training epochs", type=int, default=20) #old 15
    parser.add_argument("-b", "--mdan_batch_size", help="Batch size during training MDAN", type=int, default=100) #old 20
    parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr' , 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'digit':
        names = ['mnist', 'mnistm', 'usps', 'svhn', 'syn']
        args.class_num = 10

    args.src = ['svhn'] #不包含目标域
    args.mdan_s = ['usps', 'mnistm', 'syn', 'mnist'] #要包含目标域
    args.t = 'mnist' #目标域
    # for i in range(len(names)):
    #     if i == args.t:
    #         continue
    #     else:
    #         args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.t_domain = args.t

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i].upper()))
    args.output_dir = osp.join(args.output, args.dset, args.t.upper())

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)

    preds_b = train_target(args)

    preds_a = train_mdan(args)

    if args.t == "mnist":
        train_image, train_label, test_image, test_label = load_mnist('../../')
    elif args.t == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m('../../')
    elif args.t == "svhn":
        train_image, train_label, test_image, test_label = load_svhn('../../')
    elif args.t == "syn":
        train_image, train_label, test_image, test_label = load_syn('../../')
    elif args.t == "usps":
        train_image, train_label, test_image, test_label = load_usps('../../')
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(args.t))

    weighted_preds = (0.5 * F.softmax(preds_a.to("cpu"), dim=1)) + (0.5 * F.softmax(preds_b.to("cpu"), dim=1))
    predicted_labels = torch.argmax(weighted_preds, dim=1)
    test_label = torch.tensor(test_label).cpu()
    accuracy = torch.sum(torch.squeeze(predicted_labels).float() == test_label).item() / float(test_label.size()[0])
    print("Prediction accuracy on {} = {}".format(args.t, accuracy))