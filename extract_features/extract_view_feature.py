# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
sys.path.append('../')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from view_model import MVCNN
from view_dataset_reader import MultiViewDataSet
from classifier import Classifier

parser = argparse.ArgumentParser("feature extraction of sketch images")
# dataset
# parser.add_argument('--train-datadir', type=str, default='train_sketch_picture')
# parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/ModelNet-Sketch/12views/train_sketch_picture')
# parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/ModelNet-Sketch/4views/test_view_render_img')
parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_train_sketch_picture')
parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_view_render_img')
# parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/vis/view')
# parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/vis/view')
parser.add_argument('--workers', default=5, type=int,
                    help="number of data loading workers (default: 0)")

# test
parser.add_argument('--batch-size', type=int, default=1)
# parser.add_argument('--num-classes', type=int, default=171)
# parser.add_argument('--num-train-samples', type=int, default=8987)
# parser.add_argument('--num-test-samples', type=int, default=8987)
# parser.add_argument('--num-classes', type=int, default=15)
# parser.add_argument('--num-train-samples', type=int, default=300)
# parser.add_argument('--num-test-samples', type=int, default=300)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--num-train-samples', type=int, default=1258)
parser.add_argument('--num-test-samples', type=int, default=1258)
# parser.add_argument('--num-train-samples', type=int, default=80*13+79+64)
# parser.add_argument('--num-test-samples', type=int, default=80*13+79+64)


# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_false')
parser.add_argument('--model-dir', type=str,choices=['/data/david/sketch_uncertainty/RES','resnet50'], default='./alexnet_13_best')
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16','vgg19', 'resnet50'], default='alexnet')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)
# features
parser.add_argument('--cnn-feat-dim', type=int, default=4096)
parser.add_argument('--feat-dim', type=int, default=128)
parser.add_argument('--test-feat-dir', type=str, default='alex_L2_test_view_feature.mat')
parser.add_argument('--train-feat-dir', type=str, default='../features/sketch_features/t_sketch_feature.mat')


parser.add_argument('--pattern', type=bool, default=False,
                    help="Extract training data features or test data features,'True' is for train dataset")

args = parser.parse_args()


def get_test_data(traindir):
    """Image reading, but no image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])  # Imagenet standards
    view_data = MultiViewDataSet(traindir, transform=image_transforms)
    dataloaders = DataLoader(view_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return dataloaders


def main():
    # torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        # torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    if args.pattern:
        trainloader = get_test_data(args.train_datadir)
    else:
        trainloader = get_test_data(args.test_datadir)

    view_model = MVCNN(args.model,args.num_classes, use_gpu=True)
    classifier = Classifier(35,args.cnn_feat_dim,args.num_classes)
    if use_gpu:
        view_model = nn.DataParallel(view_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()
    # Load model
    view_model.load_state_dict(
        torch.load(args.model_dir + '/' + args.model + '_baseline_view_model_80'  + '.pth'))
    classifier.load_state_dict(
        torch.load(args.model_dir + '/' + args.model + '_baseline_view_classifier_80'  + '.pth'))
    # view_model.load_state_dict(torch.load(args.model_dir + '/' + args.model+'_baseline_view_model' + '_' + str(80) + '.pth'))
    # classifier.load_state_dict(torch.load(args.model_dir + '/' + args.model+'_baseline_classifier' + '_' + str(80) + '.pth'))

    view_model.cuda()
    classifier.cuda()
    view_model.eval()
    classifier.eval()

    if args.pattern:
        num_samplses = args.num_train_samples
    else:
        num_samplses = args.num_test_samples

    # Define two matrices to store extracted features
    view_feature = np.zeros((num_samplses, args.feat_dim))
    view_labels = np.zeros((num_samplses, 1))
    predict_labels = np.zeros((num_samplses, 1))
    paths=[]


    total = 0.0
    correct = 0.0

    for batch_idx, (data, labels,path) in enumerate(trainloader):
        path=path[0].split('/')[-2]+'/'+path[0].split('/')[-1]
        data = np.stack(data, axis=1)
        data = torch.from_numpy(data)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        #print(labels)
        # print(batch_idx)
        output = view_model.forward(data)
        if args.uncer:
            x2,logvar_embeddings,logits = classifier.forward(output)
        else:
            x2, logits = classifier.forward(output)
        #mu_embeddings, x2, logvar_embeddings, logits, concat_labels = classifier.forward(output,output,labels,labels)

        outputs = nn.functional.normalize(x2, dim=1)
        _, predicts = torch.max(logits.data, 1)

        labels_numpy = labels.detach().cpu().clone().numpy()
        predicts_numpy = predicts.detach().cpu().clone().numpy()
        outputs_numpy = outputs.detach().cpu().clone().numpy()

        view_labels[batch_idx] = labels_numpy
        predict_labels[batch_idx] = predicts_numpy
        view_feature[batch_idx] = outputs_numpy
        paths.append(path)
        # print(path[0][0].split('/')[-1][:-5])
        # paths.append(path[0][0].split('/')[-1][:-5])

        if batch_idx % 100 == 0:
            print("==> test samplses [%d/%d]" % (batch_idx, num_samplses // args.batch_size))


    view_feature_data = {'view_feature': view_feature, 'view_labels': view_labels,
                         'predict_label':predict_labels,'view_paths':paths}
    torch.save(view_feature_data,args.test_feat_dir)


if __name__ == '__main__':
    main()