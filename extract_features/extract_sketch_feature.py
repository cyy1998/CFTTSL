# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
sys.path.append('../')
#sys.path.append('../mobilenet')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sketch_model import SketchModel
from classifier import Classifier
from sketch_dataset import SketchDataSet

parser = argparse.ArgumentParser("feature extraction of sketch images")
# dataset
# parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/ModelNet-Sketch/sketch_picture')
# parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/ModelNet-Sketch/12views/sketch_test_picture')
parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_sketch_picture')
parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_sketch_test_picture')
# parser.add_argument('--train-datadir', type=str, default='E:/3d_retrieval/Dataset/vis/sketch_test')
# parser.add_argument('--test-datadir', type=str, default='E:/3d_retrieval/Dataset/vis/sketch_test')
parser.add_argument('--workers', default=5, type=int,
                    help="number of data loading workers (default: 0)")

# test
parser.add_argument('--batch-size', type=int, default=1)
# parser.add_argument('--num-classes', type=int, default=171)
# parser.add_argument('--num-train-samples', type=int, default=171*30)
# parser.add_argument('--num-test-samples', type=int, default=171*30)
# parser.add_argument('--num-classes', type=int, default=15)
# parser.add_argument('--num-train-samples', type=int, default=18*30)
# parser.add_argument('--num-test-samples', type=int, default=18*30)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--num-train-samples', type=int, default=90*30)
parser.add_argument('--num-test-samples', type=int, default=90*30)

# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_false')
parser.add_argument('--model-dir', type=str, default='./alexnet_13_best')
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16','vgg19', 'resnet50'], default='alexnet')
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)
# features
parser.add_argument('--cnn-feat-dim', type=int, default=4096)
parser.add_argument('--feat-dim', type=int, default=128)
parser.add_argument('--test-feat-dir', type=str, default='alex_L2_test_sketch_feature.mat')
parser.add_argument('--train-feat-dir', type=str, default='/home/daiweidong/david/strong_baseline/sketch_modality/shrec_14/train_sketch_picture')


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
    data = SketchDataSet(root=traindir, transform=image_transforms)
    dataloaders = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
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

    sketch_model = SketchModel(args.model, args.num_classes, use_gpu=True)
    classifier = Classifier(12,args.cnn_feat_dim,args.num_classes)

    classifier1 = torch.load(args.model_dir + '/' + args.model + '_best_baseline_sketch_classifier' + '.pth')
    # classifier1 = torch.load(args.model_dir + '/' + args.model+'_baseline_classifier' + '_' + str(80) + '.pth')
    #obj = torch.load(os.path.realpath(os.path.join(args.checkpoints_base_model, "latest")))
    class_centroid = nn.functional.normalize(classifier1["module.fc5.weight"], dim=0).permute(1, 0)
    centroid=class_centroid.data.cpu().numpy()

    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    # Load model
    sketch_model.load_state_dict(torch.load(args.model_dir + '/' + args.model + '_best_baseline_sketch_model'  + '.pth'))
    classifier.load_state_dict(torch.load(args.model_dir + '/' + args.model + '_best_baseline_sketch_classifier'  + '.pth'))
    # sketch_model.load_state_dict(torch.load(args.model_dir + '/' +args.model+ '_baseline_sketch_model' + '_' +str(80) +  '.pth'))
    # classifier.load_state_dict(torch.load(args.model_dir + '/' +args.model+ '_baseline_classifier' + '_' + str(80) + '.pth'))
    sketch_model.cuda()
    classifier.cuda()
    sketch_model.eval()
    classifier.eval() 

    if args.pattern:
        num_samplses = args.num_train_samples
    else:
        num_samplses = args.num_test_samples

    # Define two matrices to store extracted features
    sketch_feature = np.zeros((num_samplses, args.feat_dim))
    sketch_labels = np.zeros((num_samplses, 1))
    predict_labels= np.zeros((num_samplses, 1))
    sketch_uncer = np.zeros((num_samplses, 1))
    dist = np.zeros((num_samplses, 1))
    paths=[]


    total = 0.0
    correct = 0.0
    for batch_idx, (data, labels,path) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        # print(batch_idx)
        output = sketch_model.forward(data)
        if args.uncer:
            mu_embeddings,logvar_embeddings,logits = classifier.forward(output)
            sigama_sq = torch.exp(0.5 * logvar_embeddings)
            print(batch_idx)
            std_sum = torch.mean(sigama_sq,dim=1)
            print(std_sum.item())
            sketch_uncer[batch_idx] = std_sum.item()
            print("++++++++++++++++++++++++++++++++++")
        else:
            mu_embeddings, logits = classifier.forward(output)
        #mu_embeddings,logits = classifier.forward(output)
        paths.append(path)

        outputs = nn.functional.normalize(mu_embeddings, dim=1)
        _,predicts=torch.max(logits.data, 1)

        class_centroid1 = class_centroid[labels]
        dist1 = (outputs - class_centroid1) * (outputs - class_centroid1)
        dist1 = torch.mean(dist1,1)
        dist1 = torch.mean(dist1)
        dist1_numpy = dist1.detach().cpu().clone().numpy()
        #print(dist1)

        #logits = classifier.forward(outputs)
        labels_numpy = labels.detach().cpu().clone().numpy()
        predicts_numpy=predicts.detach().cpu().clone().numpy()
        outputs_numpy = outputs.detach().cpu().clone().numpy()

        sketch_labels[batch_idx] = labels_numpy
        predict_labels[batch_idx]=predicts_numpy
        sketch_feature[batch_idx] = outputs_numpy
        dist[batch_idx] = dist1_numpy

        if batch_idx % 100 == 0:
            print("==> test samplses [%d/%d]" % (batch_idx, num_samplses // args.batch_size))


    sketch_feature_data = {'sketch_feature': sketch_feature, 'sketch_labels': sketch_labels,
                           'sketch_paths':paths,'predict_label':predict_labels,'class_centroid':centroid}
    torch.save(sketch_feature_data,args.test_feat_dir)
    torch.save(sketch_uncer,"sketch_uncertainty.mat")
    torch.save(dist,'baseline_dist.mat')


if __name__ == '__main__':
    main()