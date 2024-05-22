import os
import sys
import argparse
import numpy as np
import scipy.io as sio
sys.path.append('/')
#sys.path.append('../mobilenet')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sketch_model import SketchModel
from view_model import MVCNN
from view_dataset_reader import MultiViewDataSet
from classifier import Classifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from get_mu_logvar import Get_mu_logvar

from PIL import Image
import os
import shutil

correct_path='E:\\3d_retrieval\\Dataset\\Shrec_13\\correct\\13_view_render_img\\'
wrong_path='E:\\3d_retrieval\\Dataset\\Shrec_13\\wrong\\13_view_render_img\\'

parser = argparse.ArgumentParser("feature extraction of sketch images")

parser.add_argument('--sketch_path', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_sketch_test_picture/airplane/11.png')
parser.add_argument('--view-train-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_train_sketch_picture')
parser.add_argument('--view-test-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec_13/12_views/13_view_render_img')
parser.add_argument('--workers', default=5, type=int,
                    help="number of data loading workers (default: 0)")

parser.add_argument('--batch-size', type=int, default=1)
# parser.add_argument('--num-classes', type=int, default=171)
# parser.add_argument('--num-train-samples', type=int, default=171*30)
# parser.add_argument('--num-test-samples', type=int, default=171*30)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--num-train-samples', type=int, default=90*30)
parser.add_argument('--num-test-samples', type=int, default=90*30)

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str, default='./alexnet_13')
parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16','vgg19', 'resnet50'], default='alexnet')
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)

parser.add_argument('--cnn-feat-dim', type=int, default=4096)
parser.add_argument('--feat-dim', type=int, default=128)

parser.add_argument('--test-sketch-feat-file', type=str,
                    default='./extract_features/alex_L2_test_sketch_feature.mat',
                    help="features flie of test sketches, .mat file")
parser.add_argument('--view-feat-flie', type=str,
                    default='./extract_features/alex_L2_test_view_feature.mat',
                    help="features flie of view images of 3d models, .mat file")

args = parser.parse_args()

def get_view_data(view_feat_flie):
    """" read the features and labels of sketches and 3D models
    Args:
        test_sketch_feat_file: features flie of test sketches, it is .mat file
        view_feat_flie: features flie of view images of 3d models
    """
    view_data_features1 = torch.load(view_feat_flie)

    """
    sketch_feature = sket_data_features['view_feature']
    print(sketch_feature.shape)
    sketch_label = sket_data_features['view_labels']
    """

    view_feature = view_data_features1['view_feature']
    view_label = view_data_features1['view_labels']
    view_paths=view_data_features1['view_paths']


    return view_feature, view_label,view_paths

def cal_euc_distance(sketch_feat,view_feat):
    distance_matrix = pairwise_distances(sketch_feat,view_feat)

    return distance_matrix

def cal_cosine_distance(sketch_feat,view_feat):
    distance_matrix = cosine_similarity(sketch_feat,view_feat)

    return distance_matrix

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def main():
    view_feature, view_label,view_paths = get_view_data(args.view_feat_flie)
    classes,class_to_idx=find_classes("E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_sketch_test_picture")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        # torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    sketch_model = SketchModel(args.model, args.num_classes, use_gpu=True)
    classifier = Classifier(12, args.cnn_feat_dim, args.num_classes)
    classifier1 = torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(80) + '.pth')
    class_centroid = nn.functional.normalize(classifier1["module.fc5.weight"], dim=0).permute(1, 0)
    class_centroid = class_centroid.data.cpu().numpy()
    if use_gpu:
        sketch_model = nn.DataParallel(sketch_model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    # Load model
    sketch_model.load_state_dict(torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_sketch_model' + '_' + str(80) + '.pth'))
    classifier.load_state_dict(torch.load('./extract_features/'+args.model_dir + '/' + args.model + '_baseline_classifier' + '_' + str(80) + '.pth'))

    sketch_model.cuda()
    classifier.cuda()
    sketch_model.eval()
    classifier.eval()

    sketch_label=class_to_idx[args.sketch_path.split('/')[-2]]
    sketch_id=args.sketch_path.split('/')[-1].split('.')[0]
    im = Image.open(args.sketch_path)
    im = im.convert('RGB')
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])
    data=image_transforms(im)
    data=data.unsqueeze(dim=0)
    if use_gpu:
        data= data.cuda()
    # print(batch_idx)
    output = sketch_model.forward(data)
    mu_embeddings, logits = classifier.forward(output)
    outputs = nn.functional.normalize(mu_embeddings, dim=1)
    _, predicted = torch.max(logits.data, 1)
    print("Predict Label: "+str(predicted.item()))


    sketch_feat = outputs.detach().cpu().clone().numpy()
    distance_matrix = cal_cosine_distance(sketch_feat, view_feature)
    dist_sort_index = np.argsort(-distance_matrix[0], axis=0)
    top_10_retrieval=dist_sort_index[:10]
    print("Top 10 result: "+str(top_10_retrieval))
    print("Top 10 result label: " + str(view_label[top_10_retrieval].reshape(10,)))
    print("Top 10 cosine similarity: "+str(distance_matrix[0][top_10_retrieval]))
    if not os.path.exists('retrieval_result/' + sketch_id):
        os.mkdir('retrieval_result/' + sketch_id)
    for i in top_10_retrieval:
        if view_label[i][0]==sketch_label:
            file=os.path.join(correct_path,view_paths[i],os.listdir(correct_path+view_paths[i])[0])
            shutil.copy(file,'retrieval_result/' + sketch_id)
        else:
            file = os.path.join(wrong_path, view_paths[i], os.listdir(wrong_path + view_paths[i])[0])
            shutil.copy(file, 'retrieval_result/' + sketch_id)


if __name__=='__main__':
    main()