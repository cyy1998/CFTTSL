# CFTTSL
Official repository of Sketch-Based 3D shape Retrieval via Teacher-Student Learning


## Abstract
One of the main difficulties of sketch-based 3D shape retrieval is the significant cross-modal difference between 2D sketches and 3D shapes. Most previous works adopt one-stage methods to directly learn the aligned common embedding space of sketches and shapes by a shared classifier. However, the intra-class difference of the sketch is more significant than the shape, harming the feature learning of 3D shapes when the two modalities are considered under the shared classifier. This issue harms the discrimination of the learned common embedding space. This paper proposes a novel two-stage method to learn a common aligned embedding space via teacher-student learning to address the issue. Speciﬁcally, we first employ a classification network to learn the discriminative features of shapes. The learned shape features are considered a teacher to guide the feature learning of sketches. Moreover, we design a guidance loss to achieve the feature transfer with semantic alignment. The proposed method achieves an effective, aligned cross-modal embedding space. Experiments on three public benchmark datasets prove the superiority of the proposed method over state-of-the-art methods.

## Architecture
![pipeline1](https://github.com/cyy1998/CFTTSL/assets/37933688/6246ddff-9bda-4cbd-b51e-bc14169be84d)
The overall architecture of the proposed uncertainty-aware cross-modal transfer network (UACTN) for SBSR is illustrated. We decouple the task of cross-modal matching between sketches and 3D shapes into two separate learning tasks: (1) sketch data uncertainty learning, which aims to obtain a noise-robust sketch feature extraction model by introducing sketch uncertainty information into the training of a classification model; and (2) 3D shape feature transfer, where 3D shape features are mapped into the sketch embedding space under the guidance of sketch class centers. Finally, a cross-domain discriminative embedding space (i.e., sketches and 3D shapes belonging to the same class are close, while those of different classes are apart) is learned. The two tasks are discussed in detail in the following subsections.

## Code
A workable basic version of the code for CLIP adapted for ZS-SBIR has been uploaded.
- ```train_sketch.py``` python script to train the sketch model.
- ```train_view.py``` python script to train the view model.
- ```retrieval_evaluation.py``` python script to run the experiment.

## Qualitative Results

Qualitative results on SHREC2014 by a baseline (top) method vs Ours (bottom).
![retrieval1](https://github.com/cyy1998/CFTTSL/assets/37933688/216e281f-55bd-4fc7-bebf-87c081d40d2f)


## Quantitative Results
Quantitative results of our method against a few SOTAs.
![图片](https://github.com/cyy1998/CFTTSL/assets/37933688/85677aca-597e-43c6-896c-0f9ec5f6a586)
![图片](https://github.com/cyy1998/CFTTSL/assets/37933688/73dca006-8eb2-48fc-8e4c-4c92735ccbd3)

## Bibtex
Please cite our work if you found it useful. Thanks.
```
@article{LIANG2024103903,
title = {Sketch-based 3D shape retrieval via teacher–student learning},
journal = {Computer Vision and Image Understanding},
volume = {239},
pages = {103903},
year = {2024},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2023.103903},
url = {https://www.sciencedirect.com/science/article/pii/S1077314223002837},
author = {Shuang Liang and Weidong Dai and Yiyang Cai and Chi Xie},
keywords = {Sketch, 3D shape retrieval, Cross-modal difference, Teacher–student learning, Feature transfer},
}
```  


