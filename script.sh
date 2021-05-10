#!/bin/bash

gpu=$1
portion=0.1 ## sampling portion
seed=777

# data entropy (commands below are based on CIFAR-10)
#entropy_file='./entropy_list/cifar10_resnet20_index_entropy_class.txt' # CIFAR-10
#entropy_file='./entropy_list/cifar100_resnet56_index_entropy_class.txt' # CIFAR-100
#entropy_file='./entropy_list/svhn_resnet20_index_entropy_class.txt' # SVHN
#entropy_file='./entropy_list/imagenet_resnet50_index_entropy_class.txt' # ImageNet

# exp options: 
# search - [ "cifar10-random" "cifar10-p1" "cifar10-p2" "cifar10-p3" "imagenet-p1" ]
# evaluation - [ "cifar10-eval" "imagenet-eval" ]
exp="cifar10-p1"

if [ $exp == "cifar10-random" ]; then

# CIFAR-10, Random selection
# One 2080ti GPU
arch=proxy_random_10p_$seed
python train_search_proxy_data.py --gpu $gpu --save $arch --sampling_portion $portion --seed $seed --random

elif [ $exp == "cifar10-p1" ]; then

# CIFAR-10, proposed probabilistic selection (p1)
# One 2080ti GPU
arch=proxy_histo1_10p_$seed
entropy_file='./entropy_list/cifar10_resnet20_index_entropy_class.txt'
python train_search_proxy_data.py --gpu $gpu --save $arch --index_entropy_class $entropy_file --sampling_portion $portion --seed $seed --histogram --histogram_type 1

elif [ $exp == "cifar10-p2" ]; then

# CIFAR-10, proposed probabilistic selection (p2)
# One 2080ti GPU
arch=proxy_histo2_10p_$seed
entropy_file='./entropy_list/cifar10_resnet20_index_entropy_class.txt'
python train_search_proxy_data.py --gpu $gpu --save $arch --index_entropy_class $entropy_file --sampling_portion $portion --seed $seed --histogram --histogram_type 2

elif [ $exp == "cifar10-p3" ]; then

# CIFAR-10, proposed probabilistic selection (p3)
# One 2080ti GPU
arch=proxy_histo3_10p_$seed
entropy_file='./entropy_list/cifar10_resnet20_index_entropy_class.txt'
python train_search_proxy_data.py --gpu $gpu --save $arch --index_entropy_class $entropy_file --sampling_portion $portion --seed $seed --histogram --histogram_type 3

elif [ $exp == "imagenet-p1" ]; then

# ImageNet, proposed probabilistic selection (p1)
# One V100 GPU
arch=imagenet_proxy_histo1_10p_$seed
entropy_file='./entropy_list/imagenet_resnet50_index_entropy_class.txt' # ImageNet
python train_search_proxy_data_imagenet.py --gpu $gpu --save $arch --index_entropy_class $entropy_file --sampling_portion $portion --seed $seed --histogram --histogram_type 1

elif [ $exp == "cifar10-eval" ]; then

# CIFAR-10 training
# One 2080ti GPU
# inverse transfer (search on ImageNet using DARTS with proxy data, then evaluate on CIFAR-10)
arch=imagenet_DARTS_proxy_histo1_10p
python train.py --gpu $gpu --arch $arch --save ${arch} --cutout --auxiliary


elif [ $exp == "imagenet-eval" ]; then

# ImageNet training
# Two V100 GPUs
arch=imagenet_DARTS_proxy_histo1_10p
CUDA_VISIBLE_DEVICES="0,1" python train_imagenet.py --arch $arch --save ${arch} --auxiliary --parallel --batch_size 640

fi
