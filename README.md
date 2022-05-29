# Accelerating Neural Architecture Search via Proxy Data

## Introduction
This is the offical code of [Accelerating Neural Architecture Search via Proxy Data](https://arxiv.org/abs/2106.04784) accepted in [IJCAI 2021](https://ijcai-21.org/).

Especially, we provide an implementation of DARTS with proxy data.

## Our experimental environment
```
Python >= 3.6.10, PyTorch == 1.4.0, torchvision == 0.5.0
```

## Datasets
While CIFAR-10, CIFAR-100, and SVHN can be automatically downloaded by torchvision, ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Search with proxy data
To execute DARTS with proxy data of CIFAR-10 (default sampling portion: 0.1), run
```
python train_search_proxy_data.py --gpu 0 --histogram --histogram_type 1

```

Please see `script.sh` to enjoy various examples for running the search code.

## Final evaluation (retraining)
Several neural networks searched in this study are included in `genotypes.py`.

To train a searched neural network (*e.g.*, named *your_arch*) on CIFAR-10, run
```
python train.py --gpu 0 --arch *your_arch*
```
To train a searched neural network on ImageNet, run
```
CUDA_VISIBLE_DEVICES="0,1" python train_imagenet.py --arch *your_arch* --datadir IMAGENET_PATH  --parallel --batch_size 384
```

## Reference
DARTS [ICLR19]  [code](https://github.com/quark0/darts)  [paper](https://openreview.net/pdf?id=S1eYHoC5FX)

PC-DARTS [ICLR20]  [code](https://github.com/yuhuixu1993/PC-DARTS)  [paper](https://openreview.net/pdf?id=BJlS634tPr)
