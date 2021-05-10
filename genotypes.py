from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# ImageNet
imagenet_DARTS_proxy_histo1_10p = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
imagenet_PCDARTS_proxy_histo1_10p = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# CIFAR-10 
cifar10_DARTS_proxy_histo1_10p = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
cifar10_PCDARTS_proxy_histo1_10p = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
cifar10_EcoDARTS_c4r2_proxy_histo1_10p = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
cifar10_SDARTS_proxy_histo1_10p = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
cifar10_SGAS_proxy_histo1_10p = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
