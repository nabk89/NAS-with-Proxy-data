import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

from sampler import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--datadir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='target data [cifar10, cifar100, svhn]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

################################
## NAS with proxy data options
parser.add_argument('--entropy_file', type=str, 
                    default='./entropy_list/cifar10_resnet20_entropy_file.txt', 
                    help='index/entropy/class of the target dataset obtained from a pretrained network')
parser.add_argument('--sampling_portion', type=float, default=0.2, help='proxy dataset relative size to the target dataset')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--random', action='store_true', default=False, help='use random sampling')
parser.add_argument('--histogram', action='store_true', default=False, help='use histogram-based sampling')
parser.add_argument('--histogram_type', type=int, default=1, help='choice: [1, 2, 3]') # default is P1
################################
parser.add_argument('--resolution', type=int, default=32, help='image resolution (input dim)')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  ## data load
  if args.dataset == 'cifar10':
    num_class = 10
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.datadir, train=True, download=True, transform=train_transform)
  elif args.dataset == 'cifar100':
    num_class = 100
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.datadir, train=True, download=True, transform=train_transform)
  elif args.dataset == 'svhn':
    num_class = 10
    train_transform, valid_transform = utils._data_transforms_svhn(args)
    train_data = dset.SVHN(root=args.datadir, split='train', download=True, transform=train_transform)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, num_class, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  ##############################
  ## sampling proxy data
  index, entropy, label = read_entropy_file(args.entropy_file)
  if args.random:
    indices = get_proxy_data_random(entropy, args.sampling_portion, logging)
  elif args.histogram:
    indices  = get_proxy_data_log_entropy_histogram(entropy, args.sampling_portion, args.histogram_type, args.dataset, logging)
  else:
    indices = [*range(len(train_data))]

  num_train = num_proxy_data = len(indices)
  split = int(np.floor(args.train_portion * num_proxy_data))
  logging.info('D_train: %d, D_val: %d'%(split, num_proxy_data - split))
  num_classes = [0] * num_class
  with open(os.path.join(args.save, 'proxy_train_entropy_file.txt'), 'w') as f:
    for idx in indices[:split]:
      f.write('%d %f %d\n'%(idx, entropy[idx], label[idx]))
      num_classes[label[idx]] += 1
  with open(os.path.join(args.save, 'proxy_val_entropy_file.txt'), 'w') as f:
    for idx in indices[split:num_train]:
      f.write('%d %f %d\n'%(idx, entropy[idx], label[idx]))
      num_classes[label[idx]] += 1
  logging.info(num_classes)
  ###############################

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    epoch_start = time.time()
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # get a genotype after update
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # write gap between train_acc and valid_acc
    logging.info('gap (train_acc - valid_acc): %f', train_acc - valid_acc)
    
    logging.info('epoch time %d sec. (expected finished time: after %.1f min.)', time.time() - epoch_start, (args.epochs-1-epoch)*(time.time()-epoch_start)/60)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    torch.save(model.arch_parameters(), os.path.join(args.save, 'alphas.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.train()
   
  alpha_update_freq = int(np.floor(args.train_portion / (1-args.train_portion)))

  num_steps = len(train_queue)
  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda(non_blocking=True)
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    if step % alpha_update_freq == 0:
      # get a random minibatch from the search queue with replacement
      try:
        input_search, target_search = next(valid_queue_iter)
      except:
        valid_queue_iter = iter(valid_queue)
        input_search, target_search = next(valid_queue_iter)
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)

      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0 or step == num_steps-1:
      logging.info('train (%03d/%d) loss: %e top1: %f top5: %f', step, num_steps, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  num_steps = len(valid_queue)
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, requires_grad=False).cuda(non_blocking=True)
      target = Variable(target, requires_grad=False).cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)

      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0 or step == num_steps-1:
        logging.info('valid (%03d/%d) loss: %e top1: %f top5: %f', step, num_steps, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

