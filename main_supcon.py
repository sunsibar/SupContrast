from __future__ import print_function

import os
import sys
import argparse
import time
import math
from argparse import Namespace

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, weight_decay_schedule
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss
from utils.rmsnorm_etc import RMSNorm2d
from main_ce import set_loader as set_loader_ce
from main_linear import process_opt as process_opt_linear
from main_linear import train as train_linear
from main_linear import validate as validate_linear
from main_linear import adjust_learning_rate as adjust_learning_rate_linear
from main_linear import set_optimizer as set_optimizer_linear
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
from torch.nn import BCEWithLogitsLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--reload_from_epoch', type=int, default=0,
                        help='epoch to reload from')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to checkpoint to reload from')
    

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--increase_weight_decay', action='store_true',
                        help='increase weight decay by x10 after 50% of training')
    parser.add_argument('--train_on_neg_only', action='store_true',
                        help='train on negative only')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--norm', type=str, default='batchnorm',
                        choices=['batchnorm', 'layernorm', 'rmsnorm2d'], help='normalization type in resnet')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding dimension (of the output of the projection head)')
    parser.add_argument('--proj_head', type=str, default='linear',
                        choices=['linear', 'mlp'], help='projection head type')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # label smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing factor (0.0 to disable)')
    parser.add_argument('--clip_pos', type=float, default=0.0,
                        help='upper-bound clipping factor for positive similarities in the loss (0.0 to disable)')
    parser.add_argument('--clip_neg', type=float, default=0.0,
                        help='lower-bound clipping factor for negative similarities in the loss (0.0 to disable) - pass the desired distance to the max dissimilarity of 1')
    parser.add_argument('--clip_neg_top_k', type=int, default=-1,
                        help='only include the top k negatives in the denominator of the loss (per sample). Set to -1 to deactivate. Otherwise must be at least 2.')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # linear evaluation during training
    parser.add_argument('--linear_eval', action='store_true',
                        help='perform linear evaluation during training (whenever the model is stored)')
    parser.add_argument('--eval_freq', type=int, default=0,
                        help='frequency to perform linear evaluation (0 to disable)')
    parser.add_argument('--linear_eval_epochs', type=int, default=100,
                        help='number of epochs for linear evaluation')
    parser.add_argument('--linear_learning_rate', type=float, default=0.1,
                        help='learning rate for linear evaluation')
    parser.add_argument('--linear_learning_rate_decay', type=float, default=0.2,
                        help='learning rate decay for linear evaluation')
    parser.add_argument('--linear_momentum', type=float, default=0.9,
                        help='momentum for linear evaluation')
    parser.add_argument('--linear_weight_decay', type=float, default=0,
                        help='weight decay for linear evaluation')

    # binary evaluation during training
    parser.add_argument('--binary_eval', action='store_true',
                        help='perform binary evaluation during training')
    parser.add_argument('--binary_num_classes', type=int, default=10,
                        help='number of classes for binary evaluation')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = f'./save/{opt.method}/{opt.dataset}_models' 
    # opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    opt.tb_path = f'./save/{opt.method}/{opt.dataset}_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_ls_{}_clip-pos_{}_clip-neg_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.label_smoothing,
                opt.clip_pos, opt.clip_neg, 
                 opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.linear_eval:
        linear_opt = Namespace()
        linear_opt.batch_size = opt.batch_size
        linear_opt.num_workers = opt.num_workers
        linear_opt.epochs = opt.linear_eval_epochs
        linear_opt.learning_rate = opt.linear_learning_rate
        linear_opt.lr_decay_epochs = '60,75,90'
        linear_opt.lr_decay_rate = opt.linear_learning_rate_decay
        linear_opt.momentum = opt.linear_momentum
        linear_opt.weight_decay = opt.linear_weight_decay
        linear_opt.model = opt.model
        linear_opt.norm = opt.norm
        linear_opt.emb_dim = opt.emb_dim
        linear_opt.proj_head = opt.proj_head
        linear_opt.dataset = opt.dataset 
        linear_opt.cosine = False
        linear_opt.warm = False 
        linear_opt.print_freq = 100
        linear_opt = process_opt_linear(linear_opt)
    else:
        linear_opt = None


    return opt, linear_opt


def get_model_file(opt, epoch):
    return os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))


def set_loader(opt, is_train=True):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)


    transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]) 

    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root=opt.data_folder,
            train=is_train,
            transform=TwoCropTransform(transform),  # Always use TwoCropTransform
            download=True)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            root=opt.data_folder,
            train=is_train,
            transform=TwoCropTransform(transform),
            download=True)
    elif opt.dataset == 'path':
        dataset = datasets.ImageFolder(
            root=opt.data_folder,
            transform=TwoCropTransform(transform) )
        assert is_train, "Validation set not supported for path dataset"
    else:
        raise ValueError(opt.dataset)

    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=(sampler is None and is_train),
        num_workers=opt.num_workers, pin_memory=True, sampler=sampler)

    return loader


def set_model(opt):
    if opt.norm == 'layernorm':
        raise AssertionError("Layernorm not yet working for 2d data")
    elif opt.norm == 'rmsnorm2d':
        norm = RMSNorm2d
    else:
        norm = nn.BatchNorm2d
    model = SupConResNet(name=opt.model, norm=norm, feat_dim=opt.emb_dim, head=opt.proj_head)
    criterion = SupConLoss(temperature=opt.temp, neg_only=opt.train_on_neg_only,
                            label_smoothing=opt.label_smoothing,
                            clip_pos=opt.clip_pos, clip_neg=opt.clip_neg)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    if opt.reload_from_epoch > 0:
        if opt.ckpt is not None:
            checkpoint = torch.load(opt.ckpt, map_location='cpu')
            assert int(checkpoint["epoch"]) == opt.reload_from_epoch
        else:
            checkpoint = torch.load(get_model_file(opt, opt.reload_from_epoch), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        start_epoch = int(checkpoint['epoch'])
        print(f"Loading model weights from checkpoint: {get_model_file(opt, opt.reload_from_epoch)}")
    else:
        start_epoch = 0

    return model, criterion, start_epoch


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_no_label_smoothing = AverageMeter()

    weight_decay_schedule(opt, epoch, optimizer)

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
            with torch.no_grad():
                loss_no_label_smoothing = criterion(features, labels, use_label_smoothing=False)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
            with torch.no_grad():
                loss_no_label_smoothing = criterion(features, use_label_smoothing=False)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)
        with torch.no_grad():
            losses_no_label_smoothing.update(loss_no_label_smoothing.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, losses_no_label_smoothing.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_no_label_smoothing = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method in ['SupCon', 'TargetVector']:
                loss = criterion(features, labels)
                loss_no_label_smoothing = criterion(features, labels, use_label_smoothing=False)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
                loss_no_label_smoothing = criterion(features, use_label_smoothing=False)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)
            losses_no_label_smoothing.update(loss_no_label_smoothing.item(), bsz)
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses))

    return losses.avg, losses_no_label_smoothing.avg


def set_linear_classifier(opt, linear_opt):
    """Set up linear classifier for evaluation"""
    classifier = LinearClassifier(name=opt.model, num_classes=10 if opt.dataset == 'cifar10' else 100)
    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    # Use the same optimizer settings as in main_linear.py
    optimizer = set_optimizer_linear(linear_opt, classifier)
    # optimizer = torch.optim.SGD(classifier.parameters(),
    #                             lr=opt.linear_learning_rate,
    #                             momentum=opt.linear_momentum,
    #                             weight_decay=opt.linear_weight_decay)
    
    return classifier, criterion, optimizer


def linear_eval(model, opt, linear_opt, logger, epoch, log_individual_n_classes=10):
    """Perform linear evaluation"""
    print("=== Performing linear evaluation ===")
    
    # Create data loaders for linear evaluation
    # We need to modify set_loader to handle non-TwoCropTransform case
    # train_loader = set_loader(opt, is_train=True, two_crop=False)
    # val_loader = set_loader(opt, is_train=False, two_crop=False)
    train_loader, val_loader = set_loader_ce(opt)

    
    # Set up linear classifier
    classifier, criterion, optimizer = set_linear_classifier(opt, linear_opt)
    
    best_acc = 0
    best_acc_per_class = [0] * log_individual_n_classes
    for e in range(1, opt.linear_eval_epochs + 1):
        # Adjust learning rate according to schedule

        adjust_learning_rate(linear_opt, optimizer, e)
            
        # Train for one epoch
        train_loss, train_acc, train_acc_per_class = \
            train_linear(train_loader, model, classifier, criterion, optimizer, e, linear_opt,
                         log_individual_n_classes=log_individual_n_classes)
        
        # Evaluate on validation set
        val_loss, val_acc, val_acc_per_class = \
            validate_linear(val_loader, model, classifier, criterion, linear_opt,
            log_individual_n_classes=log_individual_n_classes)
        
        if val_acc > best_acc:
            best_acc = val_acc
        for i in range(log_individual_n_classes):
            if val_acc_per_class[i] > best_acc_per_class[i]:
                best_acc_per_class[i] = val_acc_per_class[i]
            
        # Print progress every 20 epochs
        if e % 100 == 0 or e == opt.linear_eval_epochs or e == 1:
            print(f'Linear eval epoch {e}, train_acc: {train_acc:.2f}, val_acc: {val_acc:.2f}, best_acc: {best_acc:.2f}')
    
    # Log the best accuracy to tensorboard
    logger.log_value('linear_eval_acc', best_acc, epoch)
    for i in range(log_individual_n_classes):
        logger.log_value(f'linear_eval_acc_class_{i}', best_acc_per_class[i], epoch)
        
    print(f"=== Linear evaluation complete. Best accuracy: {best_acc:.2f} ===")
    
    return best_acc

class MultiBinaryClassifier(nn.Module):
    def __init__(self, name='resnet18', num_classes=10):
        super(MultiBinaryClassifier, self).__init__()
        dim_in = 2048
        if name.startswith('resnet'):
            if name.endswith('18'):
                dim_in = 512
            elif name.endswith('34'):
                dim_in = 512
            elif name.endswith('50'):
                dim_in = 2048
            elif name.endswith('101'):
                dim_in = 2048
        
        # Create a single linear layer with multiple outputs (one per class)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(x)  # Shape: [batch_size, num_classes]


def binary_eval(model, opt, linear_opt, logger, epoch, num_classes=10):
    """Perform binary single-class evaluation with joint training"""
    print(f"=== Performing joint binary evaluation for first {num_classes} classes ===")
    
    # Create data loaders for evaluation
    train_loader, val_loader = set_loader_ce(opt)
    
    # Set up multi-binary classifier
    classifier = MultiBinaryClassifier(name=opt.model, num_classes=num_classes)
    criterion = BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=opt.linear_learning_rate,
                                momentum=opt.linear_momentum,
                                weight_decay=opt.linear_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    # Track best accuracy for each class
    best_accs = [0] * num_classes
    
    for e in range(1, opt.linear_eval_epochs + 1):
        adjust_learning_rate(linear_opt, optimizer, e)
        
        # Train for one epoch
        train_loss, train_accs = train_binary_joint(train_loader, model, classifier, criterion, optimizer, e, opt, num_classes)
        scheduler.step(train_loss) # wait, isn't this done by adjust_learning_rate?

        # Evaluate
        val_loss, val_accs = validate_binary_joint(val_loader, model, classifier, criterion, opt, num_classes)
        
        # Update best accuracies
        for i in range(num_classes):
            if val_accs[i] > best_accs[i]:
                best_accs[i] = val_accs[i]
        
        # Print progress occasionally
        if e % 100 == 0 or e == opt.linear_eval_epochs or e == 1:
            avg_train_acc = sum(train_accs) / len(train_accs)
            avg_val_acc = sum(val_accs) / len(val_accs)
            avg_best_acc = sum(best_accs) / len(best_accs)
            print(f'Binary eval epoch {e}, avg_train_acc: {avg_train_acc:.2f}, avg_val_acc: {avg_val_acc:.2f}, avg_best_acc: {avg_best_acc:.2f}')
    
    # Log to tensorboard
    for i in range(num_classes):
        logger.log_value(f'binary_eval_acc_class_{i}', best_accs[i], epoch)
    
    # Log average accuracy
    avg_acc = sum(best_accs) / len(best_accs)
    logger.log_value('binary_eval_avg_acc', avg_acc, epoch)
    print(f"=== Binary evaluation complete. Average accuracy: {avg_acc:.2f} ===")
    
    return best_accs


def train_binary_joint(train_loader, model, classifier, criterion, optimizer, epoch, opt, num_classes):
    """Joint training for all binary classifiers"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = [AverageMeter() for _ in range(num_classes)]

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # Compute features once
        with torch.no_grad():
            features = model.encoder(images)
        
        # Forward pass through all binary classifiers at once
        outputs = classifier(features.detach())  # [batch_size, num_classes]
        
        # Create binary labels for each class
        # binary_labels = torch.zeros(bsz, num_classes, device=labels.device)
        # for c in range(num_classes):
        # binary_labels[:, :] = (labels[:, None] == torch.arange(num_classes, device=labels.device)).float()
        binary_labels = (labels.unsqueeze(1) == torch.arange(num_classes, device=labels.device).unsqueeze(0)).float()
            # binary_labels[:, c] = (labels == c).float()
        factor = (8 if opt.dataset == 'cifar10' else 98) # weight positive samples 9 / 99 times more
        weights = torch.ones_like(binary_labels) + factor * binary_labels 
        # Compute loss for all classifiers
        loss = F.binary_cross_entropy_with_logits(outputs, binary_labels,
                                                  weights)
        # loss = criterion(outputs, binary_labels)  # [batch_size, num_classes]
        # loss = losses_per_sample.mean()  # Average over all samples and classes
        
        # Calculate accuracy for each classifier
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predictions == binary_labels).float().sum(dim=0)
        accuracies = correct / bsz * 100
        
        # Update accuracy meters
        for c in range(num_classes):
            accs[c].update(accuracies[c].item(), bsz)

        # Update loss metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Return average loss and accuracies for each class
    return losses.avg, [acc.avg for acc in accs]


def validate_binary_joint(val_loader, model, classifier, criterion, opt, num_classes):
    """Joint validation for all binary classifiers"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = [AverageMeter() for _ in range(num_classes)]

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # Forward pass
            features = model.encoder(images)
            outputs = classifier(features)  # [batch_size, num_classes]
            
            # Create binary labels for each class - more efficient one-liner
            binary_labels = (labels.unsqueeze(1) == torch.arange(num_classes, device=labels.device).unsqueeze(0)).float()
            
            # Compute loss
            loss = criterion(outputs, binary_labels)
            
            # Calculate accuracy for each classifier
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predictions == binary_labels).float().sum(dim=0)
            accuracies = correct / bsz * 100
            
            # Update accuracy meters
            for c in range(num_classes):
                accs[c].update(accuracies[c].item(), bsz)

            # Update loss metric
            losses.update(loss.item(), bsz)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Return average loss and accuracies for each class
    return losses.avg, [acc.avg for acc in accs]


def main():
    opt, linear_opt = parse_option()

    # build data loader
    train_loader = set_loader(opt, is_train=True)
    val_loader = set_loader(opt, is_train=False)

    # build model and criterion
    model, criterion, start_epoch = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print(f'TensorBoard logger logging to: {opt.tb_folder}')

    if start_epoch == 0:
        save_file = get_model_file(opt, 0) 
        save_model(model, optimizer, opt, 0, save_file)

    if opt.linear_eval and start_epoch == 0:
        linear_acc = linear_eval(model, opt, linear_opt, logger, start_epoch, 
                                         log_individual_n_classes=opt.binary_num_classes)
        print(f'Linear evaluation accuracy before training: {linear_acc:.2f}')

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, loss_no_ls = train(train_loader, model, criterion, optimizer, epoch + start_epoch, opt)
        
        # eval for one epoch
        val_loss, val_loss_no_ls = validate(val_loader, model, criterion, opt)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}, train_loss: {:.3f}, val_loss: {:.3f}, val_loss_no_ls: {:.3f}'.format(
            epoch + start_epoch, time2 - time1, loss, val_loss, val_loss_no_ls))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch + start_epoch)
        logger.log_value('train_loss_no_label_smoothing', loss_no_ls, epoch + start_epoch)
        logger.log_value('val_loss', val_loss, epoch + start_epoch)
        logger.log_value('val_loss_no_label_smoothing', val_loss_no_ls, epoch + start_epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch + start_epoch)
        logger.log_value('weight_decay', optimizer.param_groups[0]['weight_decay'], epoch + start_epoch)
        logger.log_value('train_on_neg_only', int(opt.train_on_neg_only), epoch + start_epoch)

        # perform linear evaluation if specified
        if (epoch % opt.save_freq == 0  or epoch == opt.epochs):

            # perform binary evaluation if specified - commented out since extremely slow
            # if opt.binary_eval:
            #     binary_accs = binary_eval(model, opt, linear_opt, logger, epoch + start_epoch, num_classes=opt.binary_num_classes)
            #     avg_acc = sum(binary_accs) / len(binary_accs)
            #     print(f'Binary evaluation average accuracy at epoch {epoch + start_epoch}: {avg_acc:.2f}')
            if opt.linear_eval:
                # perform linear evaluation
                linear_acc = linear_eval(model, opt, linear_opt, logger, epoch + start_epoch, 
                                         log_individual_n_classes=opt.binary_num_classes)
                print(f'Linear evaluation accuracy at epoch {epoch + start_epoch}: {linear_acc:.2f}')
        
        if epoch % opt.save_freq == 0:
            save_file = get_model_file(opt, epoch + start_epoch)
            # save_file = os.path.join(
            #     opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch + start_epoch))
            save_model(model, optimizer, opt, epoch + start_epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs + start_epoch, save_file)


if __name__ == '__main__':
    main()
