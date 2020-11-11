import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as datasets
import pandas as pd
import os
import shutil
import argparse
import numpy as np
from dataloader import get_dataloaders
import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, _ECELoss, RecallK, \
    DistillKL, mixup_data, get_data_folder, adjust_lr, cutmix_data, CrossEntropyLoss_label_smooth


import time
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data_folder', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ratio', default=1., type=float, help='learning rate')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[50, 100], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=100, type=int, help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='batch size')
parser.add_argument('--pre-trained', default='/home/user/hhd/winycg/pretrained_models/cutmix_model.pth.tar', type=str, help='dir')
parser.add_argument('--gpu-id', type=str, default='0,1,2')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='temperature for KD distillation')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='Dataset directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--test-mode', type=str, default='single_crop', help='evaluate model')

# global hyperparameter set
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# create files and folders
if not os.path.isdir('result'):
    os.mkdir('result')

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) \
          + '_arch_' + args.arch \
          + '_' + str(args.manual_seed) + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) \
          + '_arch_' + args.arch \
          + '_' + str(args.manual_seed)

print('dir for checkpoint:', log_dir)
with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')
        

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# ---------------------------------------------------------------------------------------

import socket
def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './dataset'
    hostname = socket.gethostname()
    if hostname.startswith('ubuntu'):
        data_folder = '/dev/shm/finegrained/' 
        args.checkpoint_dir = '/home/user/winycg/accv_checkpoint/'
    elif hostname.startswith('winycgv1'):
        data_folder = '/dev/shm/'
        args.checkpoint_dir = '/home/user/hhd/winycg/accv_checkpoint/'

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    return data_folder


loaders = get_dataloaders(get_data_folder(), args)
trainloader = loaders['train']
valloader = loaders['val']
testloader = loaders['test']

print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))
print('Number of test dataset: ' ,len(testloader.dataset))
num_classes = trainloader.dataset.num_classes
print('Number of classes: ' , num_classes)
C, H, W =  trainloader.dataset[0][0].size()

# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes)

print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
    % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, (1, C, H, W)) / 1e9))

del (net)


net = model(num_classes=num_classes, pretrained=args.pre_trained).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=1000, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a, num_classes) + (1 - lam) * criterion(pred, y_b, num_classes)


# Training
def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        
        input = input.cuda()
        target = target.cuda()
        input, targets_a, targets_b, lam = mixup_data(input, target, 0.4)

        logit = net(input)
        #loss_cls = criterion_cls(logit, target)
        loss_cls = mixup_criterion(CrossEntropyLoss_label_smooth, logit, targets_a, targets_b, lam)
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)

        top1, top5 = correct_num(logit, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)
        
        
        print('Epoch:{},batch_idx:{}/{}'.format(epoch, batch_idx, len(trainloader)),  'acc:', top1_num.item() / total, 'duration:', time.time()-batch_start_time)
    
    print('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\n top1_acc: {:.4f} \t top5_acc:{:.4f}'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        (top1_num/total).item(), (top5_num/total).item()))

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\ntop1_acc: {:.4f} \t top5_acc:{:.4f} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        (top1_num/total).item(), (top5_num/total).item()))


def val(epoch, criterion_list):
    test_loss = 0.
    test_loss_cls = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    criterion_cls = criterion_list[0]

    net.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(valloader):
            input = input.float()
            input = input.cuda()
            target = target.cuda()

            logit = net(input)
            loss_cls = criterion_cls(logit, target)
            loss = loss_cls

            test_loss += loss.item() / len(valloader)
            test_loss_cls += loss_cls.item() / len(valloader)

            top1, top5 = correct_num(logit, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

    
    with open(log_txt, 'a+') as f:
        f.write('test_loss:{:.5f}\t test_loss_cls:{:.5f}'
                '\t top1_acc:{:.4f} \t top5_acc:{:.4f} \n'
                .format(test_loss, test_loss_cls, (top1_num/total).item(), (top5_num/total).item()))
        print('test epoch:{}\t top1_acc:{:.4f} \t top5_acc:{:.4f}'.format(epoch, (top1_num/total).item(), (top5_num/total).item()))
    
    return (top1_num/total).item()


def test():
    test_imgs_name = []
    for x, y in testloader.dataset.samples:  # 将数据按类标存放
        test_imgs_name.append(x.split('/')[-1])
    print(len(test_imgs_name))
    predicted_class = []

    net.eval()
    with torch.no_grad():
        if args.test_mode == 'single':
            for batch_idx, (input, target) in enumerate(testloader):
                logits = net(input)
                predicted_class.append(logits.max(dim=1)[1])
            predicted_class = torch.cat(predicted_class, dim=0).cpu().numpy()
        else:
            for batch_idx, (input, target) in enumerate(testloader):
                #print(input.size())
                bs, nc, c, h, w = input.size()
                logits = net(input.view(-1, c, h, w))
                logits_avg = logits.view(bs,nc,-1).mean(1)
                predicted_class.append(logits_avg.max(dim=1)[1])

            predicted_class = torch.cat(predicted_class, dim=0).cpu().numpy()

    df=pd.DataFrame({'image_name':test_imgs_name,'class':predicted_class})
    df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(3)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls) 
    criterion_list.append(criterion_div)
    criterion_list.cuda()

    if args.evaluate:
        print('load trained weights from '+ os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        #best_acc = checkpoint['acc']
        #start_epoch = checkpoint['epoch']
        #print(val(start_epoch, criterion_list))
        test()
    else:#
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume from '+os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            #best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            #acc = val(epoch, criterion_list)

            state = {
                'net': net.module.state_dict(),
                #'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'))
            os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)

            '''
            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, args.arch + str(args.manual_seed) + '_best.pth.tar'))
            '''
        '''
        print('Evaluate the best model:')
        checkpoint = torch.load(args.checkpoint_dir + '/' +  args.arch + str(args.manual_seed) + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        top1_acc = test()
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)
        '''

