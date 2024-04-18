from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets.modelnet40_ply import ModelNet40
from models.menghao_trafo_new import Pct
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import torch.nn.functional as F

from rail1.data.batchloader import BatchLoader
import rail1


import time 

def cal_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def forward_and_loss_fn(input, model):
    points, labels = input
    preds = model.forward(points)

    labels = labels.squeeze(-1)

    loss = cal_loss(preds, labels, smoothing=False)
    return loss.mean(0), {
        "loss": loss,
        "logits": preds,
        "targets": labels,
    }


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train(args, io):
    train_loader = BatchLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True)
    test_loader = BatchLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct().to(device)
    print(str(model))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_test_acc = 0

    datasets = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'val_loader': None,
    }
    
    rail1.fit(
        run_dir='./checkpoints/',
        model=model,
        optimizer=opt,
        datasets=datasets,
        forward_and_loss_fn=forward_and_loss_fn,
        max_steps=100_000,
        metrics_fns=[rail1.metrics.accuracy]
    )


    # for epoch in range(args.epochs):
    #     scheduler.step()
    #     train_loss = 0.0
    #     count = 0.0
    #     model.train()
    #     train_pred = []
    #     train_true = []
    #     idx = 0
    #     total_time = 0.0
    #     batches_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    #     for i in range(batches_per_epoch):

    #         data, label = train_loader[i + epoch * batches_per_epoch]
    #         data, label = data.to(device), label.to(device).squeeze() 
    #         data = data.permute(0, 2, 1)
    #         batch_size = data.size()[0]
    #         opt.zero_grad()

    #         start_time = time.time()
    #         logits = model(data)
    #         loss = criterion(logits, label)
    #         loss.backward()
    #         opt.step()
    #         end_time = time.time()
    #         total_time += (end_time - start_time)
            
    #         preds = logits.max(dim=1)[1]
    #         count += batch_size
    #         train_loss += loss.item() * batch_size
    #         train_true.append(label.cpu().numpy())
    #         train_pred.append(preds.detach().cpu().numpy())
    #         idx += 1
            
    #     print ('train total time is',total_time)
    #     train_true = np.concatenate(train_true)
    #     train_pred = np.concatenate(train_pred)
    #     outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
    #                                                                             train_loss*1.0/count,
    #                                                                             metrics.accuracy_score(
    #                                                                             train_true, train_pred),
    #                                                                             metrics.balanced_accuracy_score(
    #                                                                             train_true, train_pred))
    #     io.cprint(outstr)

        ####################
        # Test
        ####################
        # test_loss = 0.0
        # count = 0.0
        # model.eval()
        # test_pred = []
        # test_true = []
        # total_time = 0.0
        # for data, label in test_loader:
        #     data, label = data.to(device), label.to(device).squeeze()
        #     data = data.permute(0, 2, 1)
        #     batch_size = data.size()[0]
        #     start_time = time.time()
        #     logits = model(data)
        #     end_time = time.time()
        #     total_time += (end_time - start_time)
        #     loss = criterion(logits, label)
        #     preds = logits.max(dim=1)[1]
        #     count += batch_size
        #     test_loss += loss.item() * batch_size
        #     test_true.append(label.cpu().numpy())
        #     test_pred.append(preds.detach().cpu().numpy())
        # print ('test total time is', total_time)
        # test_true = np.concatenate(test_true)
        # test_pred = np.concatenate(test_pred)
        # test_acc = metrics.accuracy_score(test_true, test_pred)
        # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
        #                                                                     test_loss*1.0/count,
        #                                                                     test_acc,
        #                                                                     avg_per_class_acc)
        # io.cprint(outstr)
        # if test_acc >= best_test_acc:
        #     best_test_acc = test_acc
        #     # torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    
    os.makedirs('checkpoints/%s' % args.exp_name, exist_ok=True)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)