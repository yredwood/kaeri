import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Collater, CustomLoader
from model import Model, RNN
from logger import Logger
import sys

import pdb


def data_load(path, dtype):
    if dtype == 'train':
        shuffle = True
        batch_size = 24
    else:
        shuffle = False
        batch_size = 1
    dataset = CustomLoader(path)
    s_all, d_all, _, _, = dataset.get_full_data()
    statistics = (np.mean(s_all, 0), np.std(s_all, 0), np.mean(d_all, (0,1)), np.std(d_all, (0,1)))
    dataloader = DataLoader(dataset, num_workers=1, shuffle=shuffle,
            batch_size=batch_size, collate_fn=Collater())
    return statistics, dataloader


if __name__ == '__main__':

    # ========================
    max_epoch = 301
    batch_size = 32
    model_name = sys.argv[1]
    dataset_prefix = sys.argv[2]

    _, dataloader = data_load('data/seed_{}/train.pkl'.format(dataset_prefix), 'train')
    _, valloader = data_load('data/seed_{}/val.pkl'.format(dataset_prefix), 'train')
    stats, testloader = data_load('data/seed_{}/test.pkl'.format(dataset_prefix), 'test')
    learning_rate = [1e-2, 1e-3]
    
    if 'rnn' in model_name:
        model = RNN(stats).cuda()
    else:
        model = Model(stats).cuda()

    logdir = 'logs/{}'.format(model_name)
    print (model_name)
    # ========================
    
    lr = learning_rate[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger = Logger(logdir)

    for epoch in range(max_epoch + 100):
        
        model.train()
        losses = []
        for i, (static, dynamic, label, _) in enumerate(dataloader):
            model.zero_grad()

            pred = model.forward(static.cuda(), dynamic.cuda())
            loss = model.get_loss(pred, label.cuda())
#            pred = model.invert_forward(label.cuda(), dynamic.cuda())
#            loss = model.invert_get_loss(pred, static.cuda())
#            inverse_pred = model.invert_forward(pred, dynamic.cuda())
#            inverse_loss = model.invert_get_loss(inverse_pred, static.cuda())
#            loss_sum = loss + inverse_loss
#            loss_sum.backward()

            loss.backward()
            optimizer.step()

            losses.append(loss.data.cpu().numpy())

        model.eval()
        eval_loss = [] 
        for i, (static, dynamic, label, _) in enumerate(valloader):
            with torch.no_grad():
                pred = model(static.cuda(), dynamic.cuda())
                loss = model.get_loss(pred, label.cuda())
#                pred = model.invert_forward(label.cuda(), dynamic.cuda())
#                loss = model.invert_get_loss(pred, static.cuda())
                eval_loss.append(loss.data.cpu().numpy()) 
        
        print ('epoch {}| tr loss {:.4f} | ev loss {:.4f}'.format(epoch, np.mean(losses), np.mean(eval_loss)))
        logger.logging(epoch, np.mean(losses), np.mean(eval_loss), lr)

        if epoch > max_epoch * 0.7 and lr == learning_rate[0]:
            lr = learning_rate[1]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 100==0:
            output_fname = 'submissions/{}_epoch{}.csv'.format(model_name, epoch)
            model.eval()
            output_lines = ['id,X,Y,M,V']
            for i, (static, dynamic, label, oid) in enumerate(testloader):
                with torch.no_grad():
                    y_pred = model(static.cuda(), dynamic.cuda())

                line = '{}'.format(oid[0])
                y = y_pred[0].data.cpu().numpy()
                for i in range(4):
                    line += ',{:.4f}'.format(y[i])

                output_lines.append(line)
            output_lines.append(str(np.mean(eval_loss)))
            with open(output_fname, 'wt') as f:
                f.writelines('\n'.join(output_lines))









        #
