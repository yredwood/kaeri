import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Collater, CustomLoader
from model import Model
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
    dataloader = DataLoader(dataset, num_workers=1, shuffle=shuffle,
            batch_size=batch_size, collate_fn=Collater())
    return dataloader


if __name__ == '__main__':

    # ========================
    max_epoch = 301
    batch_size = 32
    model_name = sys.argv[1]
    dataset_prefix = sys.argv[2]

    dataloader = data_load('data/{}_train.pkl'.format(dataset_prefix), 'train')
    valloader = data_load('data/{}_val.pkl'.format(dataset_prefix), 'train')
    testloader = data_load('data/{}_test.pkl'.format(dataset_prefix), 'test')
    learning_rate = [1e-2, 1e-3]
    model = Model().cuda()
    logdir = 'logs/{}'.format(model_name)
    print (model_name)
    # ========================
    
    lr = learning_rate[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger = Logger(logdir)

    for epoch in range(max_epoch + 100):
        
        model.train()
        losses = []
        for i, (data, label, _) in enumerate(dataloader):
            model.zero_grad()

            pred = model.forward(data.cuda())
            loss = model.get_loss(pred, label.cuda())
            
            loss.backward()
            optimizer.step()

            losses.append(loss.data.cpu().numpy())

        model.eval()
        eval_loss = [] 
        for i, (data, label, _) in enumerate(valloader):
            with torch.no_grad():
                pred = model(data.cuda())
                loss = model.get_loss(pred, label.cuda())
                eval_loss.append(loss.data.cpu().numpy()) 
        
        print ('epoch {}| tr loss {:.4f} | ev loss {:.4f}'.format(epoch, np.mean(losses), np.mean(eval_loss)))
        logger.logging(epoch, np.mean(losses), np.mean(eval_loss), lr)

        if epoch > max_epoch * 0.7 and lr == learning_rate[0]:
            lr = learning_rate[1]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 100==0:
            output_fname = '../submissions/{}_epoch{}.csv'.format(model_name, epoch)
            model.eval()
            output_lines = ['id,X,Y,M,V']
            for i, (data, label, oid) in enumerate(testloader):
                with torch.no_grad():
                    y_pred = model(data.cuda())

                line = '{}'.format(oid[0])
                y = y_pred[0].data.cpu().numpy()
                for i in range(4):
                    line += ',{:.4f}'.format(y[i])

                output_lines.append(line)
            with open(output_fname, 'wt') as f:
                f.writelines('\n'.join(output_lines))










        #