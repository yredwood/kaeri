import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import CustomLoader, Collater
from model import *
from loss import Kaeri_metric
import pickle
import numpy as np
from logger import Logger

import pdb



def main():
    #####################
    # hparams
    model_name = 'rnn_3'
    output_file = 'submissions/exp_{}.csv'.format(model_name)

    seed = 0
    max_epoch = 300
    batch_size = 32
    learning_rate = [1e-3, 1e-4]
    train_pkl = 'data/train.pkl'
    val_pkl = 'data/val.pkl'
    test_pkl = 'data/test.pkl'
    label_stats = 'data/label_stats.txt'
    label_classes = 'data/label_classes.pkl'
    logdir = 'viss/' + output_file.replace('.csv', '')

    with open(label_classes, 'rb') as f:
        label_cls = pickle.load(f) # [array([-400,-300,...]), array()..]

    #####################

    torch.manual_seed(seed)
    if model_name == 'rnn_1':
        model = RNN_1(label_stats)
    elif model_name == 'lin_1':
        model = Lin_1(label_stats)
    elif model_name == 'rnn_3':
        model = RNN_3(label_stats)
    elif model_name == 'conv_1':
        model = CONV_1(label_stats)
    elif model_name == 'tmp': 
        model = CONV_1(label_stats)

    model.cuda()
    lr = learning_rate[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = Kaeri_metric()
    logger = Logger(logdir)

    trainset = CustomLoader(train_pkl)
    valset = CustomLoader(val_pkl)
    testset = CustomLoader(test_pkl)
    collate_fn = Collater()

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True, batch_size=batch_size,
            collate_fn=collate_fn)
    
    val_loader = DataLoader(valset, num_workers=1, shuffle=False, batch_size=batch_size,
            collate_fn=collate_fn)

    test_loader = DataLoader(testset, num_workers=1, shuffle=False, batch_size=1,
            collate_fn=collate_fn)

    for epoch in range(max_epoch):

        model.train()
        train_loss = []
        for i, (data, label, _) in enumerate(train_loader):

            model.zero_grad()
            y_pred = model(data.cuda())

            loss = criterion(y_pred, label.cuda())
            loss.backward()

            optimizer.step()
            train_loss.append(loss)

        train_loss = torch.stack(train_loss)
            
        model.eval()
        eval_loss = []
        for i, (data, label, oid) in enumerate(val_loader):
        
            with torch.no_grad(): 
                y_pred = model(data.cuda()).cpu()

#                for fi in range(4):
#                    _lci = torch.Tensor(label_cls[fi])
#                    cls_idx = torch.argmin(
#                            torch.abs(y_pred[:,fi:fi+1] - _lci.unsqueeze(0)), -1
#                    )
#                    y_pred[:, fi] = _lci[cls_idx]

                loss = criterion(y_pred, label)
            eval_loss.append(loss)
        eval_loss = torch.stack(eval_loss)

        print ('Epoch {:5d} | triain loss: {:.4f} | val loss: {:.4f}'.format(epoch, 
            torch.mean(train_loss).data.cpu(), torch.mean(eval_loss).data.cpu()))

        logger.logging(epoch, torch.mean(train_loss).item(), torch.mean(eval_loss).item(), lr)

        # learning rate decay
        if epoch > max_epoch * 0.7 and learning_rate[0] == lr:
            lr = learning_rate[1]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    # inference
    model.eval()

    output_lines = ['id,X,Y,M,V']
    for i, (data, label, oid) in enumerate(test_loader):

        with torch.no_grad():
            y_pred = model(data.cuda())

        # round each predictions to label_cls
        line = '{}'.format(oid[0])
        y = y_pred[0].data.cpu().numpy()
        for i in range(4):
            line += ',{:.4f}'.format(y[i])
#            cls_idx = np.argmin(abs(y[i] - label_cls[i]))
#            line += ',{:.4f}'.format(label_cls[i][cls_idx])

        output_lines.append(line)

    with open(output_file, 'wt') as f:
        f.writelines('\n'.join(output_lines))

main()




            #
