import pickle
from dataloader import Collater, CustomLoader
import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import sys

import pdb


def data_load(path, dtype):
    dataset = CustomLoader(path)
    return dataset

def metric(preds, labels):
    loss = 0.5*E1(labels, preds) + 0.5*E2(labels, preds)
    return loss

def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


if __name__=='__main__':
    model_name = sys.argv[1]
    seed = sys.argv[2]

    train_dataset = data_load('data/{}_train.pkl'.format(seed), 'train')
    test_dataset = data_load('data/{}_test.pkl'.format(seed), 'test')
    val_dataset = data_load('data/{}_val.pkl'.format(seed), 'val')
    

    tr_data, tr_label, tr_id = train_dataset.get_full_data()
    val_data, val_label, val_id = val_dataset.get_full_data()
    te_data, te_label, te_id = test_dataset.get_full_data()

    
#    dtrain = xgb.DMatrix(tr_data, tr_label)
#    dval = xgb.DMatrix(val_data, val_label)

#    num_round = 20
#    param = {'max_depth':2, 'eta': 1, 'objective': 'reg:squarederror'}
#    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    #bst = xgb.train(param, dtrain, num_round, watchlist)

    bst = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))
    bst.fit(tr_data, tr_label)

    pred = bst.predict(val_data)
    loss = metric(pred, val_label)
    print (loss)

    output_fname = '../submissions/{}.csv'.format(model_name)
    output_lines = ['id,X,Y,M,V']
    
    for i in range(len(pred)):
        line = '{}'.format(te_id[i])
        for j in range(4):
            line += ',{:.4f}'.format(pred[i][j])
        output_lines.append(line)

    with open(output_fname, 'wt') as f:
        f.writelines('\n'.join(output_lines))
