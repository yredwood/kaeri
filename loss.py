import numpy as np
import torch


class Kaeri_metric(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, pred, label):

        e1 = torch.mean(
            torch.sum((pred[:,:2]-label[:,:2])**2, 1) / 2e+04
        )

        e2 = torch.mean(
            torch.sum(((pred[:,2:]-label[:,2:]) / (label[:,2:]+1e-6))**2, 1)
        )
        return 0.5*e1 + 0.5*e2

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
