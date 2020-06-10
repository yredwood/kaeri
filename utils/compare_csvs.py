import numpy as np

def loss_func(pred, target):
    e1 = np.mean(
            np.sum((pred[:,:2] - target[:,:2])**2, 1) / 2e+4
    )
    e2 = np.mean(
            np.sum(((pred[:,2:] - target[:,2:]) / (target[:,2:]+1e-6))**2, 1)
    )
    return 0.5*e1 + 0.5*e2

def load_data(fname):
    with open(fname, 'rt') as f:
        lines = f.readlines()
    preds = []
    for i in range(1,len(lines)):
        preds.append([float(x) for x in lines[i].strip().split(',')[1:]])

    return np.array(preds)

fname1 = '../finals/ensemble_0610_1.csv'
#fname1 = '../finals/nograd_0.028.csv'
fname2 = '../finals/grad_0.019.csv'
#fname2 = '../finals/folded_constgrad_0.024.csv'

pr1 = load_data(fname1)
pr2 = load_data(fname2)

print (loss_func(pr1, pr2))
