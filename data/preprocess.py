import numpy as np
import pdb
import pickle


fname = 'train_features.csv'

# ============= load data  ===========================

def _get_data(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    output = {}
    data_all = []
    for i, line in enumerate(lines):
        if i == 0:
            headers = line.split(',')
        else:
            oid, time, s1, s2, s3, s4 = line.split(',')
            oid = int(oid)
            
            data_t = np.stack(
                    [float(time), float(s1), float(s2), float(s3), float(s4)]
            )
            data_all.append(data_t)
            if oid in output.keys():
                output[oid]['data'].append(data_t)
            else:
                output[oid] = {}
                output[oid]['data'] = [data_t]
    return output, data_all

train_data, train_data_all = _get_data('train_features.csv')
test_data, test_data_all = _get_data('test_features.csv')







# ============= get statistics and normalize it ===========================
mean, std = np.mean(train_data_all, 0), np.std(train_data_all, 0)
lines = ['t,s1,s2,s3,s4', 
        '{:.4f},{:.4f},{:.4f},{:.4f}'.format(mean[0], mean[1], mean[2], mean[3]), 
        '{:.4f},{:.4f},{:.4f},{:.4f}'.format(std[0], std[1], std[2], std[3])]

with open('data_stats.txt', 'w') as f:
    f.writelines('\n'.join(lines))

# also save static features
train_stats = []; test_stats = []
for key in train_data.keys():
    train_data[key]['data'] = (train_data[key]['data'] - mean) / std
    _td = train_data[key]['data']
    train_data[key]['static'] = np.concatenate(\
            [_td.mean(0), _td.min(0), _td.max(0), _td.std(0)]
    )
    train_stats.append(train_data[key]['static'])

for key in test_data.keys():
    test_data[key]['data'] = (test_data[key]['data'] - mean) / std
    _td = test_data[key]['data']
    test_data[key]['static'] = np.concatenate(\
            [_td.mean(0), _td.min(0), _td.max(0), _td.std(0)]
    )
    test_stats.append(test_data[key]['static'])

# confirm by all_data
#k = (train_data_all - np.expand_dims(mean, 0))  / np.expand_dims(std, 0)
#b = (test_data_all - np.expand_dims(mean, 0)) / np.expand_dims(std, 0)







# ========================= get label of data ===========================
# add labels in train_data
fname = 'train_target.csv'
with open(fname, 'r') as f:
    lines = f.readlines()

data_all = []
for i, line in enumerate(lines):
    if i == 0:
        headers = line.split(',')
    else:
        oid, x, y, m, v = line.split(',')
        oid = int(oid)
        if oid in train_data.keys():
            train_data[oid]['label'] = np.stack(
                    [float(x), float(y), float(m), float(v)]
            )
            data_all.append(train_data[oid]['label'])
        else:
            print ('warning: not exists oid')

mean, std = np.mean(data_all, 0), np.std(data_all, 0)

lines = ['t,s1,s2,s3,s4', 
        '{:.4f},{:.4f},{:.4f},{:.4f}'.format(mean[0], mean[1], mean[2], mean[3]), 
        '{:.4f},{:.4f},{:.4f},{:.4f}'.format(std[0], std[1], std[2], std[3])]


with open('label_stats.txt', 'w') as f:
    f.writelines('\n'.join(lines))

# no need to normalize here: it normalizes in criterion
#for key in train_data.keys():
#    train_data[key]['label'] = (train_data[key]['label'] - mean) / std


# unique targets (class)
output_list = []
for i in range(4):
    f = np.unique([_d[i] for _d in data_all])
    output_list.append(f)

with open('label_classes.pkl', 'wb') as f:
    pickle.dump(output_list, f)


# ========================= split train data into train/val ======================
# random split trian/val (8:2)
#np.random.seed(0)
#rnd_idx = np.random.choice(len(train_data)-2, size=int(len(train_data)*0.8), replace=False)
#
#tr_data = {}
#val_data = {}
#for i in range(len(train_data)):
#    if i in rnd_idx:
#        tr_data[i] = train_data[i]
#    else:
#        val_data[i] = train_data[i]


# select validation set based on distance from test set
# test_stats: (700,20), train_stats: (2900,20)
_keys = []
_dists = []
for key in train_data.keys():
    
    static_feature = train_data[key]['static']

    dist_from_train = ((static_feature - np.mean(train_stats)) ** 2 / np.std(train_stats)).sum()
    dist_from_test = ((static_feature - np.mean(test_stats)) ** 2 / np.std(test_stats)).sum()

    _keys.append(key)
    _dists.append(dist_from_train / dist_from_test) # if it gets bigger, it approaches to test set

_idx = np.argsort(_dists)
val_key_list = [_keys[_i] for _i in _idx[int(len(train_data)*0.8):]]

tr_data = {}
val_data = {}

for key in train_data.keys():
    if key in val_key_list:
        val_data[key] = train_data[key]
#        tr_data[key] = train_data[key]
#        tr_data[key]['importance']
    else:
        tr_data[key] = train_data[key]

    
output_fname = 'train.pkl'
with open(output_fname, 'wb') as f:
    pickle.dump(tr_data, f)


output_fname = 'val.pkl'
with open(output_fname, 'wb') as f:
    pickle.dump(val_data, f)


output_fname = 'test.pkl'
with open(output_fname, 'wb') as f:
    pickle.dump(test_data, f)

print ('done')
