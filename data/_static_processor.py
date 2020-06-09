import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

import pdb

def __name__ == '__main__':
    root_dir = '/nfs/maximoff/ext01/mike/tmp/nocad/KAERI_dataset/'

    fname = 'train_features.csv'

    # 1. load data

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


    # 2. load dataset
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

    # 2. visualize signals
    #for key in train_data.keys():
    #    data = np.array(train_data[key]['data']) # 375,5
    #
    #    f, axes = plt.subplots(4, 1)
    #    f.set_size_inches((10, 5)) 
    #    f.tight_layout() 
    #    
    #    # arrival time
    #    #at = (np.argmin(data, 0) + np.argmax(data, 0)) // 2
    #
    #    peak = data[:,1:].max(0)
    #    npeak = data[:,1:].min(0)
    #
    #    at = np.zeros(4)
    #    nat = np.zeros(4)
    #    for t in range(len(data)):
    #        for i in range(4):
    #            if data[t,i+1] > peak[i] * 0.55 and at[i] == 0:
    #                at[i] = t
    #
    #            if data[t,i+1] < npeak[i] * 0.55 and nat[i] == 0:
    #                nat[i] = t
    #
    #        if (at != 0).all() and (nat != 0).all():
    #            break
    #
    #    t = ((at + nat)//2).astype(int)
    #
    #
    #    for i in range(4):
    #        axes[i].plot(data[:,0], data[:,i+1])
    #        axes[i].plot(data[t[i],0], 0, 'r+')
    #        axes[i].set_title('S{}'.format(i+1))
    #        axes[i].set_xlabel('time')
    #    
    #    plt.savefig('pngs/{}.png'.format(key))
    #    break


    # 3. split validation set with similar distribution with test set
    # get data distribution
    def get_dist(dataset):
        ds = []; keys = []
        for key in dataset.keys():
            data = np.array(dataset[key]['data'])

            peak = data[:,1:].max(0)
            npeak = data[:,1:].min(0)

            at = np.zeros(4).astype(int)
            nat = np.zeros(4).astype(int)
            for t in range(len(data)):
                for i in range(4):
                    if data[t,i+1] > peak[i] * 0.55 and at[i] == 0:
                        at[i] = t

                    if data[t,i+1] < npeak[i] * 0.55 and nat[i] == 0:
                        nat[i] = t

                if (at != 0).all() and (nat != 0).all():
                    break

            distance_l = (at + nat)//2
            height = np.array([data[at[i],i+1] for i in range(4)]) \
                    - np.array([data[nat[i],i+1] for i in range(4)])
            lamd = np.abs(at - nat)
            # 4 dim each
            ds.append(np.concatenate([distance_l, height, lamd], -1))
            keys.append(key)

        return ds, keys

    train_feat_all, train_keys = get_dist(train_data)
    test_feat_all, test_keys = get_dist(test_data)
    with open('../loc_pred/data/statistics.pkl', 'wb') as f:
        pickle.dump((train_feat_all, test_feat_all), f)

    # save training file for location prediction


    def write_file(fname, dataset):
        xyvt = []
        for key in dataset.keys():
            data = np.array(dataset[key]['data'])

            if 'label' in dataset[key].keys():
                label = np.array(dataset[key]['label'])
                x,y,m,v = label
            else:
                x,y,m,v = None, None, None, None

            peak = data[:,1:].max(0)
            npeak = data[:,1:].min(0)

            at = np.zeros(4).astype(int)
            nat = np.zeros(4).astype(int)
            for t in range(len(data)):
                for i in range(4):
                    if data[t,i+1] > peak[i] * 0.55 and at[i] == 0:
                        at[i] = t

                    if data[t,i+1] < npeak[i] * 0.55 and nat[i] == 0:
                        nat[i] = t

                if (at != 0).all() and (nat != 0).all():
                    break

            distance_l = (at + nat)//2
            height = np.array([data[at[i],i+1] for i in range(4)]) \
                    - np.array([data[nat[i],i+1] for i in range(4)])
            lamd = np.abs(at - nat)

            data_stat = np.concatenate([distance_l, height, lamd], -1)
            # if datastat ...
            #xyvt.append((x,y,v,t))
            xyvt.append(
                (key, x,y,m,v, distance_l, height, lamd)
            )

        with open(fname, 'wb') as f:
            pickle.dump(xyvt, f)

    # split train_val
    # random split
    keys = list(train_data.keys())
    np.random.seed(seed)
    idx = np.random.choice(len(keys), size=int(len(train_data)*0.1), replace=False)
    val_key = [keys[i] for i in idx]

    tr_data = {}
    val_data = {}
    for key in train_data.keys():
        if key in val_key:
            val_data[key] = train_data[key]
        else:
            tr_data[key] = train_data[key]


    # else
    #train_feat_all = np.expand_dims(np.array(train_feat_all), 1)
    #test_feat_all = np.expand_dims(np.array(test_feat_all), 0)
    #
    #mat = ((train_feat_all - test_feat_all)**2).sum(-1)
    #mat = np.min(mat, 1)

    idx = np.argsort(mat)
    val_key = [train_keys[i] for i in idx[:int(len(idx)*0.1)]]
    tr_data = {}
    val_data = {}
    for key in train_data.keys():
        if key in val_key:
            val_data[key] = train_data[key]
        else:
            tr_data[key] = train_data[key]


    write_file('../loc_pred/data/test.pkl', test_data)
    write_file('../loc_pred/data/train.pkl', tr_data)
    write_file('../loc_pred/data/val.pkl', val_data)




    #
