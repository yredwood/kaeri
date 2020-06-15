import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
import scipy.signal as sig

import pdb


np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

def get_data(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    output = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        else:
            oid, time, s1, s2, s3, s4 = line.split(',')
            oid = int(oid)

            # we don't use time
            data_t = np.stack(
                [float(s1), float(s2), float(s3), float(s4)]
            )
            if oid in output.keys():
                output[oid]['dynamic'].append(data_t)
            else:
                output[oid] = {}
                output[oid]['dynamic'] = [data_t]
    return output


def get_label(fname, dataset):
    '''
    get label and append to dataset
    '''
    with open(fname, 'r') as f: 
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 0:
            continue
        else: 
            oid, x, y, m, v = line.split(',')
            oid = int(oid)
            if oid in dataset.keys():
                dataset[oid]['label'] = np.stack(
                    [float(x), float(y), float(m), float(v)]
                )
            else:
                print ('warning : oid not exists')
    return dataset


def get_static_data(dataset):
    static_data = []

    for key in dataset.keys():
        data = np.array(dataset[key]['dynamic'])

        if 'label' in dataset[key].keys():
            label = np.array(dataset[key]['label'])
        else:
            label = None

        # get static features
        peak = data.max(0)
        npeak = data.min(0)

        starting_peak_time = np.zeros(4).astype(int)
        starting_npeak_time = np.zeros(4).astype(int)

        signal_arrival_time = np.zeros(4).astype(int)

        diff = data[:-1] - data[1:]
        max_cnt_peak = 4
        min_peak_value = 1e+2
        local_peak_ts = np.zeros((max_cnt_peak,4)).astype(int)
        local_peak_vs = np.zeros((max_cnt_peak,4)).astype(float)
        cnt_peak = np.zeros(4).astype(int)
    
        for t in range(len(data)-2):
            for i in range(4):
                if data[t,i] > peak[i] * 0.55 and starting_peak_time[i] == 0:
                    starting_peak_time[i] = t

                if data[t,i] < npeak[i] * 0.55 and starting_npeak_time[i] == 0:
                    starting_npeak_time[i] = t
                
                if data[t,i] != 0 and signal_arrival_time[i] == 0:
                    signal_arrival_time[i] = t

                # choosing local optima 
                # values should exceed 10, and diff should be changed
                if diff[t,i] * diff[t+1,i] < 0 and abs(data[t,i]) > min_peak_value and cnt_peak[i] < max_cnt_peak:

                    local_peak_ts[cnt_peak[i],i] = t + 1
                    local_peak_vs[cnt_peak[i],i] = data[t+1,i]
                    cnt_peak[i] += 1
            
            if (starting_peak_time != 0).all() and (starting_npeak_time != 0).all() and cnt_peak.sum() == max_cnt_peak * 4:
                break


        #signal_arrival_time = (starting_peak_time + starting_npeak_time) // 2

        amplitude = np.array([data[starting_peak_time[i],i] for i in range(4)]) \
                - np.array([data[starting_npeak_time[i],i] for i in range(4)])

        wavelen = np.abs(starting_peak_time - starting_npeak_time)

        peak_data = np.array([data[starting_peak_time[i],i] for i in range(4)])
        npeak_data = np.array([data[starting_npeak_time[i],i] for i in range(4)])

#        fft_feature = []
#        for i in range(4):
#            _d = data[int(signal_arrival_time[i]):,i]
#            f, t, _ftf = sig.spectrogram(_d, fs=10, nperseg=20)
#            fft_feature.append(np.mean(np.log(_ftf), -1))
#        fft_feature = np.array(fft_feature).flatten()
        
#        # plottign
#        # =====================
#        f, axes = plt.subplots(4,1)
#        f.set_size_inches((10,5))
#        
#        for i in range(4):
#            axes[i].plot(np.arange(len(data)), data[:,i])
#            axes[i].plot(local_peak_ts[:,i], local_peak_vs[:,i], 'r+')
#            axes[i].set_title('Key: {}'.format(key))
#        
#        plt.savefig('hehehehe_{}.png'.format(key))
#        plt.close()
#        # ==============================
#        pdb.set_trace()
        

        static_data = np.concatenate([signal_arrival_time, amplitude, wavelen, 
            peak_data, npeak_data, starting_peak_time, starting_npeak_time,
            local_peak_ts.reshape(-1), local_peak_vs.reshape(-1)])

        dataset[key]['static'] = static_data
        dataset[key]['dynamic'] = data

    return dataset

def _get_static_data(dataset):
    static_data = []
    for key in dataset.keys():
        data = np.array(dataset[key]['dynamic'])

        if 'label' in dataset[key].keys():
            label = np.array(dataset[key]['label'])
        else:
            label = None


        # get static features
        peak = data.max(0)
        npeak = data.min(0)

        starting_peak_time = np.zeros(4).astype(int)
        starting_npeak_time = np.zeros(4).astype(int)

        for t in range(len(data)):
            for i in range(4):
                if data[t,i] > peak[i] * 0.55 and starting_peak_time[i] == 0:
                    starting_peak_time[i] = t

                if data[t,i] < npeak[i] * 0.55 and starting_npeak_time[i] == 0:
                    starting_npeak_time[i] = t

            if (starting_peak_time != 0).all() and (starting_npeak_time != 0).all():
                break
        
        signal_arrival_time = (starting_peak_time + starting_npeak_time) // 2
        amplitude = np.array([data[starting_peak_time[i],i] for i in range(4)]) \
                - np.array([data[starting_npeak_time[i],i] for i in range(4)])

        wavelen = np.abs(starting_peak_time - starting_npeak_time)
        
#        fft_feature = []
#        for i in range(4):
#            _d = data[int(signal_arrival_time[i]):,i]
#            f, t, _ftf = sig.spectrogram(_d, fs=10, nperseg=20)
#            fft_feature.append(np.mean(np.log(_ftf), -1))
#        fft_feature = np.array(fft_feature).flatten()

        static_data = np.concatenate([signal_arrival_time, amplitude, wavelen])
        dataset[key]['static'] = static_data
        dataset[key]['dynamic'] = data

    return dataset

def visualize_static(dataset, key, output_path):
    dynamic = dataset[key]['dynamic']
    static = dataset[key]['static']

    f, axes = plt.subplots(4,1)
    f.set_size_inches((10,5))
    
    for i in range(4):
        axes[i].plot(np.arange(len(dynamic)), dynamic[:,i])
        axes[i].plot(int(static[i]), 0, 'r+')
        axes[i].set_title('Key: {}'.format(key))
    
    plt.savefig(output_path)
    plt.close()


def split_by_seed(dataset, seed, train_ratio):
    keys = list(dataset.keys())
    if seed != 99:
        print ('Split seed: {}'.format(seed))
        np.random.seed(seed)
        idx = np.random.choice(len(keys), size=int(len(keys)*train_ratio), replace=False)
        train_key = [keys[i] for i in idx]
        tr_data = {}
        val_data = {}
        for key in dataset.keys():
            if key in train_key:
                tr_data[key] = dataset[key]
            else:
                val_data[key] = dataset[key]

    else:
        # if seed = 25
        pass

    return tr_data, val_data

def split_by_fold(dataset, fold):
    keys = list(dataset.keys())
    assert fold < 10
    datalen = int(len(keys)*0.1)
    val_keys = keys[fold*datalen:(fold+1)*datalen]
    tr_data = {}
    val_data = {}
    for key in dataset.keys():
        if key in val_keys:
            val_data[key] = dataset[key]
        else:
            tr_data[key] = dataset[key]
    return tr_data, val_data


if __name__ == '__main__':
    
    try:
        seed = int(sys.argv[1])
    except:
        seed = 99

    data_root = '/nfs/maximoff/ext01/mike/tmp/nocad/KAERI_dataset/raw'
    output_root = './data/seed_{}'.format(seed)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 1. load dynamic data
    train_dataset = get_data(os.path.join(data_root, 'train_features.csv'))
    test_dataset = get_data(os.path.join(data_root, 'test_features.csv'))

    # 2. add label
    train_dataset = get_label(os.path.join(data_root, 'train_target.csv'), train_dataset)

    # 3. add static features
    train_dataset = get_static_data(train_dataset)
    test_dataset = get_static_data(test_dataset)
    
#    for i in test_dataset.keys():
#        visualize_static(test_dataset, i, 'plot{}.png'.format(i))
    
    tr_data, val_data = split_by_seed(train_dataset, seed, 0.8)
    #tr_data, val_data = split_by_fold(train_dataset, seed)
    
    with open(os.path.join(output_root, 'train.pkl').format(seed), 'wb') as f:
        pickle.dump(tr_data, f)

    with open(os.path.join(output_root, 'test.pkl').format(seed), 'wb') as f:
        pickle.dump(test_dataset, f)

    with open(os.path.join(output_root, 'val.pkl').format(seed), 'wb') as f:
        pickle.dump(val_data, f)
