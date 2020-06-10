import numpy 
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pdb
import librosa

sig = np.load('sample_signal.npy')
f, t, sxx = signal.spectrogram(sig[:,0], fs=10)
#plt.pcolormesh(t, f, np.log(sxx))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.savefig('hello.png')

f, axes = plt.subplots(8,1)
f.set_size_inches((10,5))

for i in range(4):
    axes[i].plot(np.arange(len(sig)), sig[:,i])

for i in range(4,8):
#    f, t, sxx = signal.spectrogram(sig[:,i-4], fs=200, nperseg=50)
#    print (sxx.shape)
#    axes[i].pcolormesh(t, f, np.log(sxx))
    feat = np.fft.fft(sig[:,i-4])
    pdb.set_trace()


test = np.random.rand(2000)
a,b,c = signal.spectrogram(test, fs=1)

plt.savefig('hello.png')


pdb.set_trace()
