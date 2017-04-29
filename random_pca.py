import neurolab as nl
import os
import scipy.io as sio
import numpy as np
import pywt
import re
import random
from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import nolds
    
def random_file(filelist):
        rfile = random.sample(filelist, 2*int(len(filelist)/3))
        lfile = set(filelist) - set(rfile)
        lfile = list(lfile)
        print 'random ok'
        return rfile,lfile
        
def get_file():
        N = re.compile('N')
        path = 'F:\data'
        a = list(os.walk(path))
        a = a[0]
        a = a[2]
        rfile,lfile = random_file(a)#lfile=left file rfile=select file
        array_1 = []
        array_2 = []
        n = 0
        d = []
        d2 = []
       
        for i in rfile: 
                x = i.split('.',1)
                if x[1]=='mat':
                        if bool(N.match(x[0]))==True:
                                i = str(i)
                                a = main(i)
                                if a is None:
                                        continue
                                else:
                                        array_1.append(a)
                                        d.append(1)
                        else:
                                i = str(i)
                                a = main(i)
                                if a is None:
                                        continue
                                else:
                                        array_1.append(a)
                                        d.append(-1)
                else:
                        pass
        for i in lfile: 
                x = i.split('.',1)
                if x[1]=='mat':
                        if bool(N.match(x[0]))==True:
                                i = str(i)
                                a = main(i)
                                if a is None:
                                        continue
                                else:
                                        array_2.append(a)
                                        d2.append(1)
                        else:
                                i = str(i)
                                a = main(i)
                                if a is None:
                                        continue
                                else:
                                        array_2.append(a)
                                        d2.append(-1)
                else:
                        pass
        array_1 = np.array(array_1)
        d = np.array(d)
        
        d2 = np.array(d2)
        
        return array_1,d,array_2,d2

def get_array(name):
    channel_num = 16
    array = []
    array_value = 0
    array_1 = []
    s = 0
    n = sio.loadmat(name)
    egMatrix = (n["dataStruct"][["data"]])[0][0][0]
    egMatrix = egMatrix.T
    egSamplingRate = list(n["dataStruct"][["iEEGsamplingRate"]])[0][0][0][0][0]
    egDataLength = list(n["dataStruct"][["nSamplesSegment"]])[0][0][0][0][0]
    egChannels = list(n["dataStruct"][["channelIndices"]])[0][0][0][0]
    return egMatrix.T
def get_corrcoef(R,S):# caluclate the corrcoef of two lines H/S are 2D_array
    c = []
    for i in range(len(R[0])):# R is row EEG data S is fastICA data after pca
        a = []
        for j in range(len(S[0])):
            a.append((np.corrcoef(R[:,i],S[:,j]))[0][1])
        c.append([a])
    c = np.array(c,dtype='float64')
    c = c.reshape(16,16)
    temp = []
    for x in range(16):
        hhh = list(c[x])
        
        temp.append(hhh.index(max(hhh)))
        temp.append(hhh.index(min(hhh)))
    temp = np.array(temp)
    temp.reshape(16,2)

    return temp 
            
def main(name):    
    egMatrix = get_array(name)

    pca = PCA(n_components=8,whiten=True)

    H = pca.fit_transform(egMatrix)


    ica = FastICA(whiten=False)

    S = ica.fit_transform(H)
    c = []
    for i in range(8):
        c.append(nolds.hurst_rs(H[:,i]))
    a = max(c)
    b = min(c)
    d = np.average(c)
    e = []
    for j in c:
        f = (j - d)/(a - b)
        e.append(f)
    print e
    return e

array_1,d,array_2,d2 = get_file()
mytarget = d.reshape(len(d),1)
myinput = array_1
bpnet = nl.net.newff([[-1,1] for i in range(8)],[12,1])
err = bpnet.train(myinput,mytarget,epochs=3000,show=100,goal=0.02)
bpnet.save('1st.net')
answer = bpnet.sim(array_2)
print answer
print d2
  
