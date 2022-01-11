import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle


def compress_file(fname,new_fname,f = 10):
    data = pd.read_pickle(fname)
    data = shuffle(data)
    #compressed_data = pd.DataFrame.copy(data)

    band_data = []
    for band in ['g','r','i','z']:
        curve = data[band].values
        curve = np.vstack(curve)
        curve = np.nan_to_num(curve)

        t_d = data['time']
        t_d = np.vstack(t_d)[0]

        compressed_curve,compressed_t_d = compress_curve(t_d,curve,f)
        
        #compressed_data[band] = compressed_curve
        band_data.append(compressed_curve)
    
    #print(data['time'],compressed_t_d)

    final_data = []
    for i in np.arange(len(band_data[0])):
        
        m1,m2,l1,l2 = data['m1'][i],data['m2'][i],data['l1'][i],data['l2'][i]
        
        g = band_data[0][i]
        r = band_data[1][i]
        I = band_data[2][i]
        z = band_data[3][i]
        d = np.array([m1,m2,l1,l2,compressed_t_d,g,r,I,z])
        final_data.append(d)

    print('check 1')
    final_data = np.array(final_data)
    print('check 2')
    df = pd.DataFrame(data = final_data,
                          columns = list(['m1','m2','l1','l2','time','g','r','i','z']))
    print('check 3')

    
    
    df.to_pickle(f'Data_Cache/Comp/{new_fname}_comp.pkl')
    print(fname," compressed")
    

def compress_curve(t_d,curve,f):
    L = len(curve[0])

    compressed_curve = np.zeros((1,len(curve))).T
    compressed_t_d = []


    for i in np.arange(L):
        if i % 10 == 0:
            new_column = np.array([np.array(curve[:,i])]).T
            new_t = t_d[i]
            compressed_curve = np.hstack((compressed_curve,new_column))
            compressed_t_d.append(new_t)

    compressed_curve = compressed_curve[:,1:]
    compressed_t_d = np.array(compressed_t_d)
    return(compressed_curve,compressed_t_d)

training_data = os.listdir("Data_Cache/Original/")
print(training_data)
for f in training_data:
    fname = "Data_Cache/Original/" + f
    compress_file(fname,f)
