import numpy as np

import matplotlib
#matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from matplotlib import cm

from glasflow import RealNVP
import torch
from torch import optim

import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DU17_Model import Generate_LightCurve



def plot_model(i,n,conditional,band,bandindex,t_d,scaling_constant,curve):
    c = cmap(i/n)
    test_array = []
    for j in np.arange(n):
        temp = conditional[j]
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)
    #print(conditional)
    with torch.no_grad():
        samples = flow.sample(n,conditional = conditional)
    samples = samples.cpu().numpy()
    conditional = conditional.cpu().numpy()

    m1,m2,l1,l2 = conditional[i]
    lc = curve[i]*scaling_constant
    
    lc = np.nan_to_num(lc)

    
    m1,m2,l1,l2 = conditional[i]
    #print(m1,m2,l1,l2)
    lc_gen = Generate_LightCurve(m1,m2,l1,l2)[1]
    lc_gen = np.nan_to_num(lc_gen)
    
    plt.plot(t_d,lc,label = f"Model[{i}]",c = c)
    plt.plot(lc_gen[0],lc_gen[1][bandindex+1],linestyle = "--",label = f"Model OG[{i}]",c = c)
    plt.plot(t_d,scaling_constant*samples[i],".",ms =5,label = f"Flow[{i}]",c = c)
    
def random_data(N,band,file,N_Samples = 100):

    fname = "Data_Cache/Original/Original_combined.pkl"
    data = pd.read_pickle(fname)
    data = shuffle(data)
    
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)

    scaling_constant = np.min(curve)
    print(f'scaling_constant {band}: {scaling_constant}')
    
    #Load Data
    bandindex = ['g','r','i','z'].index(band) + 1
    
    fname = file
    print(f'file: {fname}')

    data = pd.read_pickle(fname)
    data = shuffle(data)
        
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)
        
    curve = curve/scaling_constant
    try:
        assert np.max(curve) <= 1.0
    except:
        print("Curve not normalised correctly, curve max was:\t",np.max(curve))
    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    conditional = np.vstack((m1,m2,l1,l2)).T
    print(len(m1)," Data points")
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])
        
    plt.title(f'{band} Model Test')
    plt.ylabel("Absolute Magnitude")
    plt.xlabel("Time")

    test_array = []
    indices = []
    for n in np.arange(N):
        i = random.randint(0,len(m1))
        indices.append(i)
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)

    Big_Samples = []
    
    with torch.no_grad():
        for i in np.arange(N_Samples):
            samples  = flow.sample(N,conditional = conditional)
            Big_Samples.append(samples)
    for i in np.arange(len(Big_Samples)):
        Big_Samples[i] = Big_Samples[i].cpu().numpy()
    Big_Samples = np.array(Big_Samples)

    final_samples = np.mean(Big_Samples,axis = 0)
    print(final_samples.shape)
    std = np.std(Big_Samples,axis = 0)
    max_lines = final_samples + std #np.max(Big_Samples,axis = 0)
    min_lines = final_samples - std #np.min(Big_Samples,axis = 0)

    conditional = conditional.cpu().numpy()

    cmap = cm.get_cmap('viridis')
    for n in np.arange(N):
        col = cmap(n/N)
        m1_,m2_,l1_,l2_ = conditional[n]
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        lc = np.nan_to_num(lc)
        plt.plot(lc[0],lc[1][bandindex],"--",label = f"Model[{n}]",c = col)
        plt.plot(t_d,scaling_constant*final_samples[n],"-",ms =4,label = f"Flow[{n}]",c = col)
        plt.fill_between(t_d,min_lines[n]*scaling_constant,max_lines[n]*scaling_constant,alpha = 0.2,color = col)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

def time_model(N,band,file,N_Samples = 10000):
    fname = "Data_Cache/Original/Original_combined.pkl"
    data = pd.read_pickle(fname)
    data = shuffle(data)
    
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)

    scaling_constant = np.min(curve)
    print(f'scaling_constant {band}: {scaling_constant}')
    
    #Load Data
    bandindex = ['g','r','i','z'].index(band) + 1
    
    fname = file
    print(f'file: {fname}')

    data = pd.read_pickle(fname)
    data = shuffle(data)
        
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)
        
    curve = curve/scaling_constant
    try:
        assert np.max(curve) <= 1.0
    except:
        print("Curve not normalised correctly, curve max was:\t",np.max(curve))
    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    conditional = np.vstack((m1,m2,l1,l2)).T
    print(len(m1)," Data points")
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])
    
    
    test_array = []
    indices = []
    
    for n in np.arange(N):
        i = random.randint(0,len(m1))
        indices.append(i)
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)
    Big_Samples = []
    print("Data Prepared")
    ML_start = time.time()
    
    with torch.no_grad():
        for i in np.arange(N_Samples):
            samples  = flow.sample(N,conditional = conditional)
            Big_Samples.append(samples)
    for i in np.arange(len(Big_Samples)):
        Big_Samples[i] = Big_Samples[i].cpu().numpy()
    Big_Samples = np.array(Big_Samples)

    final_samples = np.mean(Big_Samples,axis = 0)
    print(final_samples.shape)
    std = np.std(Big_Samples,axis = 0)
    max_lines = final_samples + std #np.max(Big_Samples,axis = 0)
    min_lines = final_samples - std #np.min(Big_Samples,axis = 0)
    ML_end = time.time()
    
    ML_time = ML_end - ML_start
    print("Machine Done")
    conditional = conditional.cpu().numpy()
    Model_start = time.time()
    for n in np.arange(N):
        print(f'{100*n/N}%')
        col = cmap(n/N)
        m1_,m2_,l1_,l2_ = conditional[n]
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        lc = np.nan_to_num(lc)
    Model_end = time.time()
    
    Model_time = Model_end - Model_start
    print(f'N_Samples = {N_Samples}')
    print(f'Machine took: {ML_time:.5f}ms')
    print(f'Model took: {Model_time:.5f}ms')
    print(f'Difference: {Model_time - ML_time} (Model - ML)')
    return(Model_time,ML_time)
#Define the Model
model_path = "Models/model_0_z.pth"
flow = torch.load(model_path,map_location = torch.device('cpu'))
device = torch.device('cpu')
flow.to(device)
flow.eval()

#colourmaps definitions
cmap =cm.get_cmap('hsv')
plt.style.use('seaborn-colorblind')



#while True:
#    random_data(3,"z","Data_Cache/Crop/DU17_z_cropped_84.pkl",100)
Mt = []
MLt = []
for n in np.arange(0,1000,10):
    print(f'========{n}=========')
    mt, mlt = time_model(n,"z","Data_Cache/Crop/DU17_z_cropped_84.pkl",100)
    Mt.append(mt)
    MLt.append(mlt)
Mt = np.array(Mt)
MLt = np.array(MLt)
np.save("Mt.npy",Mt)
np.save("MLt.npy",MLt)
plt.plot(mt,",")
plt.plot(MLt,",")
plt.show()
