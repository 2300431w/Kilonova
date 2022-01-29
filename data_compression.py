import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle
import random
import time
from DU17_Model import Generate_LightCurve

def compress_file(fname,new_fname,f = 10,troubleshooting = True):
    data = pd.read_pickle(fname)

    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    g = data['g'].values
    
    conditionals = np.vstack((m1,m2,l1,l2)).T
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])

    """while troubleshooting == True:
        #PASS
        #Notes: I was using the wrong lc. mind lc[1][0] is the u-band. the lowest in the data is g
        
        t_d = data['time']
        t_d = np.vstack(t_d)[0]
        i = random.randint(0,len(g))
        curve = g[i]
        m1_,m2_,l1_,l2_ = conditionals[i]
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        #lc = np.nan_to_num(lc)
        #plt.plot(compressed_t_d,band_data[0][i],label = "Compressed Curve",c = "blue")
        plt.plot(lc[0],lc[1][1],":",label = "Model",c = "blue")
        plt.plot(t_d,curve,"-.",ms = 5,label = "Curve",c = "red")
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()"""
    
    band_data = []
    for band in ['g','r','i','z']:
        curve = data[band].values
        curve = np.vstack(curve)

        t_d = data['time']
        t_d = np.vstack(t_d)[0]

        compressed_curve,compressed_t_d = compress_curve(t_d,curve,f)
        band_data.append(compressed_curve)

    #compressed_curve = np.array(compressed_curve)
    
    
    band_data = np.array(band_data)
    """while troubleshooting == True:
        i = random.randint(0,len(m1))
        print(i)
        m1_ = m1[i]
        m2_ = m2[i]
        l1_ = l1[i]
        l2_ = l2[i]
        
        print(m1_,m2_,l1_,l2_)
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        #lc = np.nan_to_num(lc)
        
        curve = data['g'].values
        curve = np.vstack(curve)
        plt.plot(compressed_t_d,band_data[0][i],label = "Compressed Curve",c = "blue")
        plt.plot(lc[0],lc[1][1],":",label = "Model",c = "red")
        plt.plot(t_d,curve[i],"-.",ms = 5,label = "Curve",c = "red")
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()"""
        
    final_data = []
    for i in np.arange(len(band_data[0])):
        
        m1,m2,l1,l2 = conditionals[i]
        
        g = band_data[0][i]
        r = band_data[1][i]
        I = band_data[2][i]
        z = band_data[3][i]
        d = np.array([m1,m2,l1,l2,compressed_t_d,g,r,I,z])
        final_data.append(d)

    
    """while troubleshooting == True:
        #pass
        i = random.randint(0,len(final_data))
        plt.plot(final_data[i][4],final_data[i][5],label = "compressed")
        curve = data["g"].values
        curve = np.vstack(curve)
        plt.plot(t_d,curve[i],":",label = "original")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()"""
    
    print('check 1')
    final_data = np.array(final_data)
    #print(final_data.shape)
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
        if i % f == 0:
            new_column = np.array([np.array(curve[:,i])]).T
            new_t = t_d[i]
            compressed_curve = np.hstack((compressed_curve,new_column))
            compressed_t_d.append(new_t)

    compressed_curve = compressed_curve[:,1:]
    compressed_t_d = np.array(compressed_t_d)
    return(compressed_curve,compressed_t_d)

def crop_index(data,crop_range,searching = False, band = "g"):
    fails = 0
    g = data[band].values
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    nonzero_g = []
    zero_g = []
    
    for i in np.arange(len(g)):
        if np.isnan(g[i]).all():
            fails += 1
            zero_g.append(g[i])
        else:
            nonzero_g.append(g[i])
    #g = []
    nonzero_g = np.array(nonzero_g)
    zero_g = np.array(zero_g)
    print(f'{fails} were empty')

    output = []
    cond_output = []
    if searching == True:
        
        crop_range = len(nonzero_g[0]) - 1
        N = []
        while crop_range >= 10:
            time.sleep(15)
            n = []
            bottom = 0
            top = bottom + crop_range
            
            while top <= len(nonzero_g[0]) - 1:
                n_ = 0
                cropped_temp = nonzero_g[:,bottom:top]
                for g in cropped_temp:
                    if np.isnan(np.min(g)):
                        pass
                    else:
                        n_ += 1
                bottom += 1
                top += 1
                n.append(n_)
            N.append([crop_range,round(100*max(n)/len(nonzero_g),2),n.index(max(n))])
            crop_range -= 50
            print(N[-1])
        N = np.array(N)
        print(N)
    bottom = 0
    top = bottom + crop_range
    n = []
    while top <= len(nonzero_g[0]) - 1:
            n_ = 0
            cropped_temp = nonzero_g[:,bottom:top]
            for g in cropped_temp:
                if np.isnan(np.min(g)):
                    pass
                else:
                    n_ += 1
            bottom += 1
            top += 1
            n.append(n_)
    
    bottom = n.index(max(n))
    top = bottom + crop_range
    cropped_temp = nonzero_g[:,bottom:top]
    cropped_zero = zero_g[:,bottom:top]
    return(bottom,top)
    
def crop_data(fname,crop_range,searching = False,band = 'g',zeroes = False,troubleshooting = False):
    fails = 0
    data = pd.read_pickle(fname)
    g = data[band].values
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    
    nonzero_g = []
    indices = []
    zero_g = []
    M1 = []
    M2 = []
    L1 = []
    L2 = []
    
    M01 = []
    M02 = []
    L01 = []
    L02 = []
    
    for i in np.arange(len(g)):
        if np.isnan(g[i]).all():
            fails += 1
            zero_g.append(g[i])
            
            m1 = data['m1'].values[i]
            m2 = data['m2'].values[i]
            l1 = data['l1'].values[i]
            l2 = data['l2'].values[i]
            
            M01.append(m1)
            M02.append(m1)
            L01.append(l1)
            L02.append(l2)
        else:
            nonzero_g.append(g[i])
            indices.append(i)
            m1 = data['m1'].values[i]
            m2 = data['m2'].values[i]
            l1 = data['l1'].values[i]
            l2 = data['l2'].values[i]
            
            M1.append(m1)
            M2.append(m1)
            L1.append(l1)
            L2.append(l2)
    #g = []
    nonzero_g = np.array(nonzero_g)
    zero_g = np.array(zero_g)
    print(f'{fails} were empty')

    output = []
    cond_output = []

    """while troubleshooting ==True:
        #PASS
        i = random.randint(0,len(nonzero_g))
        g = data[band].values
        plt.plot(t_d,nonzero_g[i],label = "non-zero_g")
        #print(indices[i])
        plt.plot(t_d,g[indices[i]],":",label = "Original")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()"""

    
    if searching == True:
        
        crop_range = len(nonzero_g[0]) - 1
        N = []
        while crop_range >= 10:
            time.sleep(15)
            n = []
            bottom = 0
            top = bottom + crop_range
            
            while top <= len(nonzero_g[0]) - 1:
                n_ = 0
                cropped_temp = nonzero_g[:,bottom:top]
                for g in cropped_temp:
                    if np.isnan(np.sum(g)):
                        pass
                    else:
                        n_ += 1
                bottom += 1
                top += 1
                n.append(n_)
            N.append([crop_range,round(100*max(n)/len(nonzero_g),2),n.index(max(n))])
            
            #CHANGE RATE CROP_RANGE GOES DOWN HERE
            crop_range -= 1
            #------------------------------------#

            
            print(N[-1])
        N = np.array(N)
        print(N)
        plt.plot(N[0][:],N[1][:])
        plt.show()

    #g-band OG data: 450
    
    
    bottom = 0
    top = bottom + crop_range
    n = []
    while top <= len(nonzero_g[0]) - 1:
            n_ = 0
            cropped_temp = nonzero_g[:,bottom:top]
            for g in cropped_temp:
                if np.isnan(np.sum(g)):
                    pass
                else:
                    n_ += 1
            bottom += 1
            top += 1
            n.append(n_)
    
    bottom = n.index(max(n))
    top = bottom + crop_range
    cropped_temp = nonzero_g[:,bottom:top]
    cropped_zero = zero_g[:,bottom:top]

    indices2 = []
    nans = 0
    for i in np.arange(len(cropped_temp)):
        if np.isnan(np.sum(cropped_temp[i])):
            nans += 1
        else:
            cond_output.append([M1[i],M2[i],L1[i],L2[i]])
            indices2.append(indices[i])
            output.append(cropped_temp[i])
    print(f'{len(cropped_temp)} - {nans} = {len(output)}')       
    """if troubleshooting == True:
        #PASS
        g = data[band].values
        while troubleshooting == True:
            i = random.randint(0,len(output))
            plt.plot(t_d[bottom:top],output[i],label = "Cropped")
            plt.plot(t_d,g[indices2[i]],":",label = "Original")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()
            print(np.sum(output[i]))"""
            
    if zeroes == True:
        for i in np.arange(len(cropped_zero)):
            cond_output.append([M01[i],M02[i],L01[i],L02[i]])
            output.append(cropped_zero[i])
    print("beginning to save data")
        
    t_d = t_d[bottom:top]
    output = np.array(output)
    t_d = list(t_d)
        
    #print(list(output[0]))
    
    final_data = []
    
    for i in np.arange(len(output)):
        #print(f'{100*i/len(output):.3g}%')
        m1 = data['m1'].values[indices2[i]]
        m2 = data['m2'].values[indices2[i]]
        l1 = data['l1'].values[indices2[i]]
        l2 = data['l2'].values[indices2[i]]
        #print(m1,m2,l1,l2)
        d = np.array([m1,m2,l1,l2,t_d,output[i]])
        
        final_data.append(d)

    print("data prepped")
    final_data = np.array(final_data)
    
    bandindex = ['g','r','i','z'].index(band) + 1
    if troubleshooting == True:
        g = data[band].values
    while troubleshooting == True:
        
        i = random.randint(0,len(final_data))
        m1,m2,l1,l2 = final_data[i][0],final_data[i][1],final_data[i][2],final_data[i][3]
        t_d = final_data[i][4]
        output = final_data[i][5]
        lc = Generate_LightCurve(m1,m2,l1,l2)[1]
        #lc = np.nan_to_num(lc)
        
        plt.plot(lc[0],lc[1][bandindex],label = "Model",c = "red")
        plt.plot(t_d,output,"--",label = "training data")

        t_d = data['time']
        t_d = np.vstack(t_d)[0]
        plt.plot(t_d,g[indices2[i]],":",label = "Original")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
    
    df = pd.DataFrame(data = final_data,
                          columns = list(['m1','m2','l1','l2','time',band]))
    df.to_pickle(f'Data_Cache/Crop/DU17_{band}_cropped_{crop_range}.pkl')
    print("df saved")
    print(df)
    #print(len(final_data))

#g og not compressed: 450 crop_range
#g og compressed: 46 crop_range
#r og compressed: 84
#i og compressed: 84
#z og compressed: 84
crop_data("Data_Cache/Comp/Comp_10_Original_Combined_comp.pkl",crop_range = 84,searching = False,band = 'z',troubleshooting = False)

#For compressing a file
"""factor = 10
f = "Original_Combined"
fname = "Data_Cache/Original/" + f + ".pkl"
compress_file(fname,f'Comp_{factor:.2g}_{f}',f = factor,troubleshooting = True)"""

