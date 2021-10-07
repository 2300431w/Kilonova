import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Global variables
M_solar = 1.988e30
L_solar = 3.846e33 #e26 Watts e33 ergs per second

#DU17 model variables, Eqn (22a-f) [1]
a_z = [1.072,0.3646,-0.122,0.00793,-0.000126]
a_i = [0.6441,0.0796,-0.122,0.00793,-0.000122]
a_r = [-2.308,1.495,-0.5740,0.0531,-0.00152] #Coefficients for r-band Bolometric Corrections
a_g = [-6.195,4.054,-1.754,0.2246,-0.009813]
a_u = [40.01,-56.79,25.73,-5.207,0.3813]
a_k = [-7.876,3.245,-0.3946,0.0216,-0.000443]


#DU17 Functions

def L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha):
    "Returns Bolometric Luminosity for a lightcurve, Eqn (17) [1]"
    # (1+θ_ej)Ɛ_th* d/dt(Ɛ_0) * M_ej * { (t/t_c)(t/1day)**(-α), t<= t_c [1]
    #                                  { (t/1day)**(-α), t > t_c

    #split the equations into more manageable parts
    A = (1+theta_ej)*e_th*e_0*M_ej

    if t<=t_c:
        B = (t/t_c)*(t)**(-alpha)
    else:
        B = (t)**(-alpha)

    return(A*B)

def M_bol(L):
    "Returns Bolometric Magnitude, Eqn (18) [1]"
    M = 4.74 - 2.5 * np.log10(L/L_solar) #relation from [1]
    return(M)

def BC_X(t,a_x,M_ej):
    "A function which returns the bolometric correction for the bandwidth M_x, Eqn (21) [1]"
    f = (10e-2 * M_solar )/M_ej #the ratio to convert between t and t' as given in [1]
    a0,a1,a2,a3,a4 = a_x
    
    BC = a0 + a1*f*t + a2*f*t**2 + a3*f*t**3 + a4*f*t**4
    
    return(BC)

def M_X(t, a_x, M_ej, theta_ej, e_th, e_0, t_c, alpha):
    "Magnitude spectrum of band X, Eqn (20) [1]"
    L = L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha)
    M_x = M_bol(L) - BC_X(t,a_x,M_ej)
    return(M_x)

# e_th in [2]
# M_ej in [2]
# e_0 in [1]
# alpha in [1]
L = []


M = []
for f in np.linspace(0.01,1.5,100):
    M_temp = [] #temporary matrix to hold the magnitude line for the fraction of solar mass
    
    M_ej = f*M_solar

    for t in np.linspace(1/1000,7,1000):
        R = M_X(t,a_r, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
        M_temp.append(R)
        
    M_temp = np.array(M_temp)
    M.append(M_temp)

M = np.array(M)
mass = np.linspace(0.01,1.5,100)

d = []
for x in np.arange(100):
    d.append(np.array([mass[x],M[x]]))

d = np.array(d)
df = pd.DataFrame(data = d,
                  columns = list(["mass","r-band"]))

df.to_pickle("r band dataframe.pkl")

#print(df.loc[0,:])
# [1] https://iopscience.iop.org/article/10.1088/1361-6382/aa6bb0/pdf
# [2] https://link.springer.com/content/pdf/10.1007/s41114-017-0006-z.pdf
