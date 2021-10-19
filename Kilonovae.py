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

    #define a day
    d = 24*60*60
    #split the equations into more manageable parts
    A = (1+theta_ej)*e_th*e_0*M_ej

    if t<=t_c:
        B = (t/t_c)*(t/d)**(-alpha)
    else:
        B = (t/d)**(-alpha)

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

def M_X(t,M1,M2,M_ej,C, a_x, e_th= 1, e_0= 1.58e10, alpha = 1.3,kappa = 10e-2):
    "Magnitude spectrum of band X, Eqn (20) [1]"
    v_p,v_z = velocities(M1,M2,C)
    v_min = 0.02 #they set it to this in [1]
    v_max = vmax(M1,M2,C,v_min = v_min)
    
    theta_ej = theta_ejecta(v_p,v_z)
    
    t_c = t_critical(theta_ej,kappa,M_ej,v_max,v_min)
    
    L = L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha)
    M_x = M_bol(L) - BC_X(t,a_x,M_ej)
    return(M_x)

def t_critical(theta_ej,kappa, M_ej,v_max,v_min):
    "Eqn (16) of [1] to work out the critical time"

    phi_ej = 4*theta_ej + np.pi/2
    numerator = theta_ej*kappa*M_ej
    denomenator = 2*phi_ej*(v_max - v_min)
    t_c = np.sqrt(numerator/denomenator)
    return(t_c)

def vmax(M1,M2,C,v_min = 0.02):
    "returns the maximum velocity, defined just above Sect 4.2.2 in [1]"
    v_ej = v_ejecta(M1,M2,C)
    return(2*v_ej - v_min)

def velocities(M1,M2,C):
    "returns v_rho and v_z"
    #first calculate v_rho ([1] (5))
    
    A = [-0.219479,0.444836,-2.67385] #a,b,c for the v_rho polynomial
    v_rho = (A[0]*(M1/M2)*(1+A[2]*C)) + A[1] # + (1<->2)?? idk what that means in (5)
    # C = compactness

    #similairly calculate v_z
    B = [-0.315585,0.63808,-1.00757]
    v_z = (B[0]*(M1/M2)*(1+B[2]*C)) + B[1]
    return(v_rho,v_z)

def v_ejecta(M1,M2,C):
    "velocity of the ejected mass accourding to (9) in [1]"
    v_rho,v_z = velocities(M1,M2,C)
    #calculate v_ej
    v_ej = np.sqrt(v_rho**2 + v_z**2)
    return(v_ej)

def theta_ejecta(v_p,v_z):
    "The estimate of [1] to get theta_ej"
    a = np.sqrt(9*v_z**2 + 4 * v_p **2) # useful to just do this once and keep equations simple
    numerator = 2**(4/3) * v_p **2 - 2**(2/3)*(v_p**2 * (3*v_z + a))**2/3
    denomenator = (v_p**5 * (3 * v_z + a))**(1/3)
    return(numerator/denomenator)

# e_th in [2]
# M_ej in [2]
# e_0 in [1]
# alpha in [1]
L = []
M = []

#---------------------------Plotting--------------------------#
M1 = 1.35*M_solar
M2 = 1.35*M_solar
M_ej = 12.2e-3*M_solar
C = 0.174
for t in np.linspace(1/1000,7,1000):
    Z = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_z)
    I = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_i)
    R = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_r)
    G = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_g)
    U = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_u)
    K = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C = C, a_x = a_k)
    M.append([Z,I,R,G,U,K])
    #L.append(L_bol(t,theta_ej = 0,e_th = 1,e_0 = 1.58e10,M_ej = 0.1*M_solar, t_c = 10,alpha = 1.3))
M = np.array(M)


labels = [['z',0],['i',1],['r',2],['g',3],['u',4],['k',5]]


ordered = [['g',3],['r',2],['i',1],['z',0]]
for x in np.arange(len(ordered)):
    #plt.subplot(411 + x)
    plt.title(ordered[x][0],loc = "left")
    plt.ylabel("Magnitude")
    plt.xlabel("Time [days]")
    plt.gca().invert_yaxis()
    plt.plot(np.linspace(1/1000,10,1000),M[:,int(ordered[x][1])],label = labels[x][0])


#plt.plot(np.linspace(1/1000,7,1000),L,label = "luminosity")
    
#plt.yscale("log")
#plt.xscale("log")

plt.xlim(2,10)
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.show()

#----------------------------Loading Data----------------------#
Data=False
if Data == True:
    for f in np.linspace(0.01,1.5,1000):
        M_temp = [] #temporary matrix to hold the magnitude line for the fraction of solar mass
        
        M_ej = f*M_solar

        for t in np.linspace(1/1000,7,1000):
            R = M_X(t,a_r, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
            M_temp.append(R)
            
        M_temp = np.array(M_temp)
        M.append(M_temp)

    M = np.array(M)
    mass = np.linspace(0.01,1.5,1000)

    d = []
    for x in np.arange(1000):
        d.append(np.array([mass[x],M[x]]))

    d = np.array(d)
    df = pd.DataFrame(data = d,
                      columns = list(["mass","r-band"]))

    df.to_pickle("r band dataframe.pkl")



#print(df.loc[0,:])
# [1] https://iopscience.iop.org/article/10.1088/1361-6382/aa6bb0/pdf
# [2] https://link.springer.com/content/pdf/10.1007/s41114-017-0006-z.pdf
