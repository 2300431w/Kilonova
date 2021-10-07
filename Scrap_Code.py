import numpy as np
import matplotlib.pyplot as plt

###----------------For Plotting All bands of M_X-----------#

# e_th in [2]
# M_ej in [2]
# e_0 in [1]
# alpha in [1]
L = []
M = []

M_ej = 0.1*M_solar

for t in np.linspace(1/1000,7,1000):
    Z = M_X(t,a_z, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    I = M_X(t,a_i, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    R = M_X(t,a_r, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    G = M_X(t,a_g, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    U = M_X(t,a_u, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    K = M_X(t,a_k, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
    M.append([Z,I,R,G,U,K])
    L.append(L_bol(t,theta_ej = 0,e_th = 1,e_0 = 1.58e10,M_ej = 0.1*M_solar, t_c = 10,alpha = 1.3))
M = np.array(M)


labels = [['z',0],['i',1],['r',2],['g',3],['u',4],['k',5]]


ordered = [['g',3],['r',2],['i',1],['z',0]]
for x in np.arange(len(ordered)):
    plt.subplot(411 + x)
    plt.title(ordered[x][0],loc = "left")
    plt.ylabel("Magnitude")
    plt.xlabel("Time [days]")
    plt.gca().invert_yaxis()
    plt.plot(np.linspace(1/1000,10,1000),M[:,int(ordered[x][1])])


#plt.plot(np.linspace(1/1000,7,1000),L,label = "luminosity")
    
#plt.yscale("log")
#plt.xscale("log")

#plt.gca().invert_yaxis()
#plt.legend()
plt.tight_layout()
plt.show()
