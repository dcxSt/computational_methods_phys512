import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define variables to help the eye
minu=60
hour=60*minu
day=24*hour#number of seconds in a day
year=365.25*day

# Half life of ordered products in the decay chain
halflife=np.array([
    4.468*1e9*year, # U238 has a long halflife, idx=0
    24.10*day,
    6.70*hour,
    245500*year, # U234, idx=3
    75380*year, # Th230, idx=4
    1600*year,
    3.8235*day,
    3.10*minu,
    26.8*minu,
    19.9*minu,
    164.3*1e-6, # pb 204 is very small, idx=10
    22.3*year,
    5.015*year,
    138.376*day,
    np.inf # pb 206 is stable, idx=14 (or -1 in python)
    ])

# Dictionary of important indices for easy access
idx = {"U238":0,"U234":3,"Th230":4,"Pb204":10,"Pb206":14}

#tau=np.log(2)/halflife # Tau time constant, diagonal matrix
decay_rate=np.log(2)/halflife

# we can use roll because final product is stable
def decay_timestep(x,y,decay_rate=decay_rate):
    """dy/dx = decay_timestep(x,y)"""
    return -y*decay_rate + np.roll(y*decay_rate,1) 

# Solve the system to figure out how much of the end product is being produced
x0=0
x1=halflife[idx["U238"]] # about 1 half life of U238
y0=np.zeros(decay_rate.shape)
y0[0]=1.0 # start with just uranium
ans=solve_ivp(decay_timestep,[x0,x1],y0,method='Radau',t_eval=np.linspace(x0,x1,1000))
t=ans['t']
y=ans['y']
u238=y[idx["U238"]]
pb206=y[idx["Pb206"]]

# # Plot the ratio of Pb206 to U238 as a function of time.
# plt.figure(figsize=(6,5))
# plt.plot(t,pb206/u238,label="Pb206/U238")
# plt.plot(t,u238,label="U238")
# plt.plot(t,pb206,label="Pb206")
# plt.title("Pb206 / U238")
# plt.legend()
# plt.ylabel("Ratio #Pb206 / #U238, and normalized # of each")
# plt.xlabel("Time, in seconds")
# plt.tight_layout()
# plt.savefig("img/u238_pb206.png",dpi=500)
# plt.show(block=True)

# Thorium 230 has a faster decay rate
x0=0
x1=30*halflife[idx["Th230"]]
y0=np.zeros(decay_rate.shape)
y0[0]=1.0 # start with just uranium
ans=solve_ivp(decay_timestep,[x0,x1],y0,method='Radau',t_eval=np.linspace(x0,x1,1000*100))
t=ans['t']
y=ans['y']
u238=y[idx["U238"]]
th230=y[idx["Th230"]]

# Plot the ratio for Th230 and U238
plt.figure(figsize=(6,4))
plt.plot(t/halflife[idx["Th230"]],th230/u238,label="th230/U238")
# plt.plot(t,u238,label="U238")
# plt.plot(t,th230,label="Th230")
plt.title("Th230 / U238")
plt.legend()
plt.ylabel("Ratio #Th230 / #U238")
plt.xlabel("Time, in halflives of Th230")
plt.tight_layout()
plt.savefig("img/u238_th230.png",dpi=500)
plt.show(block=True)




