#Academic work of Numerical Differential Equations - Federal University of Minas Gerais
""" Utils """
import math, sys 
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

def Euler(xt, yt, zt, n = 3500,T = 35):
    """Solve y’= f(y,t), y(0)=y0, with n steps until t=T."""
    x = np.zeros(n+1) # x[k] is the solution at time t[k]
    y = np.zeros(n+1) # y[k] is the solution at time t[k]
    z = np.zeros(n+1) # z[k] is the solution at time t[k]
    t = np.zeros(n+1) 
    
    x[0] = 8.0
    y[0] = 1.0
    z[0] = 1.0
    t[0] = 0
    dt = T/float(n) #0.1
    
    for k in range(n):
        t[k+1] = t[k] + dt
        x[k+1] = x[k] + dt*xt(x[k], y[k], z[k], t[k])
        y[k+1] = y[k] + dt*yt(x[k], y[k], z[k], t[k])
        z[k+1] = z[k] + dt*zt(x[k], y[k], z[k], t[k])
    return x, y, z, t
