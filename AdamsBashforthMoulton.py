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

def PredictorCorrector(xt, yt, zt,n = 3500, T = 35):
    x = np.zeros(n+2)
    y = np.zeros(n+2)
    z = np.zeros(n+2)
    t = np.zeros(n+2)
    
    x[0] = 8.0
    y[0] = 1.0
    z[0] = 1.0
    t[0] = 0
    dt = T/float(n) #0.01
    
    x, y, z, t = RungeKutta4(xt, yt, zt)
    
    f0_dx = xt(x[0],y[0],z[0],t[0])
    f0_dy = yt(x[0],y[0],z[0],t[0])
    f0_dz = zt(x[0],y[0],z[0],t[0])
    
    f1_dx = xt(x[1],y[1],z[1],t[1])
    f1_dy = yt(x[1],y[1],z[1],t[1])
    f1_dz = zt(x[1],y[1],z[1],t[1])
    
    f2_dx = xt(x[2],y[2],z[2],t[2])
    f2_dy = yt(x[2],y[2],z[2],t[2])
    f2_dz = zt(x[2],y[2],z[2],t[2])
    
    f3_dx = xt(x[3],y[3],z[3],t[3])
    f3_dy = yt(x[3],y[3],z[3],t[3]) 
    f3_dz = zt(x[3],y[3],z[3],t[3])
    
    for k in range(n-1,0,-1):
        #Predictor: The fourth-order Adams-Bashforth technique, an explicit four-step method:
        x[k+1] = x[k] + (dt/24) *(55*f3_dx - 59*f2_dx + 37*f1_dx - 9*f0_dx)
        y[k+1] = y[k] + (dt/24) *(55*f3_dy - 59*f2_dy + 37*f1_dy - 9*f0_dy)
        z[k+1] = z[k] + (dt/24) *(55*f3_dz - 59*f2_dz + 37*f1_dz - 9*f0_dz)
        
        f4_dx = xt(x[k+1],y[k+1],z[k+1],t[k+1])
        f4_dy = yt(x[k+1],y[k+1],z[k+1],t[k+1])
        f4_dz = zt(x[k+1],y[k+1],z[k+1],t[k+1])
        
        #Corrector: The fourth-order Adams-Moulton technique, an implicit three-step method:              
        x[k+1] = x[k] + (dt/24) *(9*xt(x[k+1],y[k+1],z[k+1],t[k+1]) + 19*f3_dx - 5*f2_dx + f1_dx)
        y[k+1] = y[k] + (dt/24) *(9*yt(x[k+1],y[k+1],z[k+1],t[k+1]) + 19*f3_dy - 5*f2_dx + f1_dy)
        z[k+1] = z[k] + (dt/24) *(9*yt(x[k+1],y[k+1],z[k+1],t[k+1]) + 19*f3_dz - 5*f2_dx + f1_dz)
    
    return x, y, z, t
