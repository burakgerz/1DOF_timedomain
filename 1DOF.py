# -*- coding: utf-8 -*-

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Mass, Spring, Damper
m = 9.62E-7
k = 1.0
d = 1.0E-5

eigfreq = np.sqrt(k/m)/2/np.pi

# Excitation
A = 0.5
f = eigfreq
omega = 2.0*np.pi*f

# Contact
eta = 5e-4
Fn = 3.5
mu = 0.12

# Integration settings
start_time = 0.0
end_time = 0.1
integration_points = f * end_time * 50  #50 per Period
integration_points = int(integration_points) + 1
t = np.linspace(start_time, end_time, integration_points)

# Standard coulomb contact function
def contact (v, F):
    if abs(v) <= eta: 
        FrictionForce = (F*mu/eta)*v 
    else:
        FrictionForce = F * mu * np.sign(v)
    return FrictionForce

# System
def dF(x , t):
    x1 = x[0]
    y1 = x[1]
    y1_prime = (- d*y1 - k*x1 - contact(y1, Fn) + A*np.sin(omega*t)) / m
    return [y1, y1_prime]


# Initial condition
x0 = [0, 0]

sol = odeint(dF, x0, t, printmessg = 1)
x1 = sol[:, 0]
y1 = sol[:, 1]


# Plot Solution
plt.figure()
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.plot(t, x1, label='x1: Displacement')
plt.legend(loc="lower right")
plt.show()