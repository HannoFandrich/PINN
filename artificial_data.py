import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.integrate import solve_ivp
from collections import defaultdict
import time

import tensorflow as tf
from tensorflow import keras


u_init = [1, 0]
params = [-2, 0]
t_span=[0,10]
datapoints=1000

def simulate_ODEs(u_init, params, t_span, datapoints):

    # Define the target ODEs
    def odes(t, u, p):
        du1dt = np.exp(-t/10) * u[1] * u[2]
        du2dt = u[0] * u[2]
        du3dt = p[0] * u[0] * u[1] +p[1]
        return [du1dt, du2dt, du3dt]

    # Solve ODEs
    t_eval = np.linspace(t_span[0], t_span[1], datapoints)
    sol = solve_ivp(odes, t_span, u_init, args=(params,), method='RK45', t_eval=t_eval)

    return u_obs

u_obs=simulate_ODEs(u_init, params, t_span, datapoints)
np.savetxt('data/data.csv', u_obs, delimiter=',', header='t,u1,u2,u3', comments='')

u_obs_test=simulate_ODEs(u_init, params, t_span, datapoints)
np.savetxt('data/data_test.csv', u_obs, delimiter=',', header='t,u1,u2,u3', comments='')