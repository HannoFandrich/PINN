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


y_init = [1, 0]
params = [6.77, 1.01, 1.26, 5.11]
t_span=[0,10]
obs_num=100

def simulate_ODEs(u_init, params, t_span, obs_num):
    """Simulate the ODE system and obtain observational data.

    Args:
    ----
    u_init: list of initial condition for u1, u2, and u3
    t_span: lower and upper time limit for simulation
    obs_num: number of observational data points

    Outputs:
    --------
    u_obs: observed data for u's
    """

    # Define the target ODEs
    def odes(t, u, p):
        du1dt = p[0]-u[0]-(p[1]*u[0])*(1+p[2]*u[1]**4)
        du2dt = (p[1]*u[0])*(1+p[2]*u[1]**4)-p[3]*u[1]
        return [du1dt, du2dt]

    # Solve ODEs
    t_eval = np.linspace(t_span[0], t_span[1], obs_num)
    sol = solve_ivp(odes, t_span, u_init, args=(params,), method='RK45', t_eval=t_eval)


    # Restrcture obtained data
    th_point=1
    u_obs = np.column_stack((sol.t[::th_point], sol.y[0,::th_point] , sol.y[1,::th_point]))

    return u_obs

u_obs=simulate_ODEs(y_init,
                    params,
                    t_span,
                    obs_num)
np.savetxt('data/data.csv', u_obs, delimiter=',', header='t,u0,u1', comments='')

u_obs_test=simulate_ODEs(y_init,
                         params,
                        t_span,
                        obs_num)
np.savetxt('data/data_test.csv', u_obs, delimiter=',', header='t,u0,u1', comments='')