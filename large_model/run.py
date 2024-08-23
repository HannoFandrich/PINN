import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.integrate import solve_ivp
from collections import defaultdict


u_init = np.array([1.0, 0.5, 0.3, 0.7, 1.2, 0.6, 0.4, 0.8, 0.9, 1.1, 0.2])
params = np.array([1, 0.1, 0.1, 1.0, 1.0, 0., 10, 3, 3, 0.1, 2, 0.2, 0.3 ]) 
t_span= [0,10]
obs_num=100000

def simulate_ODEs(u_init,params, t_span, obs_num):
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
    def model_large_system(t, u, p):
        # ODE system
        du = np.zeros(11)

        du[0] = -p[0] * u[0] + p[1] * u[1] - p[2] * u[0] * u[2] #
        du[1] = p[2] * u[0] * u[2] - p[3] * u[1] + p[4] * u[3] #
        du[2] = p[5] * u[2] - p[6] * u[4] + p[7] * u[1]
        du[3] = p[8] * u[2] - p[9] * u[3]
        du[4] = -p[0] * u[4] + p[1] * u[5] - p[2] * u[4] * u[6] # 4
        du[5] = p[2] * u[4] * u[6] - p[3] * u[5] + p[4] * u[9] # 5
        du[6] = p[8] * u[5] - p[9] * u[6] # 6
        du[7] = p[10] * u[7] - p[11] * u[8] + p[12] * u[9] # 7
        du[8] = -p[0] * u[8] + p[1] * u[9] - p[2] * u[8] * u[10] # 8
        du[9] = p[2] * u[8] * u[10] - p[3] * u[10] + p[4] * u[0] # 9
        du[10] = p[5] * u[9] - p[6] * u[1] + p[7] * u[8] # 10

        return du

    # Solve ODEs
    t_eval = np.linspace(t_span[0], t_span[1], obs_num)
    sol = solve_ivp(model_large_system, t_span, u_init, t_eval = t_eval, method='RK45',
                    args=(params,))
    # Restrcture obtained data
    u_obs = np.column_stack((sol.t, sol.y[0], sol.y[1], sol.y[2],sol.y[3], sol.y[4], sol.y[5],sol.y[6], sol.y[7], sol.y[8],sol.y[9], sol.y[10]))

    return u_obs


u_obs=simulate_ODEs(u_init,
                    params,
                    t_span,
                    obs_num)
np.savetxt('data/data.csv', u_obs, delimiter=',', header='t,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11', comments='')

u_obs_test=simulate_ODEs(u_init,
                        params,
                        t_span,
                        obs_num)
np.savetxt('data/data_test.csv', u_obs, delimiter=',', header='t,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11', comments='')