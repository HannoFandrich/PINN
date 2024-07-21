import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.integrate import solve_ivp
from collections import defaultdict

df = pd.read_csv('results/train_history.csv',sep=',')

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].plot(df['epoch'], df['IC_loss'])
ax[1].plot(df['epoch'], df['ODE_loss'])
ax[2].plot(df['epoch'], df['Data_loss'])
ax[0].set_title('IC Loss', fontsize=14)
ax[1].set_title('ODE Loss', fontsize=14)
ax[2].set_title('Data Loss', fontsize=14)
ax[0].set_xlabel('training epoch', fontsize=14)
ax[1].set_xlabel('training epoch', fontsize=14)
ax[2].set_xlabel('training epoch', fontsize=14)
for axs in ax:
    axs.set_yscale('log')
    axs.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
fig.savefig('plots/loss_history.png',dpi=200)



df_params=pd.read_csv('results/parameter_history.csv',sep=',')
n_params=df_params.shape[1]-1

fig, ax = plt.subplots(1,n_params, figsize=(4*n_params, 4))
for i in range(n_params):
    ax[i].plot(df_params['epoch'], df_params['parameter'+str(i+1)], lw=3)
    ax[i].set_ylabel('parameter'+str(i+1), fontsize=14)
    ax[i].set_xlabel('training epoch', fontsize=14)
    #ax[0].set_ylim((-2.2, -0.7))
    #ax[0].axhline(y=-2, color='r', linestyle='--')

for axs in ax:
    axs.tick_params(axis='both', which='major', labelsize=12)
    #axs.grid(True)
fig.suptitle('Parameter Evolution', fontsize=16)
fig.savefig('plots/parameter_history.png',dpi=80*n_params)


df_test=pd.read_csv('data/data_test.csv',sep=',')
df_data=pd.read_csv('results/model_data.csv',sep=',')
n_rates=df_data.shape[1]-1

fig, ax = plt.subplots(1,n_rates, figsize=(4*n_rates, 4))
for i in range(n_rates):
    ax[i].scatter(df_test['t'], df_test['u'+str(i)], label='truth')
    ax[i].scatter(df_data['t'], df_data['u'+str(i+1)], label='predict')
    ax[i].set_title('u'+str(i+1))
for axs in ax:
    axs.set_xlabel('Time (t)', fontsize=14)
    axs.set_ylabel('u values', fontsize=14)
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.legend(fontsize=12, frameon=True)
    #axs.grid(True)

plt.tight_layout()
fig.savefig('plots/rates.png',dpi=80*n_rates)