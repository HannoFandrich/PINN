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

df = pd.read_csv('results/train_history.csv',sep=',')

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(df['epochs'], df['IC_loss'])
ax[1].plot(df['epochs'], df['ODE_loss'])
ax[2].plot(df['epochs'], df['Data_loss'])
ax[0].set_title('IC Loss', fontsize=14)
ax[1].set_title('ODE Loss', fontsize=14)
ax[2].set_title('Data Loss', fontsize=14)
for axs in ax:
    axs.set_yscale('log')
    axs.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
fig.savefig('plots/loss_history.png',dpi=200)



df_params=pd.read_csv('results/parameter_history.csv',sep=',')

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(df_params['epochs'], df_params['parameter1'], lw=3)
ax[0].set_ylabel('a', fontsize=16)
ax[0].set_xlabel('Iterations', fontsize=16)
ax[0].set_ylim((-2.2, -0.7))
ax[0].axhline(y=-2, color='r', linestyle='--')

ax[1].plot(df_params['epochs'], df_params['parameter2'], lw=3)
ax[1].set_ylabel('b', fontsize=16)
ax[1].set_xlabel('Iterations', fontsize=16)
ax[1].set_ylim((-0.2, 1.2))
ax[1].axhline(y=0, color='r', linestyle='--')

for axs in ax:
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.grid(True)
fig.suptitle('Parameter Evolution', fontsize=16)
fig.savefig('plots/parameter_history.png',dpi=200)


df_test=pd.read_csv('data/data_test.csv',sep=',')
df_data=pd.read_csv('results/model_data.csv',sep=',')

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].scatter(df_test['t'], df_test['u1'], label='truth')
ax[0].scatter(df_data['t'], df_data['u1'], label='predict')
ax[0].set_title('u1')
ax[1].scatter(df_test['t'], df_test['u2'], label='truth')
ax[1].scatter(df_data['t'], df_data['u2'], label='predict')
ax[1].set_title('u2')
ax[2].scatter(df_test['t'], df_test['u3'], label='truth')
ax[2].scatter(df_data['t'], df_data['u3'], label='predict')
ax[2].set_title('u3')

for axs in ax:
    axs.set_xlabel('Time (t)', fontsize=14)
    axs.set_ylabel('u values', fontsize=14)
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.legend(fontsize=12, frameon=True)
    axs.grid(True)

plt.tight_layout()
fig.savefig('plots/rates.png',dpi=200)