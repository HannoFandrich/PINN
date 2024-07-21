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


### Initials
u_init = np.array([1.0, 0.5, 0.3, 0.7, 1.2, 0.6, 0.4, 0.8, 0.9, 1.1, 0.2])
n_u=len(u_init)
f_init=np.ones(5)
n_f=len(f_init)
#parameters='p0,p1,p2,p3,p4'
#parameters_init=np.ones(5)
parameters='p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13'
parameters_init=np.ones(13)
n_parameters=len(parameters_init)

### ODE Residuals
### dependend on ODE system!

def ODE_residual(du_dt,f,u,parameters):
    p=parameters

    res0=du_dt[0] - (-p[0] * u[0] + p[1] * u[1] - p[2] * u[0] * u[2])  #
    res1=du_dt[1] - (p[2] * u[0] * u[2] - p[3] * u[1] + p[4] * u[3])  #
    res2=du_dt[2] -(p[5] * u[2] - p[6] * u[4] + p[7] * u[1]) ##
    res3=du_dt[3] -(p[8] * u[2] - p[9] * u[3]) ##
    res4=du_dt[4] - (-p[0] * u[4] + p[1] * u[5] - p[2] * u[4] * u[6])  # 4
    res5=du_dt[5] - (p[2] * u[4] * u[6] - p[3] * u[5] + p[4] * u[7])  # 5
    res6=du_dt[6] -(p[8] * u[5] - p[9] * u[6]) ## 6
    res7=du_dt[7] -(p[10] * u[7] - p[11] * u[8] + p[12] * u[9]) ## 7
    res8=du_dt[8] - (-p[0] * u[8] + p[1] * u[9] - p[2] * u[8] * u[10])  # 8
    res9=du_dt[9] - (p[2] * u[8] * u[10] - p[3] * u[9] + p[4] * u[0])  # 9
    res10 = du_dt[10] -(p[5] * u[9] - p[6] * u[1] + p[7] * u[8])  ## 10

    '''
    res0=du_dt[0] -f[0] ## 0
    res1=du_dt[1] -f[1] ## 1
    res2=du_dt[2] -f[2] ## 2
    res3=du_dt[3] -f[3] ## 3
    res4 = du_dt[4] -f[4]  ## 4
    res5=du_dt[5] - (-p[0] * u[5] + p[1] * u[6] - p[2] * u[5] * u[0])  # 5
    res6=du_dt[6] - (p[2] * u[5] * u[0] - p[3] * u[6] + p[4] * u[1])  # 6
    res7=du_dt[7] - (-p[0] * u[7] + p[1] * u[8] - p[2] * u[7] * u[2])  # 7
    res8=du_dt[8] - (p[2] * u[7] * u[2] - p[3] * u[8] + p[4] * u[3])  # 8
    res9=du_dt[9] - (-p[0] * u[9] + p[1] * u[10] - p[2] * u[9] * u[4])  # 9
    res10=du_dt[10] - (p[2] * u[9] * u[4] - p[3] * u[10] + p[4] * u[5])  # 10
    '''


    '''
    res0 = du_dt[0] - (-p[0] * u[0] + p[1] * u[1] - p[2] * u[0] * u[2])  #
    res1 = du_dt[1] - (p[2] * u[0] * u[2] - p[3] * u[1] + p[4] * u[3])  #
    res2 = du_dt[2] - (p[5] * u[2] - p[6] * u[4] + p[7] * u[1])
    res3 = du_dt[3] - (p[8] * u[2] - p[9] * u[3])
    res4 = du_dt[4] - (p[10] * u[4] - p[11] * u[5] + p[12] * u[6])
    res5 = du_dt[5] - (p[13] * u[5] - p[14] * u[7] + p[15] * u[8])
    res6 = du_dt[6] - (p[16] * u[6] - p[17] * u[9])
    res7 = du_dt[7] - (p[18] * u[7] - p[19] * u[10])
    res8 = du_dt[8] - (-p[0] * u[8] + p[1] * u[9] - p[2] * u[8] * u[11])  #
    res9 = du_dt[9] - (p[2] * u[8] * u[11] - p[3] * u[9] + p[4] * u[12])  #
    res10 = du_dt[10] - (p[5] * u[10] - p[6] * u[13] + p[7] * u[9])
    res11 = du_dt[11] - (p[8] * u[10] - p[9] * u[11])
    res12 = du_dt[12] - (p[10] * u[12] - p[11] * u[13] + p[12] * u[14])
    res13 = du_dt[13] - (p[13] * u[13] - p[14] * u[15] + p[15] * u[16])
    res14 = du_dt[14] - (p[16] * u[14] - p[17] * u[17])
    res15 = du_dt[15] - (p[18] * u[15] - p[19] * u[18])
    res16 = du_dt[16] - (-p[0] * u[16] + p[1] * u[17] - p[2] * u[16] * u[19])  #
    res17 = du_dt[17] - (p[2] * u[16] * u[19] - p[3] * u[17] + p[4] * u[0])  #
    res18 = du_dt[18] - (p[5] * u[18] - p[6] * u[1] + p[7] * u[17])
    res19 = du_dt[19] - (p[8] * u[18] - p[9] * u[19])
    '''
    ODE_residual = tf.concat([res0,res1, res2, res3,res4,res5,res6,res7,res8,res9,res10], axis=1)
    return ODE_residual





### CONFIG
tf.random.set_seed(42)
n_epochs = 3000
IC_weight= tf.constant(1, dtype=tf.float32)
ODE_weight= tf.constant(0.7, dtype=tf.float32)
data_weight= tf.constant(5, dtype=tf.float32)


### Data Import
data = np.genfromtxt('data/data.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('data/data_test.csv', delimiter=',', skip_header=1)

### Organise Data
# Set batch size
data_batch_size = 100
ODE_batch_size = 1000

# Samples for enforcing data loss
X_train_data = tf.convert_to_tensor(data[:, :1], dtype=tf.float32)
y_train_data = tf.convert_to_tensor(data[:, 1:], dtype=tf.float32)
train_ds_data = tf.data.Dataset.from_tensor_slices((X_train_data, y_train_data))
train_ds_data = train_ds_data.shuffle(1000).batch(data_batch_size)

# Samples for enforcing ODE residual loss
N_collocation = 10000
X_train_ODE = tf.convert_to_tensor(np.linspace(0, 10, N_collocation).reshape(-1, 1), dtype=tf.float32)
train_ds_ODE = tf.data.Dataset.from_tensor_slices((X_train_ODE))
train_ds_ODE = train_ds_ODE.shuffle(10*N_collocation).batch(ODE_batch_size)

# Generate testing data
X_test, y_test = test_data[:, :1], test_data[:, 1:]








class ParameterLayer(tf.keras.layers.Layer):
    '''
    adding extra layer for parameter output

    code from notebook, modified by chat gpt
    '''
    def __init__(self, parameters, trainable=True):
        super(ParameterLayer, self).__init__()
        self._parameters = tf.convert_to_tensor(parameters, dtype=tf.float32)
        self.trainable = trainable

    def build(self, input_shape):
        # Create a weight for each parameter in the input list
        self.parameters = self.add_weight(
            name="parameters",
            shape=self._parameters.shape,
            initializer=tf.keras.initializers.Constant(value=self._parameters),
            trainable=self.trainable
        )
    def call(self, inputs):
        # Define the forward pass here
        return inputs #* self.parameters 

    def get_config(self):
        config = super().get_config()
        config.update({
            "parameters": self._parameters.numpy(),  # Convert to numpy array for serialization
            "trainable": self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        parameters = config.pop("parameters")
        return cls(parameters, **config)
    

def u_net(input_layer, n_rates):
    """Definition of the network for u prediction."""

    hidden = input_layer
    for _ in range(2):
        hidden = tf.keras.layers.Dense(50, activation="tanh")(hidden)
    output = tf.keras.layers.Dense(n_rates)(hidden)
    return output

def f_net(input_layers, n_ODEs, parameters_init=None):
    """Definition of the network for f prediction."""

    hidden = tf.keras.layers.Concatenate()(input_layers)
    for _ in range(2):
        hidden = tf.keras.layers.Dense(50, activation="tanh")(hidden)
    output = tf.keras.layers.Dense(n_ODEs)(hidden)
    output = ParameterLayer(parameters_init)(output)
    return output

def create_PINN(n_rates, n_ODEs, parameters_init=None, verbose=False):
    """Definition of a physics-informed neural network.

    Args:
    ----
    a_init: initial value for parameter a
    b_init: initial value for parameter b
    verbose: boolean, indicate whether to show the model summary

    Outputs:
    --------
    model: the PINN model
    """
    # Input
    t_input = tf.keras.Input(shape=(1,), name="time")

    # u-NN
    u = u_net(t_input, n_rates)

    # f-NN
    f = f_net([t_input, u], n_ODEs, parameters_init)

    # PINN model
    model = tf.keras.models.Model(inputs=t_input, outputs=[u, f])

    if verbose:
        model.summary()

    return model


@tf.function
def ODE_residual_calculator(t, model,n_u):
    """ODE residual calculation.

    Args:
    ----
    t: temporal coordinate
    model: PINN model

    Outputs:
    --------
    ODE_residual: residual of the governing ODE
    """

    # Retrieve parameters
    parameters=model.layers[-1].parameters

    with tf.GradientTape() as tape:
        tape.watch(t)
        u, f = model(t)

    # Calculate gradients
    du_dt = tape.batch_jacobian(u, t)[:, :, 0]
    du_dt=[du_dt[:,a:a+1] for a in range(len(du_dt[0]))]
    f=[f[:,a:a+1] for a in range(len(f[0]))]
    u=[u[:,a:a+1] for a in range(len(u[0]))]

    # Compute residuals
    res_arr=ODE_residual(du_dt,f,u,parameters)
    '''
    res1 = du1_dt - f[:, :1]
    res2 = du2_dt - f[:, 1:]
    res3 = du3_dt - (a*u[:, :1]*u[:, 1:2] + b)
    ODE_residual = tf.concat([res1, res2, res3], axis=1)
    '''
    print(f)
    res_arr = tf.convert_to_tensor(res_arr, dtype=tf.float32)
    return res_arr

@tf.function
def train_step(X_ODE, X, y,u_init,f_init,parameters_init,n_u,n_f,n_parameters, IC_weight, ODE_weight, data_weight, model):
    """Calculate gradients of the total loss with respect to network model parameters.

    Args:
    ----
    X_ODE: Collocation points for evaluating ODE residuals
    X: observed samples
    y: target values of the observed samples
    IC_weight: weight for initial condition loss
    ODE_weight: weight for ODE loss
    data_weight: weight for data loss
    model: PINN model

    Outputs:
    --------
    ODE_loss: calculated ODE loss
    IC_loss: calculated initial condition loss
    data_loss: calculated data loss
    total_loss: weighted sum of ODE loss, initial condition loss, and data loss
    gradients: gradients of the total loss with respect to network model parameters.
    """
    with tf.GradientTape() as tape:
        #tape.watch(model.trainable_weights)

        # Initial condition prediction
        y_pred_IC, _ = model(tf.zeros((1, 1)))

        # Equation residual
        ODE_res = ODE_residual_calculator(t=X_ODE, model=model,n_u=n_u)

        # Data loss
        y_pred_data, _ = model(X)

        # Calculate loss
        IC_loss = tf.reduce_mean(tf.keras.losses.MSE([u_init], y_pred_IC))
        ODE_loss = tf.reduce_mean(tf.square(ODE_res))
        data_loss = tf.reduce_mean(tf.keras.losses.MSE(y, y_pred_data))

        # Weight loss
        total_loss = IC_loss*IC_weight + ODE_loss*ODE_weight + data_loss*data_weight

    gradients = tape.gradient(total_loss, model.trainable_variables)

    return ODE_loss, IC_loss, data_loss, total_loss, gradients


class LossTracking:

    def __init__(self):
        self.mean_total_loss = keras.metrics.Mean()
        self.mean_IC_loss = keras.metrics.Mean()
        self.mean_ODE_loss = keras.metrics.Mean()
        self.mean_data_loss = keras.metrics.Mean()
        self.loss_history = defaultdict(list)

    def update(self, total_loss, IC_loss, ODE_loss, data_loss):
        self.mean_total_loss(total_loss)
        self.mean_IC_loss(IC_loss)
        self.mean_ODE_loss(ODE_loss)
        self.mean_data_loss(data_loss)

    def reset(self):
        self.mean_total_loss.reset_state()
        self.mean_IC_loss.reset_state()
        self.mean_ODE_loss.reset_state()
        self.mean_data_loss.reset_state()

    def print(self):
        print(f"IC={self.mean_IC_loss.result().numpy():.4e}, \
              ODE={self.mean_ODE_loss.result().numpy():.4e}, \
              data={self.mean_data_loss.result().numpy():.4e}, \
              total_loss={self.mean_total_loss.result().numpy():.4e}")

    def history(self):
        self.loss_history['total_loss'].append(self.mean_total_loss.result().numpy())
        self.loss_history['IC_loss'].append(self.mean_IC_loss.result().numpy())
        self.loss_history['ODE_loss'].append(self.mean_ODE_loss.result().numpy())
        self.loss_history['Data_loss'].append(self.mean_data_loss.result().numpy())

class PrintParameters(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nparameters: {self.model.layers[-1].parameters.numpy()}")

loss_tracker = LossTracking()
val_loss_hist = []
params_list = []

# Set up optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.002)

with tf.device("CPU:0"):

    # Instantiate the PINN model
    PINN = create_PINN(n_rates=n_u,n_ODEs=n_f,parameters_init=parameters_init)
    PINN.compile(optimizer=optimizer)

    # Configure callbacks
    _callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=100),
                 tf.keras.callbacks.ModelCheckpoint('PINN_model.keras', monitor='val_loss', save_best_only=True),
                 PrintParameters()]
    callbacks = tf.keras.callbacks.CallbackList(
                    _callbacks, add_history=False, model=PINN)

    # Start training process
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}:")

        for (X_ODE), (X, y) in zip(train_ds_ODE, train_ds_data):

            # Calculate gradients
            ODE_loss, IC_loss, data_loss, total_loss, gradients = train_step(X_ODE, X, y,
                                                                             u_init, f_init, parameters_init, n_u, n_f, n_parameters,
                                                                             IC_weight, ODE_weight, data_weight, PINN)
            # Gradient descent
            PINN.optimizer.apply_gradients(zip(gradients, PINN.trainable_variables))

            # Loss tracking
            loss_tracker.update(total_loss, IC_loss, ODE_loss, data_loss)

        # Loss summary
        loss_tracker.history()
        loss_tracker.print()
        loss_tracker.reset()

        # Parameter recording
        params_list.append(PINN.layers[-1].parameters.numpy())
        '''
        ####### Validation
        val_res = ODE_residual_calculator(tf.reshape(tf.linspace(0.0, 10.0, 1000), [-1, 1]), PINN,n_u=n_u)
        val_ODE = tf.cast(tf.reduce_mean(tf.square(val_res)), tf.float32)

        u_init=tf.constant([[1.0, 0.8, 0.5]])
        val_pred_init, _ = PINN.predict(tf.zeros((1, 1)))
        val_IC = tf.reduce_mean(tf.square(val_pred_init - u_init))
        #print(f"val_IC: {val_IC.numpy():.4e}, val_ODE: {val_ODE.numpy():.4e}, lr: {PINN.optimizer.lr.numpy():.2e}")
        print(f"val_IC: {val_IC.numpy():.4e}, val_ODE: {val_ODE.numpy():.4e}, lr: {PINN.optimizer.learning_rate.numpy():.2e}")

        
        # Callback at the end of epoch
        callbacks.on_epoch_end(epoch, logs={'val_loss': val_IC+val_ODE})
        val_loss_hist.append(val_IC+val_ODE)
        '''
        ####### Validation
        val_res = ODE_residual_calculator(X_train_data, PINN,n_u=n_u)
        val_ODE = tf.cast(tf.reduce_mean(tf.square(val_res)), tf.float32)

        val_pred_init, _ = PINN.predict(tf.zeros((1, 1)))
        val_IC = tf.reduce_mean(tf.square(val_pred_init - u_init))
        #print(f"val_IC: {val_IC.numpy():.4e}, val_ODE: {val_ODE.numpy():.4e}, lr: {PINN.optimizer.lr.numpy():.2e}")
        print(f"val_IC: {val_IC.numpy():.4e}, val_ODE: {val_ODE.numpy():.4e}, lr: {PINN.optimizer.learning_rate.numpy():.2e}")

        
        # Callback at the end of epoch
        callbacks.on_epoch_end(epoch, logs={'val_loss': tf.cast(val_IC, dtype=tf.float32)+tf.cast(val_ODE, dtype=tf.float32)})
        val_loss_hist.append(tf.cast(val_IC, dtype=tf.float32)+tf.cast(val_ODE, dtype=tf.float32))


        # Test dataset
        pred_test, _ = PINN.predict(X_test, batch_size=12800)
        print(f"RMSE: {mean_squared_error(y_test.flatten(), pred_test.flatten(), squared=False)}")


        # Re-shuffle dataset
        train_ds_data = tf.data.Dataset.from_tensor_slices((X_train_data, y_train_data))
        train_ds_data = train_ds_data.shuffle(10000).batch(data_batch_size)

        train_ds_ODE = tf.data.Dataset.from_tensor_slices((X_train_ODE))
        train_ds_ODE = train_ds_ODE.shuffle(10*N_collocation).batch(ODE_batch_size)

### training history
df=pd.DataFrame({'epoch': list(range(n_epochs))})
df['IC_loss']= loss_tracker.loss_history['IC_loss']
df['ODE_loss']= loss_tracker.loss_history['ODE_loss']
df['Data_loss']= loss_tracker.loss_history['Data_loss']
df.to_csv('results/train_history.csv', index=False)

### parameter history
df=pd.DataFrame({'epoch': list(range(n_epochs))})
for i in range(len(parameters_init)):
    df['parameter'+str(i+1)]=np.asarray(params_list)[:,i]
df.to_csv('results/parameter_history.csv', index=False)

### model data
t = X_train_data
u, f = PINN.predict(t, batch_size=12800)

df = pd.DataFrame({
    't': t.numpy().flatten()})
for i in range(n_u):
    df['u'+str(i+1)]=np.asarray(u[:,i])
'''
for i in range(n_f):
    df['f'+str(i+1)]=np.asarray(f[:,i])
'''
df.to_csv('results/model_data.csv', index=False)

np.savetxt('results/parameters.csv', params_list[-1], delimiter=',', header=parameters, comments='')