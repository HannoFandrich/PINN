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
u_init=[1,1]
n_u=len(u_init)
f_init=[1,1]
n_f=len(f_init)
parameters='a, b, c, d'
parameters_init=[1,1,1,1]
n_parameters=len(parameters_init)

### ODE Residuals
### dependend on ODE system!

def ODE_residual(du_dt,f,u,parameters):
    du1_dt = du_dt[:, :1] 
    du2_dt = du_dt[:, 1:]
    f1=f[:, :1]
    f2=f[:, 1:]
    u1=u[:, :1]
    u2=u[:, 1:]
    p=parameters

    res1 = du1_dt - p[0]-u1-(p[1]*u1)*(1+p[2]*u2**4)
    res2 = du2_dt - (p[1]*u1)*(1+p[2]*u2**4)-p[3]*u2
    ODE_residual = tf.concat([res1, res2], axis=1)
    return ODE_residual





### CONFIG
tf.random.set_seed(42)
n_epochs = 1000
IC_weight= tf.constant(1.0, dtype=tf.float32)
ODE_weight= tf.constant(1.0, dtype=tf.float32)
data_weight= tf.constant(1.0, dtype=tf.float32)


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
        return inputs * self.parameters

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

def f_net(input_layers, n_parameters, parameters_init=None):
    """Definition of the network for f prediction."""

    hidden = tf.keras.layers.Concatenate()(input_layers)
    for _ in range(2):
        hidden = tf.keras.layers.Dense(50, activation="tanh")(hidden)
    output = tf.keras.layers.Dense(n_parameters)(hidden)
    output = ParameterLayer(parameters_init)(output)
    return output

def create_PINN(n_rates, n_parameters, parameters_init=None, verbose=False):
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
    f = f_net([t_input, u], n_parameters, parameters_init)

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

n_epochs = 1000
IC_weight= tf.constant(1.0, dtype=tf.float32)
ODE_weight= tf.constant(1.0, dtype=tf.float32)
data_weight= tf.constant(1.0, dtype=tf.float32)
loss_tracker = LossTracking()
val_loss_hist = []
params_list = []

# Set up optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.002)

with tf.device("CPU:0"):

    # Instantiate the PINN model
    PINN = create_PINN(n_rates=n_u,n_parameters=n_parameters,parameters_init=parameters_init)
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
for i in range(n_f):
    df['f'+str(i+1)]=np.asarray(f[:,i])
df.to_csv('results/model_data.csv', index=False)

np.savetxt('results/parameters.csv', params_list[-1], delimiter=',', header=parameters, comments='')