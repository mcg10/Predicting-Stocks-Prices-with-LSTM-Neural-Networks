#import import_ipynb I use Jupyter, so I need to use this to import local Python files
import load_data
from load_data import *
import random
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

random.seed(420)
np.random.seed(420)
tf.random.set_random_seed(420)
date = time.strftime("%Y-%m-%d")

def create_model(length, units=256):
    model = Sequential()
    n_layers = 2
    for i in range(n_layers):
        if i == 0: # first layer
            model.add(LSTM(units, return_sequences=True, input_shape=(None, length)))
        elif i == n_layers - 1: # last layer
            model.add(LSTM(units, return_sequences=False))
        else:  # in the case of hidden layers
            model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(.45))   # add a 45% dropout after each layer
   
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mean_absolute_error", metrics=["mean_absolute_error"], optimizer= "rmsprop")
    return model

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

ticker = "TSLA"                #change ticker here to change to company of interest
data = load_data(ticker)

data["df"].to_csv(os.path.join("data", f"{ticker}_{date}.csv"))
model_name = f"{date}_{ticker}"
model = create_model(100)

# some tensorflow callbacks
checkpoint = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["x_train"], data["y_train"],
                    batch_size=64,
                    epochs=400,
                    validation_data=(data["x_test"], data["y_test"]),
                    callbacks=[checkpoint, tensorboard],
                    verbose=1)

model.save(os.path.join("results", model_name) + ".h5")
