#predict, get_accuracy, and plot_graph functions are heavily borrowed from Abdou Rockicz

#import import_ipynb I use Jupyter, so this step is necessary for me
import load_data
from load_data import *

import random
import time
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#set seed to keep the results the same throughout various runs
random.seed(420)
np.random.seed(420)
tf.random.set_random_seed(420)
date = time.strftime("%Y-%m-%d")

ticker = "TSLA"                #change ticker here to change to company of interest

def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_seq"][:100]
    # retrieve the column scalers
    column_scaler = data["scaled_columns"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price

def get_accuracy(model, data):
    y_test = data["y_test"]
    x_test = data["x_test"]
    y_pred = model.predict(x_test)
    y_test = np.squeeze(data["scaled_columns"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["scaled_columns"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-1], y_pred[1:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-1], y_test[1:]))
    return accuracy_score(y_test, y_pred)

def plot_graph(model, data):
    y_test = data["y_test"]
    x_test = data["x_test"]
    y_pred = model.predict(x_test)
    y_test = np.squeeze(data["scaled_columns"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["scaled_columns"]["adjclose"].inverse_transform(y_pred))
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
    plt.savefig(f'{date}_{ticker}.png')



def create_model(length, units=256):
    model = Sequential()
    n_layers = 3
    for i in range(n_layers):
        if i == 0: # first layer
            model.add(LSTM(units, return_sequences=True, input_shape=(None, length)))
        elif i == n_layers - 1: # last layer
            model.add(LSTM(units, return_sequences=False))
        else:  # in the case of hidden layers
            model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(.4))   # add a 45% dropout after each layer
   
    model.add(Dense(1, activation="linear"))
    model.compile(loss="huber_loss", metrics=["mean_absolute_error"], optimizer= "adam")
    return model

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

data = load_data(ticker)

data["df"].to_csv(os.path.join("data", f"{ticker}_{date}.csv"))
model_name = f"{date}_{ticker}"
model = create_model(100)

# Tensorflow callbacks
checkpoint = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["x_train"], data["y_train"],
                    batch_size=64,
                    epochs=400,
                    validation_data=(data["x_test"], data["y_test"]),
                    callbacks=[checkpoint, tensorboard],
                    verbose=1)

model.save(os.path.join("results", model_name) + ".h5")
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
mse, mae = model.evaluate(data["x_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["scaled_columns"]["adjclose"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)
# predict the future price
future_price = predict(model, data)
print(f"Future price after {1} days is ${future_price:.2f}")
print("Accuracy Score:", get_accuracy(model, data))
plot_graph(model, data)
