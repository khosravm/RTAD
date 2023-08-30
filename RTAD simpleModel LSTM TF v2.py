#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:10:25 2023
Real-time time series anomaly detection using TensorFlow

This code incorporates the steps of data normalization, using a custom loss 
function, and using an adaptive threshold based on validation data. 

@author: khosravm
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic time series data
np.random.seed(0)
n_samples = 1000
time_steps = np.arange(0, n_samples)
normal_values = np.sin(time_steps * 0.02) + np.random.normal(0, 0.1, n_samples)
anomaly_values = np.sin(time_steps * 0.1) + np.random.normal(0, 0.1, n_samples)
data = np.concatenate([normal_values, anomaly_values])

# Create sliding windows for training
window_size = 1
X_train = [data[i:i+window_size] for i in range(len(data) - window_size)]
X_train = np.array(X_train)

# Normalize the data
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std

# Build and compile the LSTM model
model = keras.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.LSTM(64, return_sequences=True, activation='relu'),
    layers.LSTM(64, return_sequences=True, activation='relu'),
    layers.Dense(1)
])

# Use custom loss function
def anomaly_loss(y_true, y_pred):
    reconstruction_error = tf.abs(y_true - y_pred)
    anomaly_weight = 5.0  # Adjust anomaly weight as needed
    return tf.reduce_mean(tf.where(y_true >= 0, reconstruction_error, anomaly_weight * reconstruction_error))

model.compile(optimizer='adam', loss=anomaly_loss)

# Train the model
X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],1 ,1)) 
model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1)

# Detect anomalies using prediction errors
predictions = model.predict(X_train)
errors = np.mean(np.abs(predictions - X_train), axis=1)

# Use adaptive thresholding based on validation data
threshold = np.percentile(errors, 95)  # Adjust percentile as needed

# Identify anomalies
anomalies = np.where(errors > threshold)[0]

print("Detected anomalies at:", anomalies)
