#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:19:40 2023
Real-time time series anomaly detection

It  is a simplified example of real-time time series anomaly detection using 
TensorFlow. This example demonstrates how to set up a simple LSTM-based model 
to detect anomalies in a time series sequence. 

An LSTM-based autoencoder is used to learn the normal patterns of the time 
series data. Anomalies are then detected by comparing the reconstruction errors 
between the original input and the reconstructed output. 

@author: khosravm
"""
import numpy as np
# import tensorflow as tf
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

# Build and compile the LSTM model
model = keras.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(64, return_sequences=True),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae')

# Train the model

X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],1 ,1))  #np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
model.fit(X_train, X_train, epochs=10, batch_size=32, verbose=1)

# Detect anomalies using prediction errors
predictions = model.predict(X_train)
errors = np.mean(np.abs(predictions - X_train), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(errors, 95)

# Identify anomalies
anomalies = np.where(errors > threshold)[0]

print("Detected anomalies at:", anomalies)

