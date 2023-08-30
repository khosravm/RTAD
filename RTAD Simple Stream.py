#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:15:52 2023
Simulate streaming data with everlasting while True!
@author: khosravm
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

# Simulate streaming data
def stream_data():
    while True:
        yield np.random.normal(0, 1)

# Initialize the model
model = IsolationForest(contamination=0.05)

# Initialize a scaler
scaler = StandardScaler()

# Initialize thresholds
upper_threshold = 0.2
lower_threshold = -0.2

# Training phase
training_data = [next(stream_data()) for _ in range(1000)]
training_data = np.array(training_data).reshape(-1, 1)
scaler.fit(training_data)
scaled_training_data = scaler.transform(training_data)
model.fit(scaled_training_data)

# Real-time anomaly detection
for new_data_point in stream_data():
    scaled_data = scaler.transform([[new_data_point]])
    anomaly_score = model.score_samples(scaled_data)[0]
    
    if anomaly_score > upper_threshold or anomaly_score < lower_threshold:
        print(f"Anomaly detected: {new_data_point}, Anomaly Score: {anomaly_score}")
    
    time.sleep(1)  # Simulate real-time streaming delay
