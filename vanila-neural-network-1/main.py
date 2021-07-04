#!/usr/bin/env python

import numpy as np
import csv

from numpy.core.fromnumeric import transpose

from loss import *
from propagation import *

NN_ARCHITECTURE = [
    {"input_dimension": 784, "output_dimension": 500, "activation": "relu"},
    {"input_dimension": 500, "output_dimension": 500, "activation": "relu"},
    {"input_dimension": 500, "output_dimension": 300, "activation": "relu"},
    {"input_dimension": 300, "output_dimension": 100, "activation": "relu"},
    {"input_dimension": 100, "output_dimension": 10, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dimension"]
        layer_output_size = layer["output_dimension"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 10 == 0):
            if(verbose):
                print("Iteration: {:05} - accuracy: {:.5f}".format(i, accuracy))
            if(callback is not None):
                callback(i, params_values)
            
    return params_values

if __name__ == '__main__':
    trainX = []
    trainY = []

    with open('../data/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                x = row[1:]
                y = [0] * 10
                y[int(row[0])] = 1

                trainX.append(x)
                trainY.append(y)

                line_count += 1

    trainX = np.array(trainX, dtype=float) / 255
    trainY = np.array(trainY, dtype=float)

    testX = []

    with open('../data/test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                x = row
                testX.append(x)

                line_count += 1

    testX = np.array(testX, dtype=float) / 255

    # Training
    params_values = train(np.transpose(trainX), np.transpose(trainY.reshape((trainY.shape[0], 10))), NN_ARCHITECTURE, 1000, 0.01, verbose=True)

    # Test
    Y_test_hat, _ = full_forward_propagation(np.transpose(testX), params_values, NN_ARCHITECTURE)

    Y_test_hat = Y_test_hat.T

    for y_test_hat in Y_test_hat.tolist():
        result = y_test_hat.index(max(y_test_hat))
        print()
