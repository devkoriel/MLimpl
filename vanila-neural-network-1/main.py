#!/usr/bin/env python

import numpy as np
import csv

from numpy.core.fromnumeric import transpose

from loss import *
from propagation import *

nn_architecture = [
    {"input_dimension": 784, "output_dimension": 100, "activation": "relu"},
    {"input_dimension": 100, "output_dimension": 100, "activation": "relu"},
    {"input_dimension": 100, "output_dimension": 100, "activation": "relu"},
    {"input_dimension": 100, "output_dimension": 20, "activation": "relu"},
    {"input_dimension": 20, "output_dimension": 10, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dimension"]
        layer_output_size = layer["output_dimension"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        l =[]
        for j in range(len(X)):
            Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            accuracy = get_accuracy_value(Y_hat, Y)
            
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate)

            # TODO: cost / accracy sum/avg 계산 후 추가
       
        cost_history.append(cost)
        accuracy_history.append(accuracy)
        
    return params_values, cost_history, accuracy_history

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
        print(f'Processed {line_count} lines.')

    trainX = np.array(trainX, dtype=int)
    trainY = np.array(trainY, dtype=int)

    print(trainY[0].T)

    train(trainX[0], trainY[0].T, nn_architecture, 20, 0.01)
