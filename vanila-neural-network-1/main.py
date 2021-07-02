#!/usr/bin/env python

import numpy as np

import activation
import propagation

nn_architecture = [
    {"input_dimension": 783, "output_dimension": 100, "activation": "relu"},
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

if __name__ == '__main__':
    params_values = init_layers(nn_architecture)
    print(params_values)
