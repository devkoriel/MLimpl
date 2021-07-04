#!/usr/bin/env python

import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def softmax_backward(Y_hat, Y):
    return Y_hat - Y
