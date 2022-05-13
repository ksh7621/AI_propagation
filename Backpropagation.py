# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:08:20 2022

@author: ksh76
"""

#actual 0.6/0.8/0.5

import numpy as np

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def trunc(values, desc):
    return np.trunc(values*10**desc)/(10**desc) 
    
def forward(w_input_hidden, w_hidden_output, input):
    x_hidden = np.dot(w_input_hidden, input)
    o_hidden = sigmoid(x_hidden)
    x_output = np.dot(w_hidden_output,o_hidden)
    output = sigmoid(x_output)
    # output = trunc(output, 3)
    return o_hidden, output

def error_square(target,output):
    error1 = 0
    error1 = np.sum(0.5 *(target-output)**2)
    error2 = target-output
    return error1, error2

def backward(w_input_hidden,w_hidden_output, o_hidden, output, error1, error2):    
    lr = 0.3 
    
    #hidden-output weight   
    d_error_w_hidden_output = -error2 * output * (1-output)  * o_hidden   

    #input-hidden weight
    e1 = -error2 * output * (1-output) * w_hidden_output.T
    e2 = e1.sum()    
    d_error_w_input_hidden = e2 * o_hidden * o_hidden * (1- o_hidden)
        
    error_w_hidden_output = w_hidden_output.T - (lr * d_error_w_hidden_output)
    error_w_input_hidden = w_input_hidden.T - (lr * d_error_w_input_hidden)  
    
    return error_w_hidden_output, error_w_input_hidden


if __name__ == "__main__":
    iter = 1000
    input = np.array([0.9,0.1,0.8])
    target = np.array([0.6,0.8,0.5])

    w_input_hidden = np.array([[0.9,0.3,0.4],[0.2,0.8,0.2],[0.1,0.5,0.6]]) 
    w_hidden_output = np.array([[0.3,0.7,0.5],[0.6,0.5,0.2],[0.8,0.1,0.9]])
     
    
    for t in range(iter):    
        o_hidden, output = forward(w_input_hidden, w_hidden_output, input)         
        error1, error2 = error_square(target, output)
        error_w_hidden_output, error_w_input_hidden = backward(w_input_hidden,w_hidden_output,\
                                                                o_hidden, output, error1, error2)
        w_input_hidden = error_w_input_hidden
        w_hidden_output = error_w_hidden_output   
          
    print("after Training output \n", output)
        
  