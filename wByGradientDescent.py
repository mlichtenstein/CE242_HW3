# this script does a basic gradient descent to find a simple classifier

#  This module contains tools for getting features from a data file


import dataToFeatures
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
matplotlib.interactive(True)
#%% ok let's make a sigmoid function:

def sigmoid(w,x):
    '''takes a feature vector and a weight vector and computes their sigmoid activation'''
    return 1/(1 + np.exp(- np.dot(w,x)))

#test it out!
if __name__=="__main__":
    x = np.linspace(-2,2,100)
    y = np.linspace(-2,2,100)
    results = np.zeros([100,100])
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            results[i,j] = sigmoid([x[i],y[j]],[ 1 , .5])
    plt.imshow(results, cmap='hot', interpolation='nearest')
    plt.title('output of example sigmoid function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    #excellent!
    
#%% Let's try doing a gradient descent!
def doGradientDescent(X,y, lamb = 0, nu = .1, alpha = 0, iters = 100, w_0 = None):
    '''does a gradient descent for a given feature list 
    Inputs:
        X,y = feature list
        lamb = regularizer weight
        nu = gradient descent speed
        alpha = gradient descent speed decay 
        iters = number of iterations
        w_0 = starting guess (default to origin)
    Outputs:  (w_f, err_f, w, err)
        w_f = final weight vector
        err_f = final error
        w = all w's
        err = all errors
    '''
    if w_0 == None:
        w_0 = np.zeros( [1, X.shape[1]] )
    # y is defined as true or false, but let's convert it to +/- 1
    #y = [(1 if y_i else -1) for y_i in y] # I think we shouldn't???
    
    def errorAtW(w_in):
        err = 0
        for i in range(0,len(y)):
            x_i =  X.getrow(i).toarray().tolist()[0]
            #w_in = w_in.tolist()[0]
            err += (y[i]*np.log(sigmoid(w_in,x_i)) 
                    + (1-y[i])*np.log(1-sigmoid(w_in,x_i)))
        return err
    
    #now iterate:
    w = [w_0]
    err = [errorAtW(w_0)]
    for t in range(0,iters):
        dGdw = 2*lamb*w
        w_t = w[-1].tolist()[0]
        for i in range(0,len(y)):
            x_i = X.getrow(i).toarray().tolist()[0]
            dGdw += -(y[i] - sigmoid(w_t,x_i))
        w_t += nu * t**(-alpha) * dGdw
        
        err.append(errorAtW(w_t))
        w.append(w_t)
    return
    
if __name__=="__main__":
    (vec, X, y) = dataToFeatures.getVectorizerFromHamSpamFile('train1.csv')
    
    doGradientDescent(X,y)