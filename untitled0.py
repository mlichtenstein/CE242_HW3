# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:41:10 2017

@author: Max
"""
#%%
import numpy as np
import timeit

x=np.linspace(-10,10,30)
timeit.timeit('
sig_ = [1/(1+np.exp(-x_i)) for x_i in x]
timeit.timeit('[1/(1+np.exp(-x_i)) for x_i in range(0,100)]',number =10000)
abs_ = [x_i/(1+abs(x_i))/2 + .5 for x_i in x]
tanh_ = [np.tanh(x_i)/2 + .5 for x_i in x]

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x,sig_,'r')
plt.plot(x,abs_,'b')
plt.plot(x,tanh_,'g')

