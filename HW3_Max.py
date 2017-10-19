
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# In[3509]:
#some basic configuration stuff:
Quick_Run_Flag = True;  #do a quick run for testing???
Quick_Run_n = 1000;          #on a quick run, how many samples?        
Feature_len = 1000; #desired feature length

# In[3510]:
# Import data:
# Read training data file:
import csv
f = open("train.csv", "r")
raw_data = list(csv.reader(f))
raw_data = raw_data[1:] #strip header line from raw data
f.close()

print('The first few lines of raw_data:\n', raw_data[0:3])

# Parse raw data into Target and String for each target
target_strings = []
X_strings = []
for row in raw_data:
    target_strings.append(row[0])
    X_strings.append(row[1])
    
# optionally, we can decrease the sample size for testing:
if Quick_Run_Flag:
    X_strings = X_strings[0:Quick_Run_n]
    target_strings = target_strings[0:Quick_Run_n]

# Convert target to numeric:
target = np.array(
        [(0 if t == "ham" else 1) for t in target_strings] )
print('\nThere are',len(target),'labels, of which',sum(target),'are spam')

# In[3516]:
# Tokenize and transform subject line strings from raw data:

vectorizer = TfidfVectorizer(
    ngram_range = (1,2),
#    max_features= 700, 
    stop_words = "english")
X_full  = vectorizer.fit_transform(X_strings).toarray()

print('X has shape',X_full.shape)

#%%
#let's do some dimensionality reduction
#we also do an SVD transform to do some dimesionality reduction:
import matplotlib.pyplot as plt

(u,s,v) = np.linalg.svd(X_full,    full_matrices= False)

#transform X into an information-rich space by taking u*s, 
#and truncate s to cut out the least useful dimensions
#(we'll have to do something similar to X_test,
#  taking X_test = (v.T.dot(X_test)[0:Feature_len] )
X_unbiased = u[:,0:Feature_len].dot(np.diag(s[0:Feature_len]))
X = np.append(np.ones([len(X_unbiased),1]), X_unbiased, axis = 1)

plt.plot(s)
plt.plot([Feature_len, Feature_len],[0, max(s)] )
plt.legend('singular values of X', 'feature cutoff')
plt.show()

print('reduced X has shape',X.shape)

#%%
# Define and test our sigmoid function:
def sigmoid(w_,x):
    return float(1 / (1 + np.exp(-(np.transpose(w_).dot(x)))))

#%%
# let's test it:
w0_domain = np.linspace(-2,2,100)
w1_domain = np.linspace(2,-2,100)
x_test = np.array([[1, 0.5]]).transpose() # a column vector
results = np.zeros([100,100])
for i,w0 in enumerate(w0_domain):
    for j,w1 in enumerate(w1_domain):
        feature = np.array([[w0,w1]]).transpose() # a column vector
        results[j,i] = sigmoid(feature,x_test)
        
plt.figure()
plt.imshow(results, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
plt.arrow(0,0,x_test.item(0),x_test.item(1),
          color='b',head_width=.1)
plt.title('output of example sigmoid function')
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()


#%%
#establish loss fcn:

def loss(w_,x,target,lmbda):
    J =0
    for i,row in enumerate(x):
        y = sigmoid(w_,row)
        J += - (target[i]*np.log(y) + (1-target[i])*np.log(1-y))
    reg = lmbda * np.sum(w_[1:]**2)
    return ( (J) + reg)

w0_domain = np.linspace(-2,2,100)
w1_domain = np.linspace(2,-2,100)
target_test = [1,0]
x_test = np.array([[1, 1, 1],[1,-1,-1]]) # a ham in the upper right, a spam in the lower left
results_0 = np.zeros([100,100])
results_1 = np.zeros([100,100])
for i,w0 in enumerate(w0_domain):
    for j,w1 in enumerate(w1_domain):
        w_here = np.array([[3, w0,w1]]).transpose() # a column vector
        results_0[j,i] = loss(w_here,x_test,target_test,0)
        results_1[j,i] = loss(w_here,x_test,target_test,1)


f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('loss over w')
ax1.imshow(results_0, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax1.set_title('lambda = 0')
ax2.imshow(results_1, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax2.set_title('lambda = 1')

for ax in (ax1,ax2):
    ax.text(1,1,'+1', bbox={'boxstyle':'circle', 'facecolor':'white',  'pad':.5})
    ax.text(-1,-1,'0', bbox={'boxstyle':'circle', 'facecolor':'white', 'pad':.5})
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
f.show()

#%%
# We also need a gradient descent function, we first make a partial derivative function:

# the partial derivative is 
def partial_der(w_,X,t, lmbda):
    partials = []
    for j in range(len(w_)):
        J = 0
        for i,row in enumerate(X):
            y = sigmoid(w_,row)
            J += (y - t[i])*row[j]
        if not j == 0:
            reg = lmbda * w_[j]
        else:
            reg = np.array([0])
        partials.append(J + reg)
        
    return(np.array(partials))
    
    
w_test = np.array([[1, 1, 0]]).transpose() # a column vector

print(partial_der(w_test,x_test,target_test,0))

# test with a plot:
f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('partial derivatives wrt w over w')
ax1.imshow(results_0, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax1.set_title('lambda = 0')
ax2.imshow(results_1, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax2.set_title('lambda = 1')

w0_domain = np.linspace(-2,2,10)
w1_domain = np.linspace(2,-2,10)
for w0 in w0_domain:
    for w1 in w1_domain:
        w_here = np.array([[0, w0,w1]]).transpose() # a column vector
        
        pd = partial_der(w_here, x_test, target_test,0)/5
        ax1.arrow(w0,w1,pd.item(1),pd.item(2),
          color='white',head_width=.1)
        
        pd = partial_der(w_here, x_test, target_test,1)/5
        ax2.arrow(w0,w1,pd.item(1),pd.item(2),
          color='white',head_width=.1)

for ax in (ax1,ax2):
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
f.show()

#%%
# Now we write a gradient descent function:

import sys

def w_update(w_, partial, rate, iteration_no):
    w_ = w_ - rate*(iteration_no**-0.9)*partial
    return(w_)

def gradient_descent(X,target, lmbda, 
                     initial_w = np.array([]), rate = .1, 
                     max_step = 100, convergence = 0.01,printout = True):
    if len(initial_w) == 0:
        w_ = X[0,:]*0 #a trick to get a zero array with size of a row of X
    else:
        w_ = initial_w
    
    loss_current = loss(w_,X,target, lmbda)
    loss_prev = loss_current+10 #make sure loss doesn't converge on first iteration
    results_loss = [loss_current]
    results_w = w_.reshape([1,-1])
    counter = 1

    while (abs(loss_current - loss_prev) > convergence) and (counter < max_step):
        loss_prev = loss_current
        partial = partial_der(w_, X, target, lmbda)
        w_ = w_update(w_, partial, rate, counter)
        loss_current = loss(w_, X, target, lmbda)
        
        results_w = np.append(results_w,w_.reshape([1,-1]), axis=0)
        results_loss.append(loss_current)
        counter += 1
        
        if printout:
            sys.stdout.write("\rDescending along gradient... %f%%" % (counter*100/max_step))
            sys.stdout.flush()
        
    results = {'results_loss': results_loss, "final_loss":loss_current,
               'results_w':results_w, 'final_w':w_}
    if printout:
        print('')
    return(results)
    
#test w/ loss vs iteration:
print('testing gradient descent with lambda = 0:')
k0 = gradient_descent(x_test,target_test, 0, initial_w = np.array([0,-1,0]),
                      convergence = 0.0, max_step = 1000)
print('testing gradient descent with lambda = 1:')
k1 = gradient_descent(x_test,target_test, 1, initial_w = np.array([0,-1,0]),
                      convergence = 0.0, max_step = 1000)

plt.figure()
plt.title('loss vs iterations of gradient descent test')
plt.plot(k0['results_loss'],'b')
plt.plot(k1['results_loss'],'r')
plt.legend(['lambda = 0', 'lambda = 1'])
plt.show()

# test by mapping gradient descent:
f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('following gradient descent')
ax1.imshow(results_0, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax1.set_title('lambda = 0')
ax1.plot(k0['results_w'][:,1],k0['results_w'][:,2])

ax2.imshow(results_1, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax2.set_title('lambda = 1')
ax2.plot(k0['results_w'][:,1],k1['results_w'][:,2])

for ax in (ax1,ax2):
    ax.text(1,1,'+1', color = 'b')
    ax.text(-1,-1,'0', color = 'b')
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
f.show()

# %%
#one more test before the big show:  Let's make sure gradient descent works on
#our training data:

k0 = gradient_descent(X,target, 0, max_step = 10)
k1 = gradient_descent(X,target, 1, max_step = 10)

plt.figure()
plt.title('loss vs iterations of gradient descent on training set')
plt.plot(k0['results_loss'],'b')
plt.plot(k1['results_loss'],'r')
plt.legend(['lambda = 0', 'lambda = 1'])
plt.show()

# %% 
# finally, let's try all this on our training data for email:
from sklearn.cross_validation import KFold

kf = KFold(len(X), n_folds = 2, shuffle = True)

mean_error_results = []

# how to explore lambda?  Well, let's start by assuming its gradient's size
# should be on the order of one
n = 15
lambda_list = np.logspace(-5,1,n)
for i in range(0, len(lambda_list)):
    lmbda = lambda_list[i]
    
    loss_results = []
    print('iteration',i,'of',len(lambda_list))
    for train_index,test_index in kf:
        w_ = np.array([1 for i in range(len(X[1]))])
    
        train = X[train_index,:]
        train_t = target[train_index]
    
        validate = X[test_index]
        validate_t = target[test_index]
    
        fit = gradient_descent(train,train_t, lmbda,max_step = 15)
        weights = fit["final_w"]
    
        valid_loss = loss(weights, validate, validate_t, 0)
        loss_results.append(valid_loss)
        
    mean_error = np.mean(loss_results)
    mean_error_results.append(mean_error)
    print('@lambda %8.5f:|w| =%8.5f lambda,|w| mean(err)=%8.5f' %
          (lmbda, np.linalg.norm(weights),mean_error))
    print('')

#plot it:
plt.figure()
plt.plot(np.log10(lambda_list),mean_error_results)
plt.xlabel('log_10 (lambda)')
plt.ylabel('mean error across 10-fold validator')
plt.show()