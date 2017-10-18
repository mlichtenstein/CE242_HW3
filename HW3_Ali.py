
# coding: utf-8

# In[3509]:
#some basic configuration stuff:
Quick_Run_Flag = False;  #do a quick run for testing???
Quick_Run_n = 100;          #on a quick run, how many samples?        
Feature_len = 30; #desired feature length

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
target = [(0 if t == "ham" else 1) for t in target_strings]
print('\nThere are',len(target),'labels, of which',sum(target),'are spam')

# In[3516]:
# Tokenize and transform subject line strings from raw data:

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer(
    ngram_range = (1,2),
    max_features= 700, 
    stop_words = "english")
X_full  = vectorizer.fit_transform(X_strings).toarray()

print('X has shape',X_full.shape)

#%%
#let's do some dimensionality reduction
#we also do an SVD transform to do some dimesionality reduction:
import matplotlib.pyplot as plt

(u,s,v) = np.linalg.svd(X_full,    full_matrices= True)

X = u[:,0:Feature_len].dot(np.diag(s[0:Feature_len]))

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

def loss(w_,x,target,lmbda):
    J =0
    for i,row in enumerate(x):
        y = sigmoid(w_,row)
        J += - (target[i]*np.log(y) + (1-target[i])*np.log(1-y))
    reg = lmbda * np.sum(w_**2)
    return ( (J) + reg)

#%%
#test this:
w0_domain = np.linspace(-2,2,100)
w1_domain = np.linspace(2,-2,100)
w_test = np.array([[1, 0]]).transpose() # a column vector
target_test = [1,0]
x_test = np.array([[1,1],[-1,-1]]) # a ham in the upper right, a spam in the lower left
results_0 = np.zeros([100,100])
results_1 = np.zeros([100,100])
for i,w0 in enumerate(w0_domain):
    for j,w1 in enumerate(w1_domain):
        w_here = np.array([[w0,w1]]).transpose() # a column vector
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
    ax.text(1,1,'+1', color = 'b')
    ax.text(-1,-1,'0', color = 'b')
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
        reg = lmbda * np.sqrt(np.sum(w_**2))
        partials.append(J+ reg)
        
    return(np.array(partials))
    
print(partial_der(w_test,x_test,target_test,0))

#%%
#test this:
w0_domain = np.linspace(-2,2,10)
w1_domain = np.linspace(2,-2,10)
w_test = np.array([[1, 0]]).transpose() # a column vector
target_test = [1,0]
x_test = np.array([[1,1],[-1,-1]]) # a ham in the upper right, a spam in the lower left
pd_results_0 = []
pd_results_1 = []
for i,w0 in enumerate(w0_domain):
    for j,w1 in enumerate(w1_domain):
        w_here = np.array([[w0,w1]]).transpose() # a column vector
        results_2.append(partial_der(w_here,x_test,target_test,0))
        results_3.append(partial_der(w_here,x_test,target_test,0))

f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('loss over w')
ax1.imshow(results_0, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax1.set_title('lambda = 0')
ax2.imshow(results_1, cmap='hot', interpolation='nearest',
            extent=[-2, 2, -2, 2])
ax2.set_title('lambda = 1')

for ax in (ax1,ax2):
    ax.text(1,1,'+1', color = 'b')
    ax.text(-1,-1,'0', color = 'b')
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
f.show()
# In[3531]:

test = partial_der(w,X,target,1)
print(test)
test.shape


# In[3532]:

def w_update(w_, partial, rate, iteration_no):
    w_ = w_ - rate*(iteration_no**-0.9)*partial
    return(w_)


# In[3533]:

lmbda = 0.1


# In[3534]:

def gradient_descent(w_,X,target, lmbda = lmbda,rate = 0.01, max_step = 100, convergence = 0.01):
    w_ = w_*0
    l = loss(w_,X,target, lmbda)
    results_loss = {0:l}
    counter = 1
    l_prev = l+10
    while (abs(l - l_prev) > convergence) and (counter < max_step):
        l_prev = l
        partial = partial_der(w_, X, target, lmbda)
        w_ = w_update(w_, partial, rate, counter)
        l = loss(w_, X, target, lmbda)
        results_loss[counter] = l
        counter += 1
        
    results = [results_loss,{"Weights in limit": w_ },{"limit":l}]
    return(results)
    


# In[3535]:

k = gradient_descent(w,X,target)


# In[3536]:

print(k[1])


# In[3537]:

i = list(k[0].keys())
l = list(k[0].values())



# In[3538]:

import matplotlib.pyplot as plt


# In[3539]:

get_ipython().magic('matplotlib inline')
fig, ax1 =plt.subplots()
ax1.plot(i,l,color='blue',label='0.01')
ax1.legend(loc='upper left')
ax1.tick_params(bottom='off',top='off',left='off',right='off')
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
plt.title('Loss constant Lambda')
plt.show()


# In[3540]:

from sklearn.cross_validation import KFold


# In[3541]:

target = np.array([i for i in target])


# In[3542]:

kf = KFold(len(X), n_folds = 10, shuffle = True)

mean_error_results = []

n = 10
lambda_list = np.logspace(-2,2,n)
for i in range(0, len(lambda_list)):
    lmbda = lambda_list[i]
    
    loss_results = []
    print('iteration',i,'of',len(lambda_list))
    for train_index,test_index in kf:
        w_ = np.array([1 for i in range(len(X[1]))])
    
        train = X[train_index]
        train_t = target[train_index]
    
        validate = X[test_index]
        validate_t = target[test_index]
    
        fit = gradient_descent(w_,train,train_t, lmbda)
        weights = fit[1]["Weights in limit"]
    
        valid_loss = loss(weights, validate, validate_t, 0)
        loss_results.append(valid_loss)
        
    mean_error = np.mean(loss_results)
    mean_error_results.append(mean_error)
    print('@lambda %8.2f:|w| =%8.2f lambda,|w| mean(err)=%8.3f' %
          (lmbda, np.linalg.norm(weights),mean_error))

    
    
    
    



# In[3543]:

len(loss_results)


# In[ ]:




# In[ ]:



