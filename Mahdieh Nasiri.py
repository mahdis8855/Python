#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import all nececssary libraries ##
## split the dataset into train and test set ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

A         = pd.read_csv('diabetes.csv')
train_no  = int(0.8*len(A))+1
train_set = A.iloc[0:train_no,:]
test_set  = A.iloc[train_no:,:]

train_output = A.iloc[0:train_no,5]
test_output  = A.iloc[train_no:,5]

del train_set[train_set.columns[5]]
del test_set[test_set.columns[5]]


# In[2]:


train_set.head(5)


# In[3]:


test_set.head(5)


# In[4]:


print(train_output)


# In[5]:


print(test_output)


# In[6]:


print(test_output)


# In[7]:


## Train a Linear Regression model to predict "BMI" using all other features avaiable in the dataset ##

# 1.Standardization

mu             = np.mean(train_set)
sigma          = np.std(train_set)
train_set_norm = (train_set - mu)/sigma
train_set_norm.head(5)


# In[8]:


# 2.Comouting the cost function

def cost(X,Y,theta):
    m = len(Y)
    J = 0           #iterative solution
    s = 0
    for i in range(0,m):
        s = s+((theta[0]+
              theta[1]*X.iloc[i][1]+
              theta[2]*X.iloc[i][2]+
              theta[3]*X.iloc[i][3]+
              theta[4]*X.iloc[i][4]+
              theta[5]*X.iloc[i][5]+
              theta[6]*X.iloc[i][6]+
              theta[7]*X.iloc[i][7]+
              theta[8]*X.iloc[i][8])-Y[i])**2
    J  = s/(2*m)
    J = float(J)
    return J


# In[9]:


# 3.Gradient decsent

def gradientDescent(X,Y,theta,alpha,iterations):
    J_his = np.zeros((iterations,1))

    m = len(Y)
    Y = Y.reshape(m,1)
    for iter in range(0,iterations):
        s8=0
        s7=0
        s6=0
        s5=0
        s4=0
        s3=0
        s2=0
        s1=0
        s0=0
        for i in range(0,m):
            s8+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][8]
            s7+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][7]
            s6+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][6]
            s5+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][5]
            s4+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][4]
            s3+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][3]
            s2+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][2]
            s1+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])*X.iloc[i][1]
            s0+=((theta[0]+theta[1]*X.iloc[i][1]+theta[2]*X.iloc[i][2]+theta[3]*X.iloc[i][3]+theta[4]*X.iloc[i][4]+theta[5]*X.iloc[i][5]+theta[6]*X.iloc[i][6]+theta[7]*X.iloc[i][7]+theta[8]*X.iloc[i][8])-Y[i])
                 
        theta[0]=theta[0]-alpha*s0/m
        theta[1]=theta[1]-alpha*s1/m
        theta[2]=theta[2]-alpha*s2/m
        theta[3]=theta[3]-alpha*s3/m
        theta[4]=theta[4]-alpha*s4/m
        theta[5]=theta[5]-alpha*s5/m
        theta[6]=theta[6]-alpha*s6/m
        theta[7]=theta[7]-alpha*s7/m
        theta[8]=theta[8]-alpha*s8/m
        
        J_his[iter] = cost(X,Y,theta)
    return theta,J_his


# In[10]:


# 3. Training the linear model using grid search to find optimal learning rate.  
a           = np.arange(0,1,0.005)
parameters  = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': a,'max_iter': [1000,100000]}]
clf         = GridSearchCV(SVR(), parameters)
clf.fit(train_set, train_output)
print(clf.best_params_)


# In[11]:


print(clf.best_params_)


# In[12]:


m                = len(train_output)
theta            = np.ones((9,1))                                            
iterations       = 50
alpha            = 0.915
itrain_set       = np.c_[np.ones((len(train_set_norm),1)),train_set_norm]    
itrain_set       = pd.DataFrame(itrain_set, columns = ['ones','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','Age','Outcome'])
to_array         = np.array(train_output) 
theta1,J_his     = gradientDescent(itrain_set,to_array,theta,alpha,iterations)


# In[13]:


itrain_set.head(5)


# In[14]:


# Plot the "cost function vs iterations" curve
it = range(0,iterations)
plt.scatter(it, J_his,color='green')
plt.title('Cost function vs Iteration number')
plt.xlabel('iteration number')
plt.ylabel('Cost')
plt.show()


# In[15]:


pd.DataFrame(theta)


# In[16]:


def predict(theta,X):
    return (theta[0]*(X.iloc[:,0].to_numpy())+theta[1]*(X.iloc[:,1].to_numpy())+theta[2]*(X.iloc[:,2].to_numpy())+theta[3]*(X.iloc[:,3].to_numpy())+theta[4]*(X.iloc[:,4].to_numpy())+theta[5]*(X.iloc[:,5].to_numpy())+theta[6]*(X.iloc[:,6].to_numpy())+theta[7]*(X.iloc[:,7].to_numpy())+theta[8]*(X.iloc[:,8].to_numpy()))


# In[17]:


# plot the "Predicted BMI value( 𝑌̂ 𝑖 )" vs "Actual BMI value( 𝑌𝑖 )" curve(it is a scatter plot)
mu             = np.mean(test_set)
sigma          = np.std(test_set)
test_set_norm = (test_set - mu)/sigma
test_set_norm.insert(0, 'new-col',1)
test_set_norm.head(5)


# In[18]:


pd.DataFrame(test_output) #actual


# In[43]:


def predict_test(theta,X):
    return (theta[0]*(X.iloc[:,0])+theta[1]*(X.iloc[:,1])+theta[2]*(X.iloc[:,2])+theta[3]*(X.iloc[:,3])+theta[4]*(X.iloc[:,4])+theta[5]*(X.iloc[:,5])+theta[6]*(X.iloc[:,6])+theta[7]*(X.iloc[:,7])+theta[8]*(X.iloc[:,8]))
# Mean Square error for training set:
    
MSE1 =  round((((train_output - predict(theta1, itrain_set))**2).sum()/len(train_output)/615),5)
print("The MSE for training set is ", MSE1)


# In[20]:


Ytest = predict_test(theta,test_set_norm)
pd.DataFrame(Ytest)


# In[21]:


plt.scatter(test_output,Ytest,color='red')
plt.title('Predicted BMI vs Actual BMI value')
plt.xlabel('Actual BMI value')
plt.ylabel('Predicted BMI')
plt.show()


# In[45]:


# Mean Square error for testing set:
mu             = np.mean(test_set)
sigma          = np.std(test_set)
test_set_norm = (test_set - mu)/sigma
itest_set = np.c_[np.ones((len(test_set_norm),1)),test_set_norm]
itest_set = pd.DataFrame(itest_set, columns = ['ones','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','Age','Outcome'])

MSE2 = round((((test_output - predict(theta1, itest_set))**2).sum()/len(test_output)/154),5)                 
print("The MSE for testing set is ", MSE2)


# In[ ]:




