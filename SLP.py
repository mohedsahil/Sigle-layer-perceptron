# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:36:01 2019

@author: sahil
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from mlxtend.plotting import plot_decision_regions 
from sklearn.preprocessing import StandardScaler

w = []
 
data = pd.read_csv('slp.csv')
b = []
y_acc = []
x_epoc = []

data = data.iloc[:,[2,3,4]]
#adding bais value 
for i in range(len(data.index)):
    b.append(-1)
data.insert(loc=0,column='bais',value=b)

# Feature Scaling(normalizing the data)
sc = StandardScaler()
data.iloc[:,[1,2]] = sc.fit_transform(data.iloc[:,[1,2]])

# Splitting the dataset into the Training set and Test set
X = data.iloc[:,[0,1,2]]
y = data['Purchased']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

def initilize_weights(data):
    no_of_weigths = len(data.columns)
    for i in range( no_of_weigths-1):
       w.append( round(random.uniform(-0.05,0.05),2))

        
def accuracy(predvalues,target,i,stop_early):
    print('Accuracy score: {}'.format(accuracy_score(target, predvalues)))
    #for early stop
    if stop_early == True:
        y_acc.append(accuracy_score(target, predvalues))
        x_epoc.append(i)
        plt.subplot(2,1,1)
        plt.plot(x_epoc,y_acc)
        plt.title("early stop")
        plt.xlabel("epocs")
        plt.ylabel("accuracy")
    

def update_weigths(w,n,ypred,t,dataarray,j):
    for i in range(len(w)):
        w[i] = w[i]- (n*(ypred-t)*dataarray[j][i])
  
    
def confusionmatrix(predvalues,t):
    print("confusion matrix")
    cm = confusion_matrix(t,predvalues)
    print(cm)
    print("<-------------------------------------------------->")
    print("\n")
    
  
def plot_decision(X,predvalues,target):
    plt.subplot(2,1,2)
    from matplotlib.colors import ListedColormap
    h = .02
    X_set, y_set = X,target 
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,h),
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,h))
    plt.contourf(X1, X2, slp.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
           plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
l =[]      
predvalues = []     
class slp:  
    def train_model(epocs,learning_rate):
        y=0
        #imp
        target = np.array(y_train)
        dataarray= np.array(X_train)
        for i in range(epocs):
            print("starting with epocs ------->"+" "+str(i+1) )
            print("\n")
            predvalues = []
            for k in range(len(X_train.index)):
                 for j in range( len(X_train.columns)-1):
                      y = y + (dataarray[k][j]*w[j])
                 #print("activation value:" +" "+str(y))
                 if y>0:
                     ypred = 1
                 else:
                     ypred = 0
                 predvalues.append(ypred)
                 if ypred != target[k]:
                     update_weigths(w,learning_rate,ypred,target[k],dataarray,k)
                     print("updated weightd are :")
    
                 print(w)
                     
            #i is the nth epoc
            print("predicted values for epoc"+" "+ str(i+1)+"  is:")
            print(predvalues)
                       
            accuracy(predvalues,target,i,True)
            confusionmatrix(predvalues,target)
        plot_decision(dataarray[:,1:3],predvalues,target)   
        
    def predict(x):
        predictedvalues = []
        for i in x:
            y=0
            for j in range(len(i)):
                  y = y + (i[j]*w[j])
                  
            if y>0:
                predictedvalues.append(1)
            else:
                predictedvalues.append(0)        
        predictedvalues = np.array(predictedvalues)
        return predictedvalues 
    
                             
        
initilize_weights(data)

slp.train_model(epocs=15,learning_rate=0.2)




#testing the dataset 

X_test = np.array(X_test)
ypredicted = slp.predict(X_test)
print(ypredicted)
print("accuracy of test data is")
accuracy(ypredicted,y_test,0,False)
print("confusion matrix")
confusionmatrix(ypredicted,y_test)
print("plot for test dataset")
plot_decision(X_test[:,[1,2]],ypredicted,y_test) 
    
#testing just give x values in csv file    
  
    