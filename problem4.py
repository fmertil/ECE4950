Python 3.6.0a1 (v3.6.0a1:5896da372fb0, May 16 2016, 15:20:48) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> WARNING: The version of Tcl/Tk (8.5.9) in use may be unstable.
Visit http://www.python.org/download/mac/tcltk/ for current information.
#Francois Mertil
#ECE 4950
#HW2


# Decision Tree Problem Skeleton Code: 


# For this problem, you can refer to: 
# http://scikit-learn.org/stable/modules/tree.html and
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# Some of the libraries you might need are loaded:
%matplotlib inline
import numpy as np
import sklearn 
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Feel free to use other libraries.

data = np.genfromtxt('wbdc.txt', delimiter=',') # wdbc is loaded into an array 'data'


# PART ONE:

size = [50, 100, 150, 200, 250, 300, 350, 400, 450] 
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
    X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
    X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
    
    # Compute the training and test errors for depth = 2, and plot them as a function of sz
    # Compute the training and test errors for depth = 3, and plot them as a function of sz
    
    # CODE GOES HERE (YOU CAN USE THE SKLEARN DECISION TREES)
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_trn, y_trn)
predictions = clf.predict(X_tst)

clf2 = tree.DecisionTreeClassifier(max_depth=3)
clf2 = clf.fit(X_trn, y_trn)
predictions2 = clf.predict(X_tst)
#print (predictions)
#print ("Accuracy is : %f" %accuracy_score(y_tst, predictions))
# You will have a total of two plots for this part (one for each depth)
#dot_data = tree.export_graphviz(clf, out_file=None,   
 #                        filled=True, rounded=True,  
 #                        special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph 

test_errors=[]
test_errors2=[]
for test_predict, test_predict2 in zip(train, train):
    test_errors.append(
        1. - accuracy_score (predictions, y_tst))
    test_errors2.append(
        1. - accuracy_score (predictions2, y_tst))
    #print ("Accuracy is : %f" %accuracy_score(y_tst, predictions))

n = len(train)
#print ("size is :  %f" %n)

m = len(test_errors)
#print ("size is :  %f" %m)
plt.figure()
plt.subplot(121)
plt.plot(range(1,n+1), test_errors)
plt.xlabel('size')
plt.ylabel('test_errors')
plt.title('test_errors vs size')

plt.subplot(122)
plt.plot(range(1,n+1), test_errors2)
plt.xlabel('size')
plt.ylabel('test_errors')
plt.title('test_errors2 vs size')
plt.show()


size = [200, 400]

for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
    X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
    X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
    
a=[]
b=[]
    # Train decision trees of maximum depth = 2,...,12 using the training set using IG criterion.
for i in range (2,13):
    a.append( tree.DecisionTreeClassifier(max_depth=i, criterion='gini'))

    #a.append (a.fit(X_trn, y_trn))
#predictions = clf.predict(X_tst)
#print(a)
#print (len(a))

prediction=[]
for j in range (len(a)):
    print(b.append( a[j].fit(X_trn, y_trn)))
    prediction.append(b[j].predict(X_tst))
    
print(a[1].predict(X_tst))
    # Plot the training and test error as a function of the depth.
    # Repeat the above using gini index
    
    # CODE GOES HERE 
    

# You will have a total of four plots for this part (2 training sets and two decision criteria)


# PART 3: We will use minimum leaf size, which ensures that each node must have at least a certain number of samples
size = [50, 200, 400]
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
    X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
    X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
    
    # Train decision trees with minimum leaf size = 1, 2,..., 15 using the training set using IG criterion.
    # Plot the training and test error as a function of the depth.
    # Repeat the above using gini index
    
    # CODE GOES HERE 
    

# You will have a total of three plots for this part


# Please justify the plots. You may want to use random_state= 0 or some small integer in the sklearn. 


