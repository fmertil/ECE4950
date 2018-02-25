#Francois Mertil
#ECE 4950
#HW2



# Gaussian Naive Bayes Skeleton Code: 

%matplotlib inline
import numpy as np
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

data = np.genfromtxt('wbdc.txt', delimiter=',')

size = [50, 100, 150, 200, 250, 300, 350, 400, 450]
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :]
    X_trn, y_trn = train[:, 2:], train[:,1]
    X_tst, y_tst = test[:, 2:], test[:,1]
    
    # Compute the label probabilities
g = GaussianNB()
g.fit(X_trn, y_trn)  # GaussianNB itself does not support sample-weights
prob_g = g.predict_proba(X_tst)[:, 1]
print (prob_g)
    
    # Compute the class conditional means, and variance for Gaussian Naive Bayes
    # You 'may' obtain four arrays/lists of size 30, two for means for each label, 
    # and two for variances for each label. 
    
    # Compute the training error, and test error of the GNB model you learnt.
    # You may want to use logarithms for computations
    
# In a single plot, show the values of training and test 
    

