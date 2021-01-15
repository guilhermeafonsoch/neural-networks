# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:43:50 2021

@author: guilh
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

data_iris = datasets.load_iris()

entradas= data_iris.data
saidas = data_iris.target 

rede_neural = MLPClassifier(verbose=True, 
                            max_iter=1000,
                            tol = 0.00001,
                            activation = "logistic",
                            learning_rate_init = 0.001,
                            )

rede_neural.fit(entradas, saidas)
rede_neural.predict([[5,7.2,5.1,10]])
