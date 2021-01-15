# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:24:29 2021

@author: guilh
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer


rede_neural = buildNetwork(2,3,1)
base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

treino = BackpropTrainer(rede_neural, dataset= base, 
                         learningrate= 0.01,
                         momentum = 0.06)

for i in range (1, 30000):
    erro = treino.train()
    if i % 1000 == 0:
        print("Erro --> " + str(erro))
        
print(rede_neural.activate([0,0]))
print(rede_neural.activate([0,1]))
print(rede_neural.activate([1,0]))
print(rede_neural.activate([1,1]))

