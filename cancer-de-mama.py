# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:45:34 2021

@author: guilh
"""

import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#DERIVADA PARCIAL DA SIGMOID
def derivadaSigmoid(sig):
    return sig * (1 - sig)

#database
base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569, 1], dtype=int)

for i in range(569):
    saidas[i] = valoresSaida[i]

#pesos aleatorios com a quantidade de neuronios
pesos0 = 2*np.random.random((30,5)) - 1
pesos1 = 2*np.random.random((5,1)) - 1

epocas = 10000

taxaDeAprendizagem = 0.3

momento = 1

for i in range(epocas):
    camadaEntrada = entradas
    
    #CAMADA OCULTA 
    somaDaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaDaSinapse0)
    
    #ULTIMA CAMADA 
    somaDaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaDeSaida = sigmoid(somaDaSinapse1)
    
    #ERRO = RESPOSTA CORRETA - RESPOSTA ERRADA
    erroDaCamadaDeSaida = saidas - camadaDeSaida
    mediaAbsolutaDoErro = np.mean(abs(erroDaCamadaDeSaida))
    print("\nErro de --> " + str(mediaAbsolutaDoErro))
    
    #DERIVADA E O DELTA DA CAMDADA DE SAIDA
    derivadaDeSaida = derivadaSigmoid(camadaDeSaida)
    deltaDeSaida = erroDaCamadaDeSaida * derivadaDeSaida
    
    
    #Para fazer a multiplicacao de peso com o delta de saida
    pesos1Transposta = pesos1.T
    
    #Formula = DeltaDeSaida * pesos * derivada sigmoide da camada oculta
    deltaCamadaOculta = deltaDeSaida.dot(pesos1Transposta) * derivadaSigmoid(camadaOculta)
    
    #Transposta da camada oculta para a multiplicacao de matrizes
    camadaOcultaTransposta = camadaOculta.T
    
    #atualizacao dos pesos da saida
    novosPesosSaida = camadaOcultaTransposta.dot(deltaDeSaida)
    pesos1 = (pesos1 * momento) + (novosPesosSaida * taxaDeAprendizagem)
    
    #atualizacao dos pesos da camda oculta
    camadaEntradaTransposta = camadaEntrada.T
    novosPesosCamadaOculta = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (novosPesosCamadaOculta *  taxaDeAprendizagem)
    
    
    