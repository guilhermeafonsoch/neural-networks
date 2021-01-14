# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:15:41 2021

@author: guilh
"""

import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

#DERIVADA PARCIAL DA SIGMOID
def derivadaSigmoid(sig):
    return sig * (1-sig)

entradas = np.array([[0,0], 
                     [0,1], 
                     [1,0], 
                     [1,1]])
 
saidas = np.array([[0], 
                   [1], 
                   [1], 
                   [0]])

#PESOS FIXOS
#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])

#pesos1 = np.array([[-0.017],
#                   [-0.893],
#                   [0.148]])


#PESOS ALEATORIOS
pesos0 = 2 * np.random.random((2,3)) -1

pesos1 = 2 * np.random.random((3,1)) -1
  
epocas = 1000000
taxaDeAprendizagem = 0.6
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
    
    
    