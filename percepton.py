# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:21:38 2021

@author: guilh
"""
#Este tipo de percepton de uma camada serve para problemas linarmente separaveis 

import numpy as np
#or
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])

pesos = np.array([0.0, 0.0])

taxaDeAprendizagem = 0.1

def stepFunction(soma):
   if (soma >= 1):
       return 1
   return 0 

#registros = entradas
def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while erroTotal != 0:
        erroTotal = 0
        for i in range(len(saidas)):
            #calculo das saidas
            print(entradas[i])
            print(pesos)
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = saidas[i] - saidaCalculada
            erroTotal += erro
            for j in range (len(pesos)):
                pesos[j] = pesos[j] + (taxaDeAprendizagem * entradas[i][j] * erro)
                print("Pesos atualizados: " + str(pesos[j]))
                
        print("Total de erros: " + str(erroTotal))

treinar()
print("\nRede neural atualizada:")
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3])) 


    
