# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:05:30 2021

@author: guilh
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

#criando a rede neural
rede_neural = FeedForwardNetwork()

#criacao das camadas
camadaDeEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)

#criacao dos bias
biasDaCamadaOculta = BiasUnit()
biasDaCamadaSaida = BiasUnit()

#colocando os valores na rede
rede_neural.addModule(camadaDeEntrada)
rede_neural.addModule(camadaOculta)
rede_neural.addModule(camadaSaida)
rede_neural.addModule(biasDaCamadaOculta)
rede_neural.addModule(biasDaCamadaSaida)


#ligacao dos neuronios
entradaOculta = FullConnection(camadaDeEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(biasDaCamadaOculta, camadaOculta)
biasSaida = FullConnection(biasDaCamadaSaida, camadaSaida)

rede_neural.sortModules()

print(rede_neural)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)


