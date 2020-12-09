#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:58:28 2020

@author: reeseboucher
"""

import heisenbergModel as hm

x=hm(1000, -1, 10000, plotBool=True)

#Simulation of lattice with heisenberg model in 3d
x.threeDimensions([10,10,10])  

#Simulation of lattice with heisenberg model in 2d
x.twoDimensions(40,25)

#Simulation of lattice with heisenberg model in 1d
x.oneDimension()

#Create new heisenberg model object to simulate the Ising Model
ising = hm(1000, -1, 10000, plotBool=True, heisenberg=False)

#Simulation of Ising model in 2D
ising.twoDimensions(40,25)