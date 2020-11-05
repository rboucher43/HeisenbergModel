#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:34:07 2020

@author: reeseboucher

N number of electrons 

D number of dimensions

B_ext array of components of external B field 
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab


class heisenbergModel:
        
    def __init__( self, N, exchange, mcSteps = 0, heisenberg = True, magneticField = None, plotBool = False, monteCarlo = False):
        self.N             = N
        self.mcSteps       = mcSteps
        self.exchange      = exchange
        self.heisenberg    = heisenberg
        self.magneticField = magneticField
        self.plotBool      = plotBool
        self.monteCarlo    = monteCarlo


    def updatePlane( self, lattice, position ):
        '''
        Calculates change in energy accounting for neighboring interactions with spin flip in given plane. 
        
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins
        position : Array
            Input array of position in lattice where spin will be flipped and energy will be updated
        Returns
        -------
        newInteraction : Float
            Interaction after change in spin flip of neighboring spins
        oldInteraction : Float
            Interaction before change in spin flip from neighboring spins

        '''
        
        newInteraction = 0
        oldInteraction = 0
        
        if position[1] > 0:
            newInteraction      +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]][position[1]-1])   #left
            oldInteraction      -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]][position[1]-1])
        if position[1] < len(lattice[position[0]])-1:
            newInteraction      +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]+1]) #right
            oldInteraction      -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]+1])
        if position[0] > 0:
            newInteraction      +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]])
            oldInteraction      -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]])   #upper interaction
            if position[1] > 0:
                newInteraction  +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]-1])
                oldInteraction  -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]-1]) #upper left interaction
            if position[1] < len(lattice[position[0]])-1:
                newInteraction  +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]+1])
                oldInteraction  -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]-1][position[1]+1]) #upper right interaction
        if position[0] < len(lattice)-1:
            newInteraction      +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]])
            oldInteraction      -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]])   #lower interaction
            if position[1] > 0:
                newInteraction  +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]-1])
                oldInteraction  -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]-1])  #lower left interaction
            if position[1] < len(lattice[position[0]])-1:
                newInteraction  +=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]+1])
                oldInteraction  -=  np.dot(lattice[position[0]][position[1]], lattice[position[0]+1][position[1]+1])  #lower right interaction
        
        return newInteraction, oldInteraction

        
    def planeEnergy(self, lattice, flipped = False):
        '''
        Calculates total energy accounting for neighboring interactions with spin flip in given plane. 

        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins
        flipped : Boolean, optional
            Determines whether every neighboring site in plane should be calculated(flipped=false), 
            or in case of 3D slices only unclaulated interactions are accounted for. 
            The default is False.
        Returns
        -------
        totalEnergy : Float
            Total calculated energy of input lattice accounting for neighboring interactions.
        '''
        totalEnergy = []
        if flipped == False:
            for i in range(len(lattice)):
                for j in range(len(lattice[i])-1): 
                    totalEnergy = np.append(totalEnergy, np.dot(lattice[i][j+1:1+j+1],lattice[i][j])) #right
            
        for i in range(len(lattice)-1):
            j = 1
            while j < len(lattice[i]):                 
                totalEnergy = np.append(totalEnergy,np.dot(lattice[i][j], lattice[i+1][j-1]))         #low Left
                if j+1 < len(lattice[i]):
                    totalEnergy = np.append(totalEnergy,np.dot(lattice[i][j], lattice[i+1][j+1]))     #low Right
                j += 2   
            if flipped == False:
                for q in range(len(lattice[i])): # down 
                    totalEnergy = np.append(totalEnergy,np.dot(lattice[i+1][q], lattice[i][q]))      
        
        for i in range(1,len(lattice)):
            j = 1
            while j < len(lattice[i]): 
                totalEnergy = np.append(totalEnergy,np.dot(lattice[i][j], lattice[i-1][j-1]))        #up Left
                if j+1 < len(lattice[i]):
                    totalEnergy = np.append(totalEnergy,np.dot(lattice[i][j],lattice[i-1][j+1]))     #up Right
                j += 2  
        
        totalEnergy = np.sum(totalEnergy)
        
        return totalEnergy
    
    def updateEnergy(self, lattice, position, dimension):
        '''
        Calculates change in energy accounting for neighboring interactions for given lattice. 

        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
       position : Array
            Input array of position in lattice where spin will be flipped and energy will be updated
        dimension : Array
            Dimension of lattice. Detemrines how energy is updated.

        Returns
        -------
        deltaE : Float
            Change in energy in system from neighboring interaction from updated spin flip.
        '''
        deltaE         = 0
        newInteraction = 0
        oldInteraction = 0
        
        if dimension == 1: 
            lattice[position]   = lattice[position] * -1
            if position > 0:
                newInteraction += lattice[position] * lattice[position-1]
                oldInteraction -= lattice[position] * lattice[position-1]
                
            if position < len(lattice)-1:
                newInteraction += lattice[position] * lattice[position+1]
                oldInteraction -= lattice[position] * lattice[position+1]
              
        elif dimension == 2:
            lattice[position[0]][position[1]] = lattice[position[0]][position[1]]*-1
            newInteraction, oldInteraction    = self.updatePlane(lattice, position)
        
        elif dimension == 3:
            
            lattice[position[0]][position[1]][position[2]] = lattice[position[0]][position[1]][position[2]]*-1
            
            if position[0] < len(lattice)-1:
                newInteraction         += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]][position[2]])
                oldInteraction         -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]][position[2]])     #in
                if position[2] < len(lattice[position[0]][position[1]])-1:
                     newInteraction    += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]][position[2]+1])  
                     oldInteraction    -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]][position[2]+1])   #in right
                if position[2] > 0:
                     newInteraction    += np.dot(lattice[position[0]][position[1]][position[2]],lattice[position[0]+1][position[1]][position[2]-1])  
                     oldInteraction    -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]][position[2]-1])   #in left
                if position[1] > 0:
                    newInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]])
                    oldInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]])   #in up
                    if position[2] < len(lattice[position[0]][position[1]])-1:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]+1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]+1]) #in up right
                    if position[2] > 0:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]-1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]-1][position[2]-1]) #in up left
                      
                if position[1] < len(lattice[position[0]])-1:
                    newInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]])
                    oldInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]])   #in down 
                    if position[2] < len(lattice[position[0]][position[1]])-1:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]+1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]+1]) #in down right
                    if position[2] > 0:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]-1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]+1][position[1]+1][position[2]-1]) #in down left
                                  
            if position[0] > 0:
                newInteraction         += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]][position[2]]) #out
                oldInteraction         -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]][position[2]])
                if position[2] < len(lattice[position[0]][position[1]])-1:
                     newInteraction    += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]][position[2]+1]) #out right
                     oldInteraction    -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]][position[2]+1])
                if position[2] > 0:
                     newInteraction    += np.dot(lattice[position[0]][position[1]][position[2]],lattice[position[0]-1][position[1]][position[2]-1]) #out left
                     oldInteraction    -= np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]][position[2]-1])
                if position[1] > 0:
                    newInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]])
                    oldInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]])   #out up
                    if position[2] < len(lattice[position[0]][position[1]])-1:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]+1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]+1]) #out up right
                    if position[2] > 0:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]-1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]-1][position[2]-1]) #out up left
                if position[1] < len(lattice[position[0]])-1:
                    newInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]])
                    oldInteraction     += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]])   #out down 
                    if position[2] < len(lattice[position[0]][position[1]])-1:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]+1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]+1]) #out down right
                    if position[2] > 0:
                        newInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]-1])
                        oldInteraction += np.dot(lattice[position[0]][position[1]][position[2]], lattice[position[0]-1][position[1]+1][position[2]-1]) #out down left
            
            
            newInteractiondE, oldInteractiondE = self.updatePlane(lattice[position[0]], [position[1],position[2]])
            newInteraction += newInteractiondE
            oldInteraction += oldInteractiondE
            
        deltaE = -self.exchange * (newInteraction - oldInteraction)
  
        return deltaE

    
    def lineEnergy(self, lattice):
        '''
        Calculates inital total energy in given 1D lattice.

        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.

        Returns
        -------
        totalEnergy : Float
            Total Energy in given 1D lattice
        '''
        totalEnergy = []
        for site in range(len(lattice)):
            right       = lattice[site+1:1+site+1]
            totalEnergy = np.append(totalEnergy, lattice[site]*right)
        totalEnergy = np.sum(totalEnergy)
        return totalEnergy
        
    def diagonalEnergy(self,lattice):  
        '''
        Calculates initial inward/outward diagonal interaction for 3D lattices.
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.

        Returns
        -------
        totalEnergy : Float
            Total Energy of diagonal in given lattice

        '''
        for i in range(len(lattice[0])):
            diagonal     = lattice.diagonal(offset=i)
            antidiagonal = np.fliplr(lattice).diagonal(offset=i)  
        
        totalEnergy  = 0 
        totalEnergy += self.lineEnergy(diagonal)
        totalEnergy += self.lineEnergy(antidiagonal)
        return totalEnergy


    def prismEnergy(self, lattice):
        '''
        Calculates initial total energy for 3D lattices.
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.

        Returns
        -------
        totalEnergy : Float
            Total energy of given 3D lattice.
        '''
        totalEnergy = []
                         
        for latticePlane in range(len(lattice)-1):
            for row in range(1,len(lattice[latticePlane])):
                for particle in range(len(lattice[latticePlane][row])-1):
                    totalEnergy = np.append(totalEnergy, np.dot(lattice[latticePlane][row][particle],lattice[latticePlane+1][row-1][particle+1]))#in up right
                for particle in range(1,len(lattice[latticePlane][row])):
                    totalEnergy = np.append(totalEnergy, np.dot(lattice[latticePlane][row][particle],lattice[latticePlane+1][row-1][particle-1]))#in up left
                
            for row in range(len(lattice[latticePlane])-1):
                for particle in range(len(lattice[latticePlane][row])-1):
                    totalEnergy = np.append(totalEnergy, np.dot(lattice[latticePlane][row][particle],lattice[latticePlane+1][row+1][particle+1]))#in bottom right
                for particle in range(1,len(lattice[latticePlane][row])):
                    totalEnergy = np.append(totalEnergy, np.dot(lattice[latticePlane][row][particle],lattice[latticePlane+1][row+1][particle-1]))#in bottom left

        totalEnergy = np.sum(totalEnergy)
        
        for plane in range(len(lattice)): # vertical plane interactions
            totalEnergy += self.planeEnergy(lattice[plane])       

        for i in range(len(lattice[0])):  # horizontal plane interactions
            totalEnergy += self.planeEnergy(lattice[:,i], True) 
                
        totalEnergy = -self.exchange * totalEnergy       
        
        return totalEnergy
    
    def buildLattice(self, dimension, shape = None):   
        '''
        Creates lattice with shape determined by shape initialized with random spin for specified model either Heisenberg or Ising.
        
        Parameters
        ----------
        dimension : Array
            Dimension of desired lattice
        shape : TYPE, optional
            Shape of desired lattice. The default is None. 
            If shape equals None, the lattice defaults to the shape  (dimension by ((number of atoms)\dimension))

        Returns
        -------
        lattice : Array
            Lattice with random spin for specified model either Heisneberg or Ising.
        '''
        if shape == None:
            shape = [dimension, int(self.N/dimension)]   
        
        if self.heisenberg == False:  # Ising Model 
            lattice = np.random.randint(0, 2, self.N)
            lattice = np.where(lattice != 0, lattice, lattice-1)
            lattice = np.reshape(lattice,shape)

        elif self.heisenberg == True: # Heisenberg Model
            lattice = []
            for i in range(self.N):
                theta = np.random.randint(0,181)
                phi   = np.random.randint(0,361)
                if dimension == 1: ############################## Check this ######################################
                    randomVector = np.cos(phi)
                    
                elif dimension == 2:
                    randomVector = np.array([np.cos(phi),np.sin(phi)])

                elif dimension == 3:
                    randomVector = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
                lattice.append(randomVector)
            
            newLattice = lattice
            if self.magneticField != None:
                newLattice = np.array([])
                
                for vector in lattice:
                    newLattice = np.append( newLattice, self.project( vector, self.magneticField ) )
            
            shape.append(dimension)
            lattice = np.reshape( newLattice, shape )
            print(lattice)


        return lattice
    
    
    def project( self, u, v ):  #project u on v
        '''
        Projects vector u onto vector v
        
        Parameters
        ----------
        u : Array
            Input Vector that will be projected on v
        v : TYPE, optional
            Input vector that will be projected on
            
        Returns
        -------
        proj_of_u_on_v : Array
            Projection of vector u on vector v
        '''
    
        u              = np.array(u)
        v              = np.array(v) 
        v_norm         = np.sqrt(sum(v**2))     
        proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v 
        
        return proj_of_u_on_v

    
    
    def oneDimension( self ): 
        '''
        Builds 1D lattice, calculates total energy, then relaxes lattice to ground state based on neighboring interaction.
        Plots Monte Carlo distribution is self.monteCarlo == True

        Returns
        -------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        lowEnergy : Float
             Energy of final lattice configuration 
        '''
        lattice    = self.buildLattice( 1 )
        lattice    = lattice.flatten()
        lowEnergy  = self.lineEnergy( lattice )
        
        if self.monteCarlo == True:  
            monteCarloEnergies = np.array([])
            
            for steps in range( self.mcSteps ):
                randomPosition     = np.random.randint( 0, self.N )
                deltaE             = self.updateEnergy( lattice, randomPosition, 1 )
                
                monteCarloEnergies = np.append( monteCarloEnergies, self.lineEnergy(lattice) )
    
                if deltaE <= 0:
                    lattice[randomPosition] = lattice[randomPosition] * -1
                    lowEnergy += deltaE    
            self.plotMonteCarlo(monteCarloEnergies)
        
        else:
            for steps in range( self.mcSteps ):
                randomPosition = np.random.randint( 0, self.N )
                deltaE         = self.updateEnergy( lattice, randomPosition, 1 )
    
                if deltaE <= 0:
                    lattice[randomPosition] = lattice[randomPosition] * -1
                    lowEnergy += deltaE  

        lowEnergy = self.lineEnergy(lattice)
        return lattice, lowEnergy
    
    
    def twoDimensions(self, shape):
        '''
        Parameters
        ----------
        shape : Array
            Array filled with dimension of lattice

        Returns
        -------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        lowEnergy : Float
             Energy of final lattice configuration 

        '''
        lattice   = self.buildLattice(2, shape)
        lowEnergy = self.planeEnergy(lattice)
        
        
        if self.monteCarlo == True: 
            monteCarloEnergies = np.array([])
            for steps in range( self.mcSteps ):
                randomPosition   = [np.random.randint(0,shape[0]), np.random.randint(0,shape[1])]
                deltaE           = self.updateEnergy(lattice,randomPosition,2)
                monteCarloEnergies = np.append(monteCarloEnergies,self.planeEnergy(lattice))
                if deltaE < 0:
                    lattice[randomPosition[0]][randomPosition[1]] = lattice[randomPosition[0]][randomPosition[1]] * -1
                    print(steps)

            self.plotMonteCarlo(monteCarloEnergies)
        else:
            for steps in range( self.mcSteps ):
                randomPosition = [np.random.randint(0,shape[0]), np.random.randint(0,shape[1])]
                deltaE         = self.updateEnergy(lattice,randomPosition,2)
                if deltaE < 0:
                    lattice[randomPosition[0]][randomPosition[1]] = lattice[randomPosition[0]][randomPosition[1]] * -1 
        
                
        lowEnergy = self.planeEnergy(lattice)

        if self.plotBool == True:
            self.plotSpins( lattice, shape, 2 )
            
        return lattice, lowEnergy
    
    
    def threeDimensions(self, shape): 
        '''
        Parameters
        ----------
        shape : Array
            Array filled with dimension of lattice
            
        Returns
        -------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        lowEnergy : Float
             Energy of final lattice configuration 

        '''
        lattice   = self.buildLattice(3, shape)
        lowEnergy = self.prismEnergy(lattice) 
                       
        if self.monteCarlo == True:
            monteCarloEnergies = np.array([])
            for steps in range( self.mcSteps ):
                randomPosition     = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])]
                deltaE             = self.updateEnergy(lattice, randomPosition,3)
                monteCarloEnergies =    np.append(monteCarloEnergies, self.prismEnergy(lattice))
                
                if deltaE < 0:
                    lattice[randomPosition[0]][randomPosition[1]][randomPosition[2]] = lattice[randomPosition[0]][randomPosition[1]][randomPosition[2]] * -1
                    print(steps)
            self.plotMonteCarlo( monteCarloEnergies )
        
        else:
            for steps in range( self.mcSteps ):
               randomPosition = [np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])]
               deltaE         = self.updateEnergy(lattice, randomPosition,3)
               if deltaE < 0:
                   print(steps)
                   lattice[randomPosition[0]][randomPosition[1]][randomPosition[2]] = lattice[randomPosition[0]][randomPosition[1]][randomPosition[2]] * -1
       
        
        lowEnergy = self.prismEnergy(lattice)

        if self.plotBool == True:

            self.plotSpins( lattice, shape, 3 )

            
        return lattice, lowEnergy
    
    def plotMonteCarlo( self, energies ):
        '''
        Plots Distribution of given energies
        Parameters
        ----------
        energies : Array
            Array of each energy calculated as a spin was flipped

        Returns
        -------
        None.

        '''
        plt.hist(energies, histtype='step', bins="auto")
        plt.xlabel("Energy")
        plt.savefig("monteCarloDistribution_mag.png")
        plt.show()

    
    def plotSpins(self, lattice, shape, dimension):
        '''
        Creates image of input lattice based on dimension and spin 
        Parameters
        ----------
        lattice : Array
            Input arry with shape of lattice containing direction of spins.
        shape : Array
            Shape of desired lattice.
        dimension : Integer
            Dimension of desired lattice
        
        Returns
        -------
        None.

        '''
        lattice = np.array(lattice, dtype=float)
        if dimension == 2:

            x = np.array(np.tile(np.arange(shape[0]), shape[1]), dtype = float)
            y = np.array(np.tile(np.arange(shape[1]), shape[0]), dtype = float)

            if self.heisenberg == False:

                yComp  = lattice[:].flatten()
                xComp  = np.zeros( len( yComp ) )
                modelName = "ISING_2d"

                
            else:
                xComp = lattice[ :, :, 0 ].flatten()
                yComp = lattice[ :, :, 1 ].flatten()
                modelName = "HEISENBERG_2d_Mag"

        
            # plot1 = plt.figure()
            plt.quiver(x, y, xComp, yComp, headlength = 4) 
            plt.savefig(modelName + ".png")

            # plt.show( plot1 )
            
        
        elif dimension == 3:   

            fig = mlab.gcf()

            if self.heisenberg == False:
                #fix this, not working properly
                x     = np.tile(np.arange(np.shape(lattice)[0]), shape[1] * shape[2])
                y     = np.tile(np.arange(np.shape(lattice)[1]), shape[0] * shape[2])
                z     = np.tile(np.arange(np.shape(lattice)[2]), shape[1] * shape[0]) 
                xComp = lattice.flatten()
                yComp = np.zeros(len(xComp))
                zComp = np.zeros(len(xComp))
                
                mlab.quiver3d( x, y, z, xComp, yComp, zComp )
                modelName = "ISING_3d"
                
            else:   
                U  = lattice[ :, :, :, 0 ]
                V  = lattice[ :, :, :, 1 ]
                W  = lattice[ :, :, :, 2 ] 
                
                mlab.quiver3d( U, V, W )
                modelName = "HEISENBERG_3d_mag"

            mlab.options.offscreen = False
            mlab.savefig(figure    = fig, filename = modelName + '.png', size=(200,200))

    # def Ising(self):
    #     return -self.exchange
    
    # def heisenberg(self):
    #     return self.exchange*0.5   
    
# x = heisenbergModel(1000,1,100000, heisenberg = True, plotBool = (True), monteCarlo=(False),magneticField=[0,0,1]).threeDimensions([10,10,10])
# x = heisenbergModel(240, -1, 100000, plotBool=True, heisenberg=True, monteCarlo=(True),magneticField=[0,1]).twoDimensions([15, 16])

# x = heisenbergModel(1200,-1,10000, heisenberg=(False), monteCarlo=(True)).oneDimension()
# print(x)

# y = heisenbergModel(20, 1)
# print(y.oneDimension())


