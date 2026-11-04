# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:39:52 2025

@author: Wstev
"""
# Import relevant modules
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numba import jit
import random
import math
import time
import multiprocessing
import pandas as pd
# Load the initial lattice and spin configuration
Spins = np.load("Spin80.npy")   # Initial configuration of spins; shape (N,3)
Lattice = np.load("Points80.npy")  # Lattice points
Distance = np.load("Distance80.npy")  # Pairwise distances
DistanceX = np.load("DistanceX80.npy")  # X components of distances
DistanceY = np.load("DistanceY80.npy")
indices = np.load("Indices80.npy")
with open("Neighbours80.pkl", "rb") as f:
    Neighbours = pickle.load(f)
with open("Next_Nearest_Neighbours80.pkl", "rb") as f:
    Next_Nearest_Neighbours = pickle.load(f)

# Convert the neighbor lists to NumPy arrays (slicing only first 19800 entries)
Neighbours = np.array(Neighbours[0:19800], dtype=np.int64)
Next_Nearest_Neighbours = np.array(Next_Nearest_Neighbours[0:19800], dtype=np.int64)
indices = np.array(indices, dtype=np.int64)
# Global physical parameters (set according to your system)
dipole_cutoff = 30               # cutoff distance (Angstroms)
mu_0 = 4 * math.pi * 1e-7        # Permeability of free space (H/m)
k = (6.24*10**21)*float(1*10**30) # lattice scaling constant
Kb = 1.38e-23                    # Boltzmann constant (J/K)

#---------------------------------------------------------------------
# Numba-jitted helper functions

def Randomspins(L):
    Spins = np.random.uniform(low=-1.0,high=1.0,size=(L,2))
             
    Spins /= np.linalg.norm(Spins, axis=1)[:, np.newaxis]
    return Spins
L = 1

@jit(nopython=True)
def calculate_spin(spins, Lattice):
    Number_of_Spins = len(Lattice)
    Sx = np.sum((spins[:, 0])) / Number_of_Spins
    Sy = np.sum((spins[:, 1])) / Number_of_Spins
    Sz = np.sum((spins[:, 2])) / Number_of_Spins
    Sx1 = np.sum((spins[:, 0]))
    Sy1 = np.sum((spins[:, 1]))
    Sxy = math.sqrt(Sx1**2 + Sy1**2)/Number_of_Spins
    return Sx, Sy, Sz, Sxy

@jit(nopython=True,)

def dipole_energy(spins, lattice, Distance, DistanceX, DistanceY, i, mu,indices):
    energy = 0.0
    distance = Distance[i]
    distanceX = DistanceX[i]
    distanceY = DistanceY[i]
    index = indices[i]
    S_i = spins[i]
    # Loop over all spins (excluding self)
    for j in range(len(distance)):
        if distance[j] == 0:
            continue
        if distance[j] < dipole_cutoff:
            # Build vector r_ij with zero z-component
            r_ij = np.array([distanceX[j], distanceY[j], 0.0])
            dist = distance[j]
            a = index[j]
            # (r_ij/dist) gives the unit vector, though here we only use r_ij itself.
            S_j = spins[a]
            factor = k * (mu_0 * (3 * mu)**2)
            F = factor / (4 * np.pi * dist**3)
            energy = F * ((S_i[0]*S_j[0] + S_i[1]*S_j[1] + S_i[2]*S_j[2]) - 
                          3 * (((S_i[0]*r_ij[0] + S_i[1]*r_ij[1] + S_i[2]*r_ij[2]) * 
                                (S_j[0]*r_ij[0] + S_j[1]*r_ij[1] + S_j[2]*r_ij[2])) / dist**2))
    return energy

@jit(nopython=True)
def metropolis_step(spins, lattice, Neighbours, Next_Nearest_Neighbours, Distance, DistanceX, DistanceY,
                    kbT, J, gamma1, gamma2, J2, B, D, muB,new_spin,indices,C):
    # Choose a random spin index, assuming there are 19800 spins to update
    i = np.random.randint(0, len(lattice))
    neighbors = Neighbours[i]
    next_neighbors = Next_Nearest_Neighbours[i]
    
    # Propose a new spin vector (perturb current one slightly)
    ix = new_spin[0]
    iy = new_spin[1]
    iz = spins[i, 2] + np.random.uniform(-0.3, 0.3)
    norm = math.sqrt(ix**2 + iy**2 + iz**2)
    new_spin = np.array([ix, iy, iz]) / norm
    old_spin = spins[i].copy()
    
    Sx_new, Sy_new, Sz_new = new_spin[0], new_spin[1], new_spin[2]
    delta_E = 0.0
    
    # Nearest-neighbor energy contribution
    for j in neighbors:
        delta_E += 0.5*(J * (spins[j, 0]*(Sx_new - old_spin[0]) +
                                spins[j, 1]*(Sy_new - old_spin[1]) +
                                spins[j, 2]*(Sz_new - old_spin[2])) +
                          gamma1 * (spins[j, 2]*(Sz_new - old_spin[2])))
    # Next-nearest-neighbor contribution
    for j in next_neighbors:
        delta_E +=  0.5*(J2 * (spins[j, 0]*(Sx_new - old_spin[0]) +
                                spins[j, 1]*(Sy_new - old_spin[1]) +
                                spins[j, 2]*(Sz_new - old_spin[2])) +
                          gamma2 * (spins[j, 2]*(Sz_new - old_spin[2])))
    # Field and anisotropy contribution
    delta_E += D * (Sz_new**2 - old_spin[2]**2)
    delta_E += C * (Sx_new**2 - old_spin[0]**2)
    delta_E += -(B[0]*(Sx_new - old_spin[0]) + B[1]*(Sy_new - old_spin[1]) + B[2]*(Sz_new - old_spin[2]))
    
    # Dipole-dipole contribution
    old_dipole = dipole_energy(spins, lattice, Distance, DistanceX, DistanceY, i, muB,indices)
    spins[i] = new_spin  # temporarily update to compute new dipole energy
    new_dipole = dipole_energy(spins, lattice, Distance, DistanceX, DistanceY, i, muB,indices)
    delta_E += (new_dipole - old_dipole)
    
    # Revert back then check the Metropolis acceptance criterion
    spins[i] = old_spin
    if delta_E < 0 or np.random.random() < math.exp(-delta_E/kbT):
        spins[i] = new_spin

#---------------------------------------------------------------------
# Simulation routine

def Simulation_run(spins, Lattice, Neighbours, Next_Nearest_Neighbours, Distance, DistanceX, DistanceY,
                   J, gamma1, gamma2, J2, B, D, mu_B, num_steps, T,Random_spins,indices,C):
    S_X, S_Y, S_Z, S_XY = [], [], [], []
    kbT = 1000*Kb*T/(1.6e-19)   # thermal energy (J)
    S_XY_store = []
    S_X_store = []
    S_Y_store = []
    S_Z_store = []
     
    for step in range(num_steps):
        new_spin = Random_spins[step]
        metropolis_step(spins, Lattice, Neighbours, Next_Nearest_Neighbours, Distance, DistanceX, DistanceY,
                        kbT, J, gamma1, gamma2, J2, B, D, mu_B,new_spin,indices,C)
        if step % 1000 == 0:
            Sx, Sy, Sz, Sxy = calculate_spin(spins, Lattice)
            S_X.append(Sx)
            S_Y.append(Sy)
            S_Z.append(Sz)
            S_XY.append(Sxy)
        if step % 1000000 == 0:
            print(step)
    S_XY_store.append(np.average(S_XY[int(len(S_XY)-1000):int(len(S_XY))]))
    S_Z_store.append(np.average(S_Z[int(len(S_XY)-1000):int(len(S_XY))]))
    S_X_store.append(np.average(S_X[int(len(S_XY)-1000):int(len(S_XY))]))
    S_Y_store.append(np.average(S_Y[int(len(S_XY)-1000):int(len(S_XY))]))
        
            
    return S_X_store, S_Y_store, S_Z_store, S_XY_store, T,spins

#---------------------------------------------------------------------
# Define a wrapper function that runs the simulation at a given temperature.
# This is the function we will call in paralle .
def run_simulation_for_loop(T):
    # Set parameter values for the simulation:
    J = -0.79          # Nearest-neighbor interaction (meV)
    J2 = -0.071        # Next-nearest-neighbor interaction (meVV
    gamma1 = -0.002    # Nearest-neighbor anisotropic exchange term
    gamma2 = 0.0004    # Next-nearest-neighbor anisotropic exchange term
    D = -0.007        # Anisotropy constant (meV)
    num_steps = 20000000 #carlo steps (set to a low number for testing)
    mu_B = 9.274e-24   # Bohr magneton (eV/T)
    C = 0 # in-plane anisotropy constant
    T1 = T # temperature
    B_0 = np.linspace(0,-0.5,51)
    B_0 = np.append(B_0,np.linspace(-0.5,0.5,101))
    B_0 = np.append(B_0,np.linspace(0.5,-0.5,101))
    #B_0 = np.linspace(0,-2,2)
    #B_0 = np.append(B_0,np.linspace(2,0.05,5))
    #B_0 = np.append(B_0,np.linspace(0.05,-0.05,31))
   # B_0 = np.append(B_0,np.linspace(-0.05,-2,5))
    #B_0 = np.append(B_0,np.linspace(-2,-0.05,5))
   # B_0 = np.append(B_0,np.linspace(-0.05,0.05,31))
    #B_0 = np.append(B_0,np.linspace(0.05,2,5)
    result_store = np.array([])
    k = 0
    S_XY_store = np.array([])
    S_X_store = np.array([])
    S_Y_store = np.array([])
    S_Z_store = np.array([])
    Spin_store = []
    theta = 0.524#
    # all spins startingalong z axis
    #Starting_Spins = np.load('Vortex_State_80.npy')
    Spins = np.load("Spin80.npy")
    spins_local = Spins
    #
    while k < len(B_0):
    
        B = np.array([B_0[k]*np.cos(theta), 0.0,B_0[k]*np.sin(theta)])  # Exteral magnetic field vector (meV)
    
    # Initialise random starting spins
        Random_spins = Randomspins(num_steps)
    # Make a local copy of the initial spin configuration for this run.

        if B[0] == 0:
            num_steps_0 = num_steps
            Random_spins = Randomspins(num_steps_0)
            result = Simulation_run(spins_local, Lattice, Neighbours, Next_Nearest_Neighbours,
                                    Distance, DistanceX, DistanceY,
                                    J, gamma1, gamma2, J2, B, D, mu_B,
                                    num_steps_0, T1,Random_spins,indices,C)
        else:
            
            result = Simulation_run(spins_local, Lattice, Neighbours, Next_Nearest_Neighbours,
                                    Distance, DistanceX, DistanceY,
                                    J, gamma1, gamma2, J2, B, D, mu_B,
                                    num_steps, T1,Random_spins,indices,C)
        S_X_store = np.append(S_X_store,result[0])
        S_Y_store = np.append(S_Y_store,result[1])
        S_Z_store = np.append(S_Z_store,result[2])
        S_XY_store = np.append(S_XY_store,result[3])
        Spin_store.append(spins_local.copy())
        k = k + 1
    result_store = S_X_store,S_Y_store,S_Z_store,S_XY_store,B_0,Spin_store,T
    return result_store

#---------------------------------------------------------------------
# Main block to run the simulation in parallel over a range of temperatures
if __name__ == "__main__": 
    start_time = time.time()
    print("Thanks Freddie!!!!!")
    # Define a range of temperatures to simulate (in Kelvin or any chosen unit proportional to energy)
    
    temperatures = np.array([2,4,6,8,10,12,14,20])
    
    # Use multiprocessing.Pool to run one simulation per temperature in parallel.
    with multiprocessing.Pool(processes=9) as pool:
        results = pool.map(run_simulation_for_loop, temperatures)
    
    end_time = time.time()
    with open("Simulation_results_weekend_30_Freddies_Computer.pkl","wb") as file:
        pickle.dump(results,file)
    print("Results saved")
    print(f"Parallel simulation completed in {end_time - start_time:.2f} seconds.")
# #%%

#     B_0 = np.linspace(0,-0.5,51)
#     B_0 = np.append(B_0,np.linspace(-0.5,0.5,101))
#     B_0 = np.append(B_0,np.linspace(0.5,-0.5,101))
#     B1 = B_0
#     S_XY_store_average = np.zeros(len(B_0))
#     S_X_store_average = np.zeros(len(B_0))
#     S_Y_store_average = np.zeros(len(B_0))
#     S_Z_store_average = np.zeros(len(B_0))

#     T_store = []
#     Spin_save = []
#     S_X_save = []

#     for res in results:
        
#         S_X_store,S_Y_store,S_Z_store,S_XY_store,B_0,Spin_store,T = res
        
#         plt.axvline(x=0, color='k', linestyle='--') 
#         plt.axhline(y=0, color='k', linestyle='--')
#        # if T == 6:
#         plt.plot(B_0[5:len(B_0)],S_X_store[5:len(B_0)],label = T)

#         plt.legend()
#         T_store.append(T)
#         S_X_save.append(S_X_store)
#         Spin_save.append(Spin_store)
#         S_X_store_average = S_X_store_average + S_X_store
#         S_Y_store_average = S_Y_store_average + S_Y_store
#         S_Z_store_average = S_Z_store_average + S_Z_store
#         S_XY_store_average = S_XY_store_average + S_XY_store

#     plt.title("Temperature dependent hysteresis loops ('HOPG')")
#     plt.xlabel("Field (T)") 
#     plt.ylabel("M (uB)")
#     B_0 = B1 # field value
    
    
#     R = results[0]
#     B = R[4]
#     Spin_store = R[5]
#     Field = B
#     print("Finished")
#     with open("Spin_store.pkl", "wb") as f:
#         pickle.dump(Spin_store, f)
    

   
   
   
#%%

   
   