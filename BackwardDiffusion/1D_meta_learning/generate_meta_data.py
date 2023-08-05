#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:30:41 2022

@author: jjx323
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import pickle

import fenics as fe
import dolfin as dl

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.misc import save_expre

from BackwardDiffusion.common import EquSolver

test = True
# test = False
DATA_DIR = './DATA/'
# env = "simple"
env = "complex"

## domain for solving PDE
equ_nx = 600
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

noise_level = 0.05
np.save(DATA_DIR + "noise_level", noise_level)
## Environments
T = 0.02
num_steps = 20
n = 200
meta_data = []
coordinates_list = []
u_list = []

def generate_data(u=None, itr=0, T=0.1, num_steps=20, env='simple'):
    beta = np.random.normal(0.5, 0.5)
    a = np.random.uniform(10, 15) 
    b = np.random.normal(0, 0.1)
    c = np.random.normal(4, 1)
    
    if u is None:
        if env == 'simple':
            # expression = "(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*sin(3.14*x[0])"
            expression = "(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*exp(-30*pow(x[0]-0.5, 2))"
            # expression = "(beta*x[0]*5 + a*sin(2*(x[0]*5 - b)) + c)*exp(-30*pow(x[0]-0.5, 2))"
        elif env == 'complex':
            if itr % 2 == 0:
                # expression = "(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*sin(3.14*x[0])"
                expression = "(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*exp(-30*pow(x[0]-0.5, 2))"
            else:
                # expression = "-(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*sin(3.14*x[0])"
                expression = "-(beta*x[0]*5 + a*sin(1.5*(x[0]*5 - b)) + c)*exp(-30*pow(x[0]-0.5, 2))"

        u_expre = fe.Expression(expression, degree=3, 
                                beta=beta, a=a, b=b, c=c) 
        u = fe.interpolate(u_expre, domain.function_space)
    else:
        u = fe.interpolate(u, domain.function_space)
    
    # fe.plot(u)
    # plt.title("u" + str(itr+1))
    num_points = 100 #np.random.randint(10, 100)
    coordinates = np.random.uniform(0, 1, (num_points,))
    # coordinates = np.linspace(0, 1, 20)

    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates]).T, m=u)
    d_clean = equ_solver.S@equ_solver.forward_solver()
    
    d = d_clean + noise_level*np.random.normal(0, 1, (len(d_clean),))
    
    return coordinates, d, d_clean, u


for itr in range(n):
    coordinates, d, _, u = generate_data(itr=itr, T=T, num_steps=num_steps, env=env)
    coordinates_list.append(coordinates)
    meta_data.append(d)
    u_list.append(u)
    
    
plt.figure()
min_val, max_val = 0, 0
for itr in range(n):
    uu = u_list[itr]
    if itr == 0:
        min_val = min(uu.vector()[:])
        max_val = max(uu.vector()[:])
    else:
        min_val = min(min_val, min(uu.vector()[:]))
        max_val = max(max_val, max(uu.vector()[:]))
    fe.plot(uu)
    plt.ylim([min_val, max_val])

u_vectors = []
for itr in range(n):
    u_vectors.append(u_list[itr].vector()[:])
    
## save the mesh information 
os.makedirs(DATA_DIR, exist_ok=True)

if test == False:
    file2 = fe.File(DATA_DIR + env + '_saved_mesh_meta.xml')
    file2 << domain.function_space.mesh()
    
    with open(DATA_DIR + env + '_meta_parameters', "wb") as f:
        pickle.dump(u_vectors, f)
    
    with open(DATA_DIR + env + '_meta_data_x', "wb") as f:
        pickle.dump(coordinates_list, f)
        
    with open(DATA_DIR + env + '_meta_data_y', "wb") as f:
        pickle.dump(meta_data, f)
        
    equ_params = [T, num_steps]
    np.save(DATA_DIR + env + "_equation_parameters", equ_params)
elif test == True:
    file2 = fe.File(DATA_DIR + env + '_saved_mesh_meta_test.xml')
    file2 << domain.function_space.mesh()
    
    with open(DATA_DIR + env + '_meta_parameters_test', "wb") as f:
        pickle.dump(u_vectors, f)
    
    with open(DATA_DIR + env + '_meta_data_x_test', "wb") as f:
        pickle.dump(coordinates_list, f)
        
    with open(DATA_DIR + env + '_meta_data_y_test', "wb") as f:
        pickle.dump(meta_data, f)
        
    equ_params = [T, num_steps]
    np.save(DATA_DIR + env + "_equation_parameters_test", equ_params)
else:
    raise NotImplementedError("test should be True or False")

# with open(DATA_DIR + 'meta_parameters', 'rb') as f: 
#     mylist = pickle.load(f)







