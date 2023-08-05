#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:40:56 2022

@author: jjx323
"""

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spsl
import fenics as fe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import pickle

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from core.model import Domain1D
from core.misc import eval_min_max
from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from NN_library import FNO1d, d2fun


## load data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
env = 'complex'

def eva_mean_FNO(equ_nx, idx_p=0, idx_n=1, train_dim=200):
    domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
    d2v = fe.dof_to_vertex_map(domain.function_space)
    ## gridx contains coordinates that are match the function values obtained by fun.vector()[:]
    ## More detailed illustrations can be found in Subsections 5.4.1 to 5.4.2 of the following book:
    ##     Hans Petter Langtangen, Anders Logg, Solving PDEs in Python: The FEniCS Tutorial I. 
    gridx = domain.mesh.coordinates()[d2v]
    ## transfer numpy.arrays to torch.tensor that are used as part of the input of FNO 
    gridx_tensor = torch.tensor(gridx, dtype=torch.float32)
    
    ## load model parameters 
    with open(meta_data_dir + env + "_meta_parameters", 'rb') as f: 
        u_meta = pickle.load(f)
    num_samples = len(u_meta)
    u_meta = np.array(u_meta)
    mesh_meta = fe.Mesh(meta_data_dir + env + '_saved_mesh_meta.xml')
    V_meta = fe.FunctionSpace(mesh_meta, 'P', 1)
    u_meta_fun = fe.Function(V_meta)
    
    with open(meta_data_dir + env + "_meta_data_x_test", 'rb') as f: 
        meta_data_x_test = pickle.load(f)
    with open(meta_data_dir + env + "_meta_data_y_test", 'rb') as f: 
        meta_data_y_test = pickle.load(f)
    T, num_steps = np.load(meta_data_dir + env + "_equation_parameters_test.npy")
    num_steps = np.int64(num_steps)
    coordinates_test = meta_data_x_test[idx_p]
    equ_solver1 = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates_test]).T, m=None)
    coordinates_test = meta_data_x_test[idx_n]
    equ_solver2 = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                            points=np.array([coordinates_test]).T, m=None)
    
    Sy1 = d2fun(meta_data_y_test[idx_p], equ_solver1)
    Sy2 = d2fun(meta_data_y_test[idx_n], equ_solver2)

    ## load results of f(S;\theta)
    with_dir = meta_results_dir + env + str(train_dim) + "_meta_FNO_mean_prior"
    
    nnprior_mean1 = FNO1d(
        modes=32, width=64
        )
    nnprior_mean1.load_state_dict(torch.load(with_dir)) 
    nnprior_mean1.eval()
    
    mean_fS1_prior = fe.Function(domain.function_space)
    mean_fS1_prior.vector()[:] = np.array(nnprior_mean1(Sy1, gridx_tensor).reshape(-1).detach().numpy())
    mean_fS2_prior = fe.Function(domain.function_space)
    mean_fS2_prior.vector()[:] = np.array(nnprior_mean1(Sy2, gridx_tensor).reshape(-1).detach().numpy())

    return mean_fS1_prior, mean_fS2_prior

## 
mean1, mean2 = [], []
dims = [100, 200, 300, 400]

for NN in dims:
    m1, m2 = eva_mean_FNO(NN, idx_p=0, idx_n=1, train_dim=200)
    mean1.append(m1)
    mean2.append(m2)

## draw the mean functions with different discretized dimensions 
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)

linestyles = ["-.", "--", "-", ":"]
for itr in range(len(dims)):
    label = "Discritized dim = " + str(dims[itr])
    fe.plot(mean1[itr], linewidth=2, linestyle=linestyles[itr], label=label)
plt.legend()
plt.title("(a) Predicted mean (first branch) by FNO")

plt.subplot(1,2,2)
for itr in range(len(dims)):
    label = "Discritized dim = " + str(dims[itr])
    fe.plot(mean2[itr], linewidth=2, linestyle=linestyles[itr], label=label)
plt.legend()
plt.title("(a) Predicted mean (second branch) by FNO")

plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2)  



















