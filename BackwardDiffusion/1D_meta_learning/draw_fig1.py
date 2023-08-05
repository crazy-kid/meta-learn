#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 17:37:30 2022

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
env = 'simple'

equ_nx = 200
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

## load results of f(\theta)
without_dir = meta_results_dir + env + str(equ_nx) + "_meta_mean.npy"
mean_f = fe.Function(domain.function_space)
mean_f.vector()[:] = np.array(np.load(without_dir))

with_dir = meta_results_dir + env + str(equ_nx) + "_meta_mean_prior.npy"
mean_f_prior = fe.Function(domain.function_space)
mean_f_prior.vector()[:] = np.array(np.load(with_dir))

## load test sample
with open(meta_data_dir + env + "_meta_parameters_test", 'rb') as f: 
    u_meta_test = pickle.load(f)
u_meta_test = np.array(u_meta)
mesh_meta_test = fe.Mesh(meta_data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
u_meta_fun_test = fe.Function(V_meta)
u_meta_fun_test.vector()[:] = np.array(u_meta_test[0])

with open(meta_data_dir + env + "_meta_data_x_test", 'rb') as f: 
    meta_data_x_test = pickle.load(f)
with open(meta_data_dir + env + "_meta_data_y_test", 'rb') as f: 
    meta_data_y_test = pickle.load(f)
T, num_steps = np.load(meta_data_dir + env + "_equation_parameters_test.npy")
num_steps = np.int64(num_steps)
coordinates_test = meta_data_x_test[0]
equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                        points=np.array([coordinates_test]).T, m=u_meta_fun_test)

## load results of f(S;\theta)
without_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"

nnprior_mean0 = FNO1d(
    modes=32, width=64
    )
nnprior_mean0.load_state_dict(torch.load(without_dir)) 
nnprior_mean0.eval()

Sy = d2fun(meta_data_y_test[0], equ_solver)
mean_fS = fe.Function(domain.function_space)
mean_fS.vector()[:] = np.array(nnprior_mean0(Sy, gridx_tensor).reshape(-1).detach().numpy())

## load results of f(S;\theta)
with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"

nnprior_mean1 = FNO1d(
    modes=32, width=64
    )
nnprior_mean1.load_state_dict(torch.load(with_dir)) 
nnprior_mean1.eval()

mean_fS_prior = fe.Function(domain.function_space)
mean_fS_prior.vector()[:] = np.array(nnprior_mean1(Sy, gridx_tensor).reshape(-1).detach().numpy())

## draw figures
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
for itr in range(num_samples):
    u_meta_fun.vector()[:] = np.array(u_meta[itr])
    fe.plot(u_meta_fun)
plt.ylim([-4.5, 8.5])
plt.title("(a) Environment Samples")

plt.subplot(1, 3, 2)
fe.plot(mean_f, linestyle="-.", linewidth=1.5, label="Estimate without hyperprior") 
u_meta_fun_test.vector()[:] = np.array(u_meta_test[0])
fe.plot(mean_f_prior, linewidth=2.0, color="red", label="Estimate with hyperprior")
min_val, max_val = eval_min_max([
    u_meta_fun_test.vector()[:], mean_f.vector()[:], mean_f_prior.vector()[:]
    ])
plt.ylim([min_val-0.5, max_val+0.5])
plt.legend(loc="lower right")
plt.title("(b) Estimates With Data-Independent Prior")

plt.subplot(1, 3, 3)
fe.plot(mean_fS, alpha=0.8, linestyle="-.", linewidth=1.5, label="Estimate without hyperprior") 
fe.plot(u_meta_fun_test, linestyle="--", linewidth=2.0, color="blue", label="True model parameter")
fe.plot(mean_fS_prior, linewidth=2.0, color="red", label="Estimate with hyperprior")
min_val, max_val = eval_min_max([
    u_meta_fun_test.vector()[:], mean_fS.vector()[:150], mean_fS_prior.vector()[:]
    ])
plt.ylim([min_val-0.5, max_val+0.5])
plt.legend(loc="upper right")
plt.title("(c) Estimates With Data-Dependent Prior") 
plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2)   












