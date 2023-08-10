#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:26:09 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import cupy as cp
import cupyx.scipy.sparse as cpss
import cupyx.scipy.sparse.linalg as cpssl
import time

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing, print_my

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import GaussianElliptic2Torch, \
    GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual


## set data and result dir
data_dir = './DATA/'

device = "cuda"
# device = "cpu"

noise_level = 0.01

## domain for solving PDE
equ_nx = 100
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

model_params = np.load(data_dir + "model_params.npy")
dataset_x = np.load(data_dir + "dataset_x.npy")
dataset_y = np.load(data_dir + "dataset_y.npy")

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, data_dir + 'truth_fun.xml')

truth_fun.vector()[:] = np.array(model_params[0,:])
truth_fun = fe.interpolate(truth_fun, domain.function_space)

points = dataset_x[0, :]
d_noisy = dataset_y[0, :]

## define the prior measure 
domain_ = Domain2D(nx=50, ny=50, mesh_type='P', mesh_order=1)
gaussian_measure = GaussianFiniteRankTorch(
    domain=domain, domain_=domain_, alpha=0.01, beta=1, s=2
    )
gaussian_measure.calculate_eigensystem()
gaussian_measure.trans2torch(device=device)

f = load_expre(data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

equ_solver = EquSolver(domain_equ=domain, m=truth_fun, f=f, points=points)

###############################################################################
start = time.time()
sol_cpu = equ_solver.forward_solver()
end = time.time()
print(end - start)

start = time.time()
equ_solver.to(device="cuda")
sol_gpu = equ_solver.forward_solver()
end = time.time()
print(end - start)

err = np.linalg.norm(sol_gpu.get() - sol_cpu)/np.linalg.norm(sol_cpu)*100
print("error: ", err)

###############################################################################
# data = equ_solver.S@sol_gpu
# data = data.get()

# start = time.time()
# equ_solver.to(device="cpu")
# sol_adjoint_cpu = equ_solver.adjoint_solver(data)
# end = time.time()
# print(end - start)

# start = time.time()
# equ_solver.to(device="cuda")
# sol_adjoint_gpu = equ_solver.adjoint_solver(cp.asarray(data))
# end = time.time()
# print(end - start)

# err = np.linalg.norm(sol_adjoint_cpu - sol_adjoint_gpu.get())/np.linalg.norm(sol_adjoint_cpu)*100
# print("error: ", err)



























