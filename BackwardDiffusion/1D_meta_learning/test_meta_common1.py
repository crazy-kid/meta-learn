#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:33:26 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import torch

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID

from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from BackwardDiffusion.meta_common import GaussianFiniteRank, PDEFun, PDEasNet, \
                    LossResidual, PriorFun, LossPrior, PriorFunFR

"""
The aim of the program: 
    Test the functions of evaluating gradients are works well written in meta_common.
"""

prior_fun = PriorFun.apply 
pde_fun = PDEFun.apply    

## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.01

## domain for solving PDE
equ_nx = 200
domain_equ = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=1, a_fun=1, theta=0.1, boundary='Dirichlet'
    )

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")

## setting the forward problem
T = 0.1
num_steps = 10
equ_solver = EquSolver(domain_equ=domain_equ, T=T, num_steps=num_steps, \
                        points=np.array([measurement_points]).T, m=truth_fun)

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")

## setting the noise
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelBackwarDiffusion(
    d=d, domain_equ=domain_equ, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver
    )

## ---------------------------------------------------------------------------

pdeasnet = PDEasNet(pde_fun, model.equ_solver)
criterion1 = LossResidual(model.noise)
criterion2 = LossPrior(prior_fun, model.prior)

dim = model.M.shape[0]
u = torch.zeros(dim, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([u], lr=0.001)

target = torch.tensor(model.d, dtype=torch.float32, requires_grad=False)

max_iter = 10000
for itr in range(max_iter):
    optimizer.zero_grad()
    preds = pdeasnet(u)
    loss1 = criterion1(preds, target) 
    loss2 = criterion2(u)
    loss = loss1 + loss2 
    loss.backward() 
    optimizer.step() 
    
    if itr % 100 == 0:
        print("Iter = %d/%d, loss1 = %.4f, loss2 = %.4f, loss = %.4f" 
              % (itr, max_iter, loss1.item(), loss2.item(), loss.item()))
    
m_vec = np.array(u.detach())
noise.to_numpy()
fun = fe.Function(model.domain_equ.function_space)
fun.vector()[:] = m_vec

model.update_m(m_vec, update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
fe.plot(truth_fun, label='truth')
fe.plot(fun, label='estimate')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(d_est, label='d_est')
plt.plot(d, label='d')
plt.legend()
plt.title('Gradient Descent')
plt.show()
    