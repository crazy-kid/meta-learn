#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:51:48 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import cupy as cp
import cupyx.scipy.sparse as cpss
import cupyx.scipy.sparse.linalg as cpssl

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
result_dir = './RESULT/MAP/'

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

f = load_expre(data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

equ_solver = EquSolver(domain_equ=domain, m=fe.Constant(0.0), f=f, points=points)

pdefun = PDEFun.apply
pdeasnet = PDEasNet(pdefun, equ_solver)

## setting the noise
noise_level_ = noise_level*max(d_noisy)
noise = NoiseGaussianIID(dim=len(points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelDarcyFlow(
    d=d_noisy, domain_equ=domain, prior=gaussian_measure, 
    noise=noise, equ_solver=equ_solver
    )

## set optimizer NewtonCG
newton_cg = NewtonCG(model=model)
max_iter = 200

## Without a good initial value, it seems hard for us to obtain a good solution
init_fun = smoothing(truth_fun, alpha=0.1)
newton_cg.re_init(init_fun.vector()[:])

loss_pre = model.loss()[0]
for itr in range(max_iter):
    newton_cg.descent_direction(cg_max=30, method='cg_my')
    # newton_cg.descent_direction(cg_max=30, method='bicgstab')
    print(newton_cg.hessian_terminate_info)
    newton_cg.step(method='armijo', show_step=False)
    if newton_cg.converged == False:
        break
    loss, loss_res, loss_reg = model.loss()[0:3]
    print("iter = %2d/%d, loss = %.4f, loss_res = %.4f, loss_reg = %.4f" 
          % (itr+1, max_iter, loss, loss_res, loss_reg))
    if np.abs(loss - loss_pre) < 1e-3*loss:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_newton_cg = fe.Function(domain.function_space)
m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
model.update_m(m_newton_cg.vector()[:], update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
fig = fe.plot(m_newton_cg, label='estimate')
plt.colorbar(fig)
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 3, 2)
fig = fe.plot(truth_fun, label='truth')
plt.colorbar(fig)
plt.title("Truth")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(d_est, label='d_est')
plt.plot(d_noisy, label='d')
plt.legend()
plt.title('Predicted Data')


## set optimizer GradientDescent
model.update_m(
    fe.interpolate(fe.Constant(0.0), domain.function_space).vector()[:], 
    update_sol=True
    )
gradient_descent = GradientDescent(model=model)
max_iter = 2000

## Without a good initial value, it seems hard for us to obtain a good solution
init_fun = smoothing(truth_fun, alpha=0.1)
gradient_descent.re_init(init_fun.vector()[:])

loss_pre = model.loss()[0]
for itr in range(max_iter):
    gradient_descent.descent_direction()
    gradient_descent.step(method='armijo', show_step=False)
    if gradient_descent.converged == False:
        break
    loss, loss_res, loss_reg = model.loss()[0:3]
    print("iter = %2d/%d, loss = %.4f, loss_res = %.4f, loss_reg = %.4f" 
          % (itr+1, max_iter, loss, loss_res, loss_reg))
    if np.abs(loss - loss_pre) < 1e-5:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_gradient_descent = fe.Function(domain.function_space)
m_gradient_descent.vector()[:] = np.array(gradient_descent.mk.copy())
model.update_m(m_gradient_descent.vector()[:], update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
fig = fe.plot(m_newton_cg, label='estimate')
plt.colorbar(fig)
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 3, 2)
fig = fe.plot(truth_fun, label='truth')
plt.colorbar(fig)
plt.title("Truth")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(d_est, label='d_est')
plt.plot(d_noisy, label='d')
plt.legend()
plt.title('Predicted Data')
plt.show()
























