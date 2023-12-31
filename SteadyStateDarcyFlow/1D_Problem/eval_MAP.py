#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:15:26 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.01

## domain for solving PDE
equ_nx = 300
domain_equ = Domain1D(n=equ_nx, mesh_type='CG', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=1, a_fun=1, theta=0.1, boundary='Neumann'
    )

# domain_ = Domain1D(n=100, mesh_type='P', mesh_order=1)
# prior_measure = GaussianFiniteRank(
#     domain=domain_equ, domain_=domain_, alpha=0.1, beta=1, s=2
#     )
# prior_measure.calculate_eigensystem()

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_1D.npy")

## setting the forward problem
f_expre = load_expre(DATA_DIR + 'f_1D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)

equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, \
                       points=np.array([measurement_points]).T,)

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_1D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_1D.npy")

## setting the noise
noise_level_ = noise_level*max(d_clean)
noise = NoiseGaussianIID(dim=len(measurement_points))
noise.set_parameters(variance=noise_level_**2)

## setting the Model
model = ModelDarcyFlow(
    d=d, domain_equ=domain_equ, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver
    )

## set optimizer NewtonCG
newton_cg = NewtonCG(model=model)
max_iter = 2000

## Without a good initial value, it seems hard for us to obtain a good solution
init_fun = smoothing(truth_fun, alpha=0.1)
newton_cg.re_init(init_fun.vector()[:])

loss_pre, _, _ = model.loss()
for itr in range(max_iter):
    # newton_cg.descent_direction(cg_max=100, method='bicgstab')
    newton_cg.descent_direction(cg_max=100, method='cg_my')
    print("CG terminate info: ", newton_cg.hessian_terminate_info)
    newton_cg.step(method='armijo', show_step=False)
    loss, _, _ = model.loss()
    print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    if newton_cg.converged == False:
        break
    if np.abs(loss - loss_pre) < 1e-3*loss:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_newton_cg = fe.Function(domain_equ.function_space)
m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
model.update_m(m_newton_cg.vector()[:], update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
fe.plot(m_newton_cg, label='estimate')
fe.plot(truth_fun, label='truth')
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 2, 2)
plt.plot(d_est, label='d_est')
plt.plot(d, label='d')
plt.legend()
plt.title('Predicted Data')


## set optimizer GradientDescent
model.update_m(fe.interpolate(fe.Constant(0.0), domain_equ.function_space).vector()[:], update_sol=True)
gradient_descent = GradientDescent(model=model)
max_iter = 2000

## Without a good initial value, it seems hard for us to obtain a good solution
init_fun = smoothing(truth_fun, alpha=0.1)
gradient_descent.re_init(init_fun.vector()[:])

loss_pre, _, _ = model.loss()
for itr in range(max_iter):
    gradient_descent.descent_direction()
    gradient_descent.step(method='armijo', show_step=False)
    if gradient_descent.converged == False:
        break
    loss, _, _ = model.loss()
    if itr % 100 == 0:
        print("iter = %4d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    if np.abs(loss - loss_pre) < 1e-3*loss:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_gradient_descent = fe.Function(domain_equ.function_space)
m_gradient_descent.vector()[:] = np.array(gradient_descent.mk.copy())
model.update_m(m_gradient_descent.vector()[:], update_sol=True)
d_est = model.S@model.p.vector()[:]

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
fe.plot(m_gradient_descent, label='estimate')
fe.plot(truth_fun, label='truth')
plt.legend()
plt.title("Estimated Function")
plt.subplot(1, 2, 2)
plt.plot(d_est, label='d_est')
plt.plot(d, label='d')
plt.legend()
plt.title('Predicted Data')
plt.show()




















