#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:35:31 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.approximate_sample import LaplaceApproximate
from core.misc import load_expre, smoothing

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow


## set data and result dir
DATA_DIR = './DATA/'
RESULT_DIR = './RESULT/MAP/'
noise_level = 0.01

## domain for solving PDE
equ_nx = 100
domain_equ = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1)

## loading the truth for testing algorithms 
mesh_truth = fe.Mesh(DATA_DIR + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'CG', 1)
truth_fun = fe.Function(V_truth, DATA_DIR + 'truth_fun.xml')
truth_fun = fe.interpolate(truth_fun, domain_equ.function_space)

## setting the prior measure
prior_measure = GaussianElliptic2(
    domain=domain_equ, alpha=1, a_fun=1, theta=0.1, boundary='Neumann'
    )

# ## The following prior cannot be used currently, 
# ## Error: AttributeError: 'GaussianFiniteRank' object has no attribute 'K'
# domain_ = Domain2D(nx=50, ny=50, mesh_type='P', mesh_order=1)
# prior_measure = GaussianFiniteRank(
#     domain=domain_equ, domain_=domain_, alpha=0.1, beta=2, s=2
#     )
# prior_measure.calculate_eigensystem()
# print("Eigensystem of prior measure has been obtained!")

## loading coordinates of the measurement points
measurement_points = np.load(DATA_DIR + "measurement_points_2D.npy")

## setting the forward problem
f_expre = load_expre(DATA_DIR + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f_expre, degree=2), domain_equ.function_space)

equ_solver = EquSolver(domain_equ=domain_equ, m=truth_fun, f=f, points=measurement_points)

## load the measurement data
d = np.load(DATA_DIR + "measurement_noise_2D" + "_" + str(noise_level) + ".npy")
d_clean = np.load(DATA_DIR + "measurement_clean_2D.npy")

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

## Without a good initial value, it seems hard for us to obtain a good solution
init_fun = smoothing(truth_fun, alpha=0.1)
newton_cg.re_init(init_fun.vector()[:])

## calculate the posterior mean 
max_iter = 50
loss_pre, _, _ = model.loss()
for itr in range(max_iter):
    newton_cg.descent_direction(cg_max=50, method='cg_my')
    print(newton_cg.hessian_terminate_info)
    newton_cg.step(method='armijo', show_step=False)
    if newton_cg.converged == False:
        break
    loss, _, _ = model.loss()
    print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
    if np.abs(loss - loss_pre) < 1e-3*loss:
        print("Iteration stoped at iter = %d" % itr)
        break 
    loss_pre = loss

m_newton_cg = fe.Function(domain_equ.function_space)
m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())

## calculate the posterior variance
laplace_approximate = LaplaceApproximate(model)
laplace_approximate.calculate_eigensystem(num_eigval=30, method="double_pass")
# laplace_approximate.calculate_eigensystem(num_eigval=30, method="scipy_eigsh")
laplace_approximate.set_mean(m_newton_cg.vector()[:])

## sampling from the posterior measure 
sample_fun = fe.Function(domain_equ.function_space)
sample_fun.vector()[:] = laplace_approximate.generate_sample()

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
fig = fe.plot(truth_fun, label='truth')
plt.legend()
plt.colorbar(fig)
plt.title("Truth")
plt.subplot(1,2,2)
fig = fe.plot(sample_fun, label='sample')
plt.legend()
plt.colorbar(fig)
plt.title("One Sample")

## calculate the prior and posterior pointwise variance field
num = 10
points = np.linspace(0, 1, num).reshape(num, 1)
pointwise_variance_field_posterior = laplace_approximate.pointwise_variance_field(points, points)
pointwise_variance_field_prior = prior_measure.pointwise_variance_field(points, points)

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
plt.imshow(pointwise_variance_field_posterior)
plt.colorbar()
plt.title("Posterior")
plt.subplot(1,2,2)
plt.imshow(pointwise_variance_field_prior)
plt.colorbar()
plt.title("Prior")
plt.show()















