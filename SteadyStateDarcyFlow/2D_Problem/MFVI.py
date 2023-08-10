#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:21:27 2022

@author: Junxiong Jia
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.approximate_sample import rMAP
from core.misc import smoothing, load_expre

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






























