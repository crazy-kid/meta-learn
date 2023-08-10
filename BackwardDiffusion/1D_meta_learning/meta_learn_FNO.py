#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:28:17 2022

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
import copy
import pickle

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.misc import construct_measurement_matrix, trans2spnumpy, spnumpy2sptorch, \
    sptorch2spnumpy, eval_min_max
from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from BackwardDiffusion.meta_common import GaussianFiniteRank, GaussianElliptic2Learn
from BackwardDiffusion.meta_common import Gaussian1DFiniteDifference
from BackwardDiffusion.meta_common import PDEFun, PDEasNet, LossResidual, PriorFun
from BackwardDiffusion.meta_common import LossPrior, PriorFunFR
from NN_library import FNO1d, d2fun

"""
The prior mean function f(S;\theta) is assumed to be the Fourier Neural Operator with 
\theta to be the network parameters and the input is M^{-1}S.T@d. 
"""

# Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
env = 'simple'
# env = 'complex'
# with_hyper_prior = True
with_hyper_prior = False
max_iter = 1000

print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

# domain for solving PDE
equ_nx = 70
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
# d2v is used to transfer the grid coordinates.
d2v = fe.dof_to_vertex_map(domain.function_space)
# gridx contains coordinates that are match the function values obtained by fun.vector()[:]
# More detailed illustrations can be found in Subsections 5.4.1 to 5.4.2 of the following book:
# Hans Petter Langtangen, Anders Logg, Solving PDEs in Python: The FEniCS Tutorial I.
gridx = domain.mesh.coordinates()[d2v]
# transfer numpy.arrays to torch.tensor that are used as part of the input of FNO
gridx_tensor = torch.tensor(gridx, dtype=torch.float32)

noise_level = np.load(meta_data_dir + "noise_level.npy")

# load the meta_data.
# meta_data_x: coordinates of measurement points;
# a list with elements of np.arrays with different length;
# meta_data_y: the noisy datasets obtained by different parameters generated from one environment distribution
# a list with elements of np.arrays with different length;
with open(meta_data_dir + env + "_meta_data_x", 'rb') as f:
    meta_data_x = pickle.load(f)
with open(meta_data_dir + env + "_meta_data_y", 'rb') as f:
    meta_data_y = pickle.load(f)
n = len(meta_data_x)
T, num_steps = np.load(meta_data_dir + env + "_equation_parameters.npy")
num_steps = np.int64(num_steps)
coordinates = meta_data_x[0]
# in the present setting, for each parameter u, there is only one dataset
m = 1

# -----------------------------------------------------------------------
# loading the background true parameters for comparision
with open(meta_data_dir + env + "_meta_parameters", 'rb') as f:
    u_meta = pickle.load(f)
u_meta = np.array(u_meta)
mesh_meta = fe.Mesh(meta_data_dir + env + '_saved_mesh_meta.xml')
V_meta = fe.FunctionSpace(mesh_meta, 'P', 1)

if env == 'simple':
    u_meta_fun = fe.Function(V_meta)
    u_meta_fun.vector()[:] = np.array(u_meta[0])
elif env == 'complex':
    u_meta_fun1 = fe.Function(V_meta)
    u_meta_fun2 = fe.Function(V_meta)
    u_meta_fun1.vector()[:] = np.array(u_meta[0])
    u_meta_fun2.vector()[:] = np.array(u_meta[1])
else:
    raise NotImplementedError("env should be simple or complex")

# -----------------------------------------------------------------------
# constructing equ_solver
if env == 'simple':
    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                           points=np.array([coordinates]).T, m=u_meta_fun)
    sol = equ_solver.forward_solver()
    sol_fun = fe.Function(domain.function_space)
    sol_fun.vector()[:] = np.array(sol)
elif env == 'complex':
    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                           points=np.array([coordinates]).T, m=u_meta_fun1)
    sol = equ_solver.forward_solver()
    sol_fun1 = fe.Function(domain.function_space)
    sol_fun1.vector()[:] = np.array(sol)

    equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                           points=np.array([coordinates]).T, m=u_meta_fun2)
    sol = equ_solver.forward_solver()
    sol_fun2 = fe.Function(domain.function_space)
    sol_fun2.vector()[:] = np.array(sol)
else:
    raise NotImplementedError("env should be simple or complex")

# -----------------------------------------------------------------------
# construct the base prior measure that is a Gaussian measure
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
# domain_ = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
# num_gamma = equ_nx
# domain_ = Domain1D(n=60, mesh_type='P', mesh_order=1)
# num_gamma = 60
# alpha_prior, beta_prior = 0.1, 1.0
# prior = GaussianFiniteRank(
#     domain=domain, domain_=domain_, num_gamma=num_gamma,
#     alpha=alpha_prior, beta=beta_prior, s=2#, boundary="DirichletZero"
#     )
# prior.calculate_eigensystem()
# ## the mean function is not learnable, the learnable parameters are parameters in FNO
# prior.trans2learnable(mean=False, log_gamma=True)

prior = GaussianElliptic2Learn(
    domain, alpha=0.01, a_fun=fe.Constant(1.0), theta=1.0,
    mean_fun=None, tensor=False, boundary='Neumann'
)
learn_var = False
prior.trans2learnable(learn_mean=False, learn_var=learn_var)

# prior = Gaussian1DFiniteDifference(
#     nx=equ_nx+1, a_fun=beta_prior, alpha=alpha_prior
#     )
# prior.trans2learnable(learn_mean=False, learn_a=True)

# -----------------------------------------------------------------------
# construct the hyper-prior measure that is a Gaussian measure
alpha_hyper_prior, beta_hyper_prior = 0.0, 0.05
# hyper_prior_mean = GaussianFiniteRank(
#     domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_hyper_prior,
#     beta=beta_hyper_prior, s=2, boundary="DirichletZero"
#     )
# hyper_prior_mean.calculate_eigensystem()
# hyper_prior_mean.trans2learnable(mean=False, log_gamma=False)

# hyper_prior_mean = GaussianElliptic2Learn(
#     domain=domain, alpha=alpha_hyper_prior, a_fun=fe.Constant(beta_hyper_prior),
#     #boundary="Dirichlet"
#     )
# hyper_prior_mean.trans2learnable(learn_mean=False)

hyper_prior_mean = Gaussian1DFiniteDifference(
    nx=equ_nx + 1, a_fun=alpha_hyper_prior, alpha=beta_hyper_prior
)
hyper_prior_mean.trans2learnable(learn_mean=False, learn_a=False)


def make_hyper_prior_log_gamma(mean_vec, weight=0.01):
    mean_vec = copy.deepcopy(mean_vec)
    weight = weight

    def hyper_prior_log_gamma(val):
        temp = val - mean_vec
        return weight * torch.sum(temp * temp)

    return hyper_prior_log_gamma


if learn_var == True:
    hyper_prior_log_gamma = make_hyper_prior_log_gamma(prior.log_gamma_learn, weight=0.001)
# hyper_prior_log_gamma = make_hyper_prior_log_gamma(prior.a_fun_learn, weight=0.001)

# randomly select some datasets (for repeatable, we fix the selection here)
rand_idx = [0, 1, 2, 3]  # np.random.choice(n, 4)

# hyper_prior_mean = GaussianElliptic2Learn(
#     domain, alpha=1e-1, a_fun=fe.Constant(0.01), theta=1.0,
#     mean_fun=None, tensor=False, boundary='Dirichlet'
#     )
# hyper_prior_mean.trans2learnable(learn_mean=False)

# -----------------------------------------------------------------------
# Set the neural network for learning the prediction policy of the mean functions.
# The parameters are specified similar to the original paper proposed FNO.
nnprior_mean = FNO1d(
    modes=32, width=64
)

# -----------------------------------------------------------------------
# Set the noise
noise_level_ = noise_level
noise = NoiseGaussianIID(dim=len(meta_data_y[0]))
noise.set_parameters(variance=noise_level_**2)
noise.to_tensor()

loss_residual = LossResidual(noise)
# -----------------------------------------------------------------------
# transfer the PDEs as a layer of the neural network that makes the loss.backward() useable
pde_fun = PDEFun.apply

# L: the number of samples used to approximate Z_m(S, P_{S}^{\theta})
L = 10

# batch_size: only use batch_size number of datasets for each iteration
batch_size = 10

if learn_var == True:
    optimizer = torch.optim.AdamW(
        [{"params": nnprior_mean.parameters(), "lr": 0.0001},
         {"params": prior.log_gamma_learn, "lr": 0.01}],
        # lr=0.001,
        weight_decay=0.0
    )

# optimizer = torch.optim.AdamW(
#     [{"params": nnprior_mean.parameters(), "lr": 0.0001},
#       {"params": prior.a_fun_learn, "lr": 0.01}],
#     #lr=0.001,
#     weight_decay=0.0
#     )

if learn_var == False:
    optimizer = torch.optim.AdamW(
        [{"params": nnprior_mean.parameters()}],
        lr=0.0001,
        weight_decay=0.0
    )

loss_list = []

weight_of_lnZ = n / ((m + 1) * batch_size)

for kk in range(max_iter):

    optimizer.zero_grad()
    lnZ = torch.zeros(1)

    batch = np.random.choice(np.arange(n), batch_size)
    panduan = 0

    for itr in batch:
        coordinates = meta_data_x[itr]

        # -------------------------------------------------------------------
        # Since the data dimension is different for different task, we need
        # different loss functions.
        noise_level_ = noise_level
        noise = NoiseGaussianIID(dim=len(meta_data_y[itr]))
        noise.set_parameters(variance=noise_level_**2)
        noise.to_tensor()

        loss_residual = LossResidual(noise)
        # -------------------------------------------------------------------
        # for each dataset, we need to reconstruct equ_solver since the points are changed
        if panduan == 0:
            equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                                   points=np.array([coordinates]).T, m=None)
            pdeasnet = PDEasNet(pde_fun, equ_solver)
            panduan = 1
        else:
            pdeasnet.equ_solver.update_points(np.array([coordinates]).T)

        loss_res_L = torch.zeros(L)

        targets = torch.tensor(meta_data_y[itr], dtype=torch.float32)
        # M^{-1/2}S^{T}d
        Sy = d2fun(meta_data_y[itr], equ_solver)
        prior.mean_vec_learn = nnprior_mean(Sy, gridx_tensor).reshape(-1)

        for ii in range(L):
            ul = prior.generate_sample_learn()
            preds = pdeasnet(ul)
            loss_res_L[ii] = -loss_residual(preds, targets)

        lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
            - torch.log(torch.tensor(L, dtype=torch.float32))

    if with_hyper_prior == True:
        prior1 = 0
        for itr in rand_idx:
            Sy = d2fun(meta_data_y[itr], equ_solver)
            prior1 += hyper_prior_mean.evaluate_CM_inner(
                nnprior_mean(Sy, gridx_tensor).reshape(-1)
            )
        if learn_var == True:
            prior1 += hyper_prior_log_gamma(prior.log_gamma_learn)
        # prior1 += hyper_prior_log_gamma(prior.a_fun_learn)
        nlogP = -weight_of_lnZ * lnZ + prior1
        # nlogP = prior1
    else:
        nlogP = -weight_of_lnZ * lnZ

    nlogP.backward()

    if kk % 200 == 0:
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * 1.0

    optimizer.step()

    if kk % 10 == 0:
        if with_hyper_prior == True:
            print("Iter = %4d/%d, nlogP = %.4f, lnZ = %.4f, prior = %.4f" %
                  (kk, max_iter, nlogP.item(), -weight_of_lnZ * lnZ.item(), prior1.item()))
        else:
            print("Iter = %4d/%d, nlogP = %.4f" % (kk, max_iter, nlogP.item()))
        fun = fe.Function(domain.function_space)
        equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                               points=np.array([meta_data_x[0]]).T, m=None)
        targets = torch.tensor(meta_data_y[0], dtype=torch.float32)
        Sy = d2fun(meta_data_y[0], equ_solver)
        prior.mean_vec_learn = nnprior_mean(Sy, gridx_tensor).reshape(-1)
        fun.vector()[:] = np.array(prior.mean_vec_learn.detach().numpy())
        equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                               points=np.array([meta_data_x[0]]).T, m=fun)
        sol_est = equ_solver.forward_solver()
        sol_fun_est = fe.Function(domain.function_space)
        sol_fun_est.vector()[:] = np.array(sol_est)
        if env == "simple":
            plt.figure()
            min_val, max_val = eval_min_max([
                u_meta_fun.vector()[:], sol_fun.vector()[:], fun.vector()[:],
                sol_fun_est.vector()[:]
            ])
            fe.plot(u_meta_fun, label="one_meta_u")
            fe.plot(sol_fun, label="sol_meta_u")
            fe.plot(fun, label="estimated_mean")
            fe.plot(sol_fun_est, label="sol_estimated_meta")
            plt.ylim([min_val - 0.1, max_val + 0.1])
            plt.legend()
        elif env == "complex":
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            min_val, max_val = eval_min_max([
                u_meta_fun1.vector()[:], sol_fun1.vector()[:], fun.vector()[:],
                sol_fun_est.vector()[:]
            ])
            fe.plot(u_meta_fun1, label="one_meta_u")
            fe.plot(sol_fun1, label="sol_meta_u")
            fe.plot(fun, label="estimated_mean")
            fe.plot(sol_fun_est, label="sol_estimated_meta")
            plt.ylim([min_val - 0.1, max_val + 0.1])
            plt.legend()
            plt.subplot(1, 3, 2)
            fun = fe.Function(domain.function_space)
            equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                                   points=np.array([meta_data_x[1]]).T, m=None)
            targets = torch.tensor(meta_data_y[1], dtype=torch.float32)
            Sy = d2fun(meta_data_y[1], equ_solver)
            prior.mean_vec_learn = nnprior_mean(Sy, gridx_tensor).reshape(-1)
            fun.vector()[:] = np.array(prior.mean_vec_learn.detach().numpy())
            equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                                   points=np.array([meta_data_x[1]]).T, m=fun)
            sol_est = equ_solver.forward_solver()
            sol_fun_est = fe.Function(domain.function_space)
            sol_fun_est.vector()[:] = np.array(sol_est)

            min_val, max_val = eval_min_max([
                u_meta_fun2.vector()[:], sol_fun2.vector()[:], fun.vector()[:],
                sol_fun_est.vector()[:]
            ])
            fe.plot(u_meta_fun2, label="one_meta_u")
            fe.plot(sol_fun2, label="sol_meta_u")
            fe.plot(fun, label="estimated_mean")
            fe.plot(sol_fun_est, label="sol_estimated_meta")
            plt.ylim([min_val - 0.1, max_val + 0.1])
            plt.legend()
            plt.subplot(1, 3, 3)
            # plt.plot(prior.log_gamma_learn.detach().numpy(), label="Learned log_gamma")
            # plt.plot(prior.a_fun_learn.detach().numpy(), label="Learned log_gamma")
            # plt.plot(prior.log_gamma, label="Prior log_gamma")
            # plt.plot(np.log(prior.a_fun), label="Prior log_gamma")
            # plt.legend()
            if learn_var == True:
                print("learned_gamma: ", torch.exp(prior.log_gamma_learn))
        else:
            raise NotImplementedError("env should be simple or complex")

        if with_hyper_prior == True:
            plt.savefig(meta_results_dir + env + "_figs_FNO_prior/" + 'fig' + str(kk) + '.png')
        else:
            plt.savefig(meta_results_dir + env + "_figs_FNO/" + 'fig' + str(kk) + '.png')

        plt.close()

    loss_list.append(nlogP.item())


# results
if with_hyper_prior == True:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env +
        str(equ_nx) + "_meta_FNO_mean_prior"
    )
    np.save(meta_results_dir + env + str(equ_nx) +
            "_meta_FNO_loss_prior", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior",
        prior.log_gamma_learn.detach().numpy()
    )
else:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env +
        str(equ_nx) + "_meta_FNO_mean"
    )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma",
        prior.log_gamma_learn.detach().numpy()
    )

# draw
u_meta_fun = fe.Function(V_meta)
u_meta_fun.vector()[:] = np.array(u_meta[0])

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
uu = fe.Function(domain.function_space)
for itr in range(len(u_meta)):
    if itr % 2 == 0:
        u_meta_fun.vector()[:] = np.array(u_meta[itr])
        uu = fe.interpolate(u_meta_fun, domain.function_space)
        plt.plot(uu.vector()[:], alpha=0.1)

fun = fe.Function(domain.function_space)
equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                       points=np.array([meta_data_x[0]]).T, m=None)
targets = torch.tensor(meta_data_y[0], dtype=torch.float32)
Sy = d2fun(meta_data_y[0], equ_solver)
prior.mean_vec_learn = nnprior_mean(Sy, gridx_tensor).reshape(-1)
fun.vector()[:] = np.array(prior.mean_vec_learn.detach().numpy())
equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                       points=np.array([meta_data_x[0]]).T, m=fun)
sol_est = equ_solver.forward_solver()
sol_fun_est = fe.Function(domain.function_space)
sol_fun_est.vector()[:] = np.array(sol_est)

plt.plot(fun.vector()[:], label="estimated_mean")
plt.plot(sol_fun_est.vector()[:], label="sol_estimated_meta")
plt.legend()

plt.subplot(1, 2, 2)
uu = fe.Function(domain.function_space)
for itr in range(len(u_meta)):
    if itr % 2 == 1:
        u_meta_fun.vector()[:] = np.array(u_meta[itr])
        uu = fe.interpolate(u_meta_fun, domain.function_space)
        plt.plot(uu.vector()[:], alpha=0.1)

fun = fe.Function(domain.function_space)
equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                       points=np.array([meta_data_x[1]]).T, m=None)
targets = torch.tensor(meta_data_y[1], dtype=torch.float32)
Sy = d2fun(meta_data_y[1], equ_solver)
prior.mean_vec_learn = nnprior_mean(Sy, gridx_tensor).reshape(-1)
fun.vector()[:] = np.array(prior.mean_vec_learn.detach().numpy())
equ_solver = EquSolver(domain_equ=domain, T=T, num_steps=num_steps,
                       points=np.array([meta_data_x[1]]).T, m=fun)
sol_est = equ_solver.forward_solver()
sol_fun_est = fe.Function(domain.function_space)
sol_fun_est.vector()[:] = np.array(sol_est)

plt.plot(fun.vector()[:], label="estimated_mean")
plt.plot(sol_fun_est.vector()[:], label="sol_estimated_meta")
plt.legend()
