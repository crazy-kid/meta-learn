#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:43:19 2022

@author: Junxiong Jia
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import cupy as cp
import cupyx.scipy.sparse as cpss
import cupyx.scipy.sparse.linalg as cpssl
import copy
import time

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing, print_my
from core.misc import construct_measurement_matrix, trans2spnumpy, spnumpy2sptorch, \
    sptorch2spnumpy, eval_min_max

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import GaussianElliptic2Torch, \
    GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual, PriorFun


## Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
os.makedirs(meta_results_dir, exist_ok=True)

env = 'complex'
with_hyper_prior = True
# with_hyper_prior = False
max_iter = 2000
# device = "cpu"
device = "cuda"

print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

noise_level = 0.01

model_params = np.load(meta_data_dir + "model_params.npy")
dataset_x = np.load(meta_data_dir + "dataset_x.npy")
dataset_y = np.load(meta_data_dir + "dataset_y.npy")
n = dataset_x.shape[0]
m = dataset_y.shape[1]

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
u_meta_fun1 = fe.Function(V_truth, meta_data_dir + 'truth_fun.xml')
u_meta_fun2 = fe.Function(V_truth, meta_data_dir + 'truth_fun.xml')

u_meta_fun1.vector()[:] = np.array(model_params[0,:])
u_meta_fun1 = fe.interpolate(u_meta_fun1, domain.function_space)
u_meta_fun2.vector()[:] = np.array(model_params[1,:])
u_meta_fun2 = fe.interpolate(u_meta_fun2, domain.function_space)


f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

points = dataset_x[0, :]
equ_solver = EquSolver(domain_equ=domain, m=u_meta_fun1, f=f, points=points)
sol = equ_solver.forward_solver()
sol_fun1 = fe.Function(domain.function_space)
sol_fun1.vector()[:] = np.array(sol)

points = dataset_x[1, :]
equ_solver = EquSolver(domain_equ=domain, m=u_meta_fun2, f=f, points=points)
sol = equ_solver.forward_solver()
sol_fun2 = fe.Function(domain.function_space)
sol_fun2.vector()[:] = np.array(sol)

## -----------------------------------------------------------------------
## construct the base prior measure that is a Gaussian measure 
domain_ = Domain2D(nx=30, ny=30, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain, domain_=domain_, alpha=0.1, beta=1, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)
prior.learnable_mean()
prior.learnable_loglam()

# prior = GaussianElliptic2Torch( 
#     domain=domain, alpha=0.1, a_fun=fe.Constant(1.0), theta=1.0, 
#     boundary="Neumann"
#     )
# prior.trans2torch(device=device)
# prior.learnable_mean()

## -----------------------------------------------------------------------
## construct the hyper-prior measure that is a Gaussian measure 
# alpha_hyper_prior, beta_hyper_prior = 1, 1
# small_n = 30
# domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
# hyper_prior_mean = GaussianFiniteRankTorch(
#     domain=domain, domain_=domain_, alpha=alpha_hyper_prior, 
#     beta=beta_hyper_prior, s=2
#     )
# hyper_prior_mean.calculate_eigensystem()  
# hyper_prior_mean.trans2torch(device=device)

hyper_prior_mean_ = GaussianElliptic2Torch( 
    domain=domain, alpha=0.1, a_fun=fe.Constant(0.1), theta=1.0, 
    boundary="Neumann"
    )
hyper_prior_mean_.trans2torch(device=device)
hyper_prior_mean = PriorFun.apply

def make_hyper_prior_loglam(mean_vec, weight=0.01):
    mean_vec = copy.deepcopy(mean_vec)
    weight = weight
    
    def hyper_prior_loglam(val):
        temp = val - mean_vec
        return weight*torch.sum(temp*temp)
    
    return hyper_prior_loglam
    
hyper_prior_loglam = make_hyper_prior_loglam(prior.log_lam, weight=0.01)

## -----------------------------------------------------------------------
## Set the noise 
noise_level_ = noise_level*max(dataset_y[0])
noise = NoiseGaussianIID(dim=len(dataset_y[0]))
noise.set_parameters(variance=noise_level_**2)
noise.to_torch(device=device)

loss_residual = LossResidual(noise)

## -----------------------------------------------------------------------
## transfer the PDEs as a layer of the neural network that makes the loss.backward() useable
pde_fun = PDEFun.apply 

## L: the number of samples used to approximate Z_m(S, P_{S}^{\theta}) 
L = 10

## batch_size: only use batch_size number of datasets for each iteration 
batch_size = 10

optimizer = torch.optim.AdamW(
    [{"params": prior.mean_vec_torch, "lr": 2e-2}, 
     {"params": prior.log_lam, "lr": 5e-3}],
    weight_decay=0.00
    )  

loss_list = []

weight_of_lnZ = n/((m+1)*batch_size)

for kk in range(max_iter):
    
    optimizer.zero_grad()
    lnZ = torch.zeros(1).to(device)
    
    batch = np.random.choice(np.arange(n), batch_size)
    panduan = 0
    start = time.time()
    for itr in batch:
        points = dataset_x[itr,:]
        
        ## -------------------------------------------------------------------
        ## Since the data dimension is different for different task, we need 
        ## different loss functions. 
        noise_level_ = noise_level*max(dataset_y[itr,:])
        noise = NoiseGaussianIID(dim=len(dataset_y[itr,:]))
        noise.set_parameters(variance=noise_level_**2)
        noise.to_torch(device=device)

        loss_residual = LossResidual(noise)
        ## -------------------------------------------------------------------
        ## for each dataset, we need to reconstruct equ_solver since the points are changed
        if panduan == 0:
            equ_solver = EquSolver(
                domain_equ=domain, points=points, m=fe.Constant(0.0), f=f
                )
            pdeasnet = PDEasNet(pde_fun, equ_solver)
            panduan = 1
        else:
            pdeasnet.equ_solver.update_points(points)
        
        loss_res_L = torch.zeros(L).to(device)
        
        targets = torch.tensor(dataset_y[itr], dtype=torch.float32).to(device)
        
        for ii in range(L):
            ul = prior.generate_sample()
            ## By experiments, I think the functions provided by CuPy (solving
            ## Ax=b with A is a large sparse matrix) are not efficient compared 
            ## with the cpu version given by SciPy. 
            preds = pdeasnet(ul.cpu())
            preds = preds.to(targets.device)
            val = -loss_residual(preds, targets)
            loss_res_L[ii] = val
        
        ## torch.logsumexp is used to avoid possible instability of computations
        lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                  - torch.log(torch.tensor(L, dtype=torch.float32)).to(device)
                  
    if with_hyper_prior == True:
        # prior1 = hyper_prior_mean.evaluate_CM_inner(prior.mean_vec_torch)
        prior1 = hyper_prior_mean(prior.mean_vec_torch, hyper_prior_mean_)
        prior1 += hyper_prior_loglam(prior.log_lam)
        nlogP = -weight_of_lnZ*lnZ + prior1 
    else:
        nlogP = -weight_of_lnZ*lnZ
        
    nlogP.backward()
    
    # torch.nn.utils.clip_grad_norm_(
    #     prior.mean_vec_torch, 1e10, norm_type=2.0, error_if_nonfinite=False
    #     )
    
    if kk % 100 == 0:
        for g in optimizer.param_groups:
            g["lr"] = g["lr"]*1.0
            
    optimizer.step() 
    loss_list.append(nlogP.item())
    end = time.time()
    
    if kk % 10 == 0:
        print("Iter = %4d/%d, nlogP = %.4f, term1 = %.4f, prior1 = %.4f" 
              % (kk, max_iter, nlogP.item(), (-weight_of_lnZ*lnZ).item(), \
                  prior1.item()))
        print("Time: ", end-start)
        fun = fe.Function(domain.function_space)
        fun.vector()[:] = np.array(prior.mean_vec_torch.cpu().detach().numpy())
        # equ_solver = EquSolver(domain_equ=domain, points=points, m=fun, f=f)
        # sol_est = equ_solver.forward_solver()
        # sol_fun_est = fe.Function(domain.function_space)
        # sol_fun_est.vector()[:] = np.array(sol_est)
        
        plt.figure(figsize=(12, 12)) 
        plt.subplot(2,2,1)
        fig = fe.plot(u_meta_fun1, vmin=0, vmax=3.4)
        plt.colorbar(fig)
        plt.title("Meta u")
        plt.subplot(2,2,2)
        fig = fe.plot(fun, vmin=0, vmax=3.4)
        plt.colorbar(fig)
        plt.title("Estimated Mean") 
        plt.subplot(2,2,3)
        fig = fe.plot(fun)
        plt.colorbar(fig)
        plt.title("Estimated Mean")   
        plt.subplot(2,2,4)
        plt.plot(prior.log_lam.cpu().detach().numpy(), linewidth=2, label="learned loglam")
        plt.plot(loglam, label="loglam")
        plt.legend()
        plt.title("Log-Lam")

        if with_hyper_prior == True:
            os.makedirs(meta_results_dir + env + "_figs_prior/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs_prior/" + 'fig' + str(kk) + '.png')
        else:
            os.makedirs(meta_results_dir + env + "_figs/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs/" + 'fig' + str(kk) + '.png')
        plt.close()



















