#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 19:56:56 2022

@author: Junxiong Jia
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
        GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual, PriorFun, \
        FNO2d, Dis2Fun, UnitNormalization, LpLoss


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

## Generate meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"
os.makedirs(meta_results_dir, exist_ok=True)

env = 'complex'
with_hyper_prior = True
# with_hyper_prior = False
max_iter = int(1e4)
# device = "cpu"
device = "cuda"

noise_level = 0.01

print("-------------" + str(env) + "_" + str(with_hyper_prior) + "-------------")

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
## d2v is used to transfer the grid coordinates. 
d2v = np.array(fe.dof_to_vertex_map(domain.function_space), dtype=np.int64)
v2d = np.array(fe.vertex_to_dof_map(domain.function_space), dtype=np.int64)

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

def trans_model_data(model_params, V1, V2):
    n, ll = model_params.shape
    params = np.zeros((n, V2.dim()))
    fun1 = fe.Function(V1)
    for itr in range(n):
        fun1.vector()[:] = np.array(model_params[itr, :])
        fun2 = fe.interpolate(fun1, V2)
        params[itr, :] = np.array(fun2.vector()[:])
    return np.array(params)

## load the train model, train dataset x and train dataset y 
train_model_params = np.load(meta_data_dir + "train_model_params.npy")
train_dataset_x = np.load(meta_data_dir + "train_dataset_x.npy")
train_dataset_y = np.load(meta_data_dir + "train_dataset_y.npy")
## n is the number of the training data
n = train_dataset_x.shape[0]
m = train_dataset_y.shape[1]

train_model_params = trans_model_data(
    train_model_params, V_truth, domain.function_space
    )

## Transfer discrete observation points to the functions defined on domain
dis2fun = Dis2Fun(domain=domain, points=train_dataset_x[0, :], alpha=0.01)

train_dataset_Sy = []
for itr in range(n):
    dis2fun.reset_points(train_dataset_x[itr, :])
    Sy = dis2fun(train_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    train_dataset_Sy.append(Sy)
train_dataset_Sy = np.array(train_dataset_Sy)

train_model_params = torch.tensor(train_model_params, dtype=torch.float32)
train_model_params = train_model_params[:, v2d].reshape(n, equ_nx+1, equ_nx+1)
normalize_model_params = UnitNormalization(train_model_params)
train_dataset_x = torch.tensor(train_dataset_x, dtype=torch.float32)
train_dataset_Sy = torch.tensor(train_dataset_Sy, dtype=torch.float32)
normalize_train_Sy = UnitNormalization(train_dataset_Sy)

test_model_params = np.load(meta_data_dir + "test_model_params.npy")
test_dataset_x = np.load(meta_data_dir + "test_dataset_x.npy")
test_dataset_y = np.load(meta_data_dir + "test_dataset_y.npy")
n_test = test_dataset_x.shape[0]

test_model_params = trans_model_data(
    test_model_params, V_truth, domain.function_space
    )

test_dataset_Sy = []
for itr in range(n_test):
    dis2fun.reset_points(test_dataset_x[itr, :])
    Sy = dis2fun(test_dataset_y[itr])[v2d].reshape(equ_nx+1, equ_nx+1)
    Sy = Sy.reshape(equ_nx+1, equ_nx+1)
    test_dataset_Sy.append(Sy)
test_dataset_Sy = np.array(test_dataset_Sy)  

test_model_params = torch.tensor(test_model_params, dtype=torch.float32)
test_model_params = test_model_params[:, v2d].reshape(n_test, equ_nx+1, equ_nx+1)
test_dataset_x = torch.tensor(test_dataset_x, dtype=torch.float32)
test_dataset_Sy = torch.tensor(test_dataset_Sy, dtype=torch.float32)
normalize_test_Sy = UnitNormalization(test_dataset_Sy)

# u_meta_fun1.vector()[:] = np.array(test_model_params[0,:])
# u_meta_fun1 = fe.interpolate(u_meta_fun1, domain.function_space)
# u_meta_fun2.vector()[:] = np.array(test_model_params[1,:])
# u_meta_fun2 = fe.interpolate(u_meta_fun2, domain.function_space)


f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

# points = test_dataset_x[0, :]
# equ_solver = EquSolver(domain_equ=domain, m=u_meta_fun1, f=f, points=points)
# sol = equ_solver.forward_solver()
# sol_fun1 = fe.Function(domain.function_space)
# sol_fun1.vector()[:] = np.array(sol)

# points = test_dataset_x[1, :]
# equ_solver = EquSolver(domain_equ=domain, m=u_meta_fun2, f=f, points=points)
# sol = equ_solver.forward_solver()
# sol_fun2 = fe.Function(domain.function_space)
# sol_fun2.vector()[:] = np.array(sol)

## -----------------------------------------------------------------------
## construct the base prior measure that is a Gaussian measure 
small_n = 40 # equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)
# prior.learnable_mean() 
prior.learnable_loglam()

## construct interpolation matrix
coor = domain_.mesh.coordinates()
# v2d = fe.vertex_to_dof_map(self.domain_.function_space)
d2v_ = fe.dof_to_vertex_map(domain_.function_space)
## full to small matrix
f2sM = construct_measurement_matrix(coor[d2v_], domain.function_space).todense()
f2sM = torch.tensor(f2sM, dtype=torch.float32).to(torch.device(device))

coor = domain.mesh.coordinates()
# v2d = fe.vertex_to_dof_map(self.domain.function_space)
d2v_ = fe.dof_to_vertex_map(domain.function_space)
## small to full matrix
s2fM = construct_measurement_matrix(coor[d2v_], domain_.function_space).todense()
s2fM = torch.tensor(s2fM, dtype=torch.float32).to(torch.device(device))

# prior = GaussianElliptic2Torch( 
#     domain=domain, alpha=0.1, a_fun=fe.Constant(1.0), theta=1.0, 
#     boundary="Neumann"
#     )
# prior.trans2torch(device=device)
# # prior.learnable_mean()

## -----------------------------------------------------------------------
## construct the hyper-prior measure that is a Gaussian measure 
# alpha_hyper_prior, beta_hyper_prior = 1, 1
# domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
# hyper_prior_mean = GaussianFiniteRankTorch(
#     domain=domain, domain_=domain_, alpha=alpha_hyper_prior, 
#     beta=beta_hyper_prior, s=2.5
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


## randomly select some datasets (for repeatable, we fix the selection here)
rand_idx = [0, 1, 2, 3]#np.random.choice(n, 4)

## -----------------------------------------------------------------------
## Set the neural network for learning the prediction policy of the mean functions.
## The parameters are specified similar to the original paper proposed FNO.
nnprior_mean = FNO2d(
    modes1=12, modes2=12, width=32, coordinates=domain.mesh.coordinates()
    ).to(device)

## -----------------------------------------------------------------------
## Set the noise 
noise_level_ = noise_level*max(train_dataset_y[0])
noise = NoiseGaussianIID(dim=len(train_dataset_y[0]))
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
    [{"params": nnprior_mean.parameters(), "lr": 1e-3}, 
      {"params": prior.log_lam, "lr": 1e-2}],
    weight_decay=0.00
    )  

# optimizer = torch.optim.AdamW(
#     [{"params": nnprior_mean.parameters(), "lr": 1e-4}],
#     weight_decay=0.00
#     )  

loss_list = []

weight_of_lnZ = n/((m+1)*batch_size)

normalize_model_params.to(torch.device(device))
normalize_train_Sy.to(torch.device(device))
normalize_test_Sy.to(torch.device(device))

loss_lp = LpLoss()

def eva_grad_norm(model):
    total_norm = []
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = torch.abs(p.grad.detach().data).max()
        total_norm.append(param_norm)
    total_norm = max(total_norm)
    return total_norm

start = time.time()
print("Start learning ......")
for kk in range(0, max_iter):
    
    optimizer.zero_grad()
    lnZ = torch.zeros(1).to(device)
    
    batch = np.random.choice(np.arange(n), batch_size)
    
    Sy = train_dataset_Sy[batch, :, :].to(torch.device(device))
    Sy = normalize_train_Sy.encode(Sy)
    Sy = Sy.reshape(batch_size, equ_nx+1, equ_nx+1, 1)
    output_FNO = nnprior_mean(Sy).reshape(batch_size, equ_nx+1, equ_nx+1)
    output_FNO = normalize_model_params.decode(output_FNO)
    output_FNO = (output_FNO.reshape(batch_size, -1))[:, d2v]
    
    panduan = 0
    prior0 = 0.0
    
    for idx, itr in enumerate(batch):
        points = train_dataset_x[itr,:].cpu().detach().numpy()
        
        ## -------------------------------------------------------------------
        ## Since the data dimension is different for different task, we need 
        ## different loss functions. 
        noise_level_ = noise_level*max(train_dataset_y[itr,:])
        noise = NoiseGaussianIID(dim=len(train_dataset_y[itr,:]))
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
        
        targets = torch.tensor(train_dataset_y[itr], dtype=torch.float32).to(device)
        # prior.mean_vec_torch = output_FNO[idx, :]
        prior.mean_vec_torch = torch.matmul(f2sM, output_FNO[idx, :])
        prior0 += 1e2*torch.norm(torch.matmul(s2fM, prior.mean_vec_torch) - output_FNO[idx, :])
        
        for ii in range(L):
            ul = prior.generate_sample()
            ## By experiments, I think the functions provided by CuPy (solving
            ## Ax=b with A is a large sparse matrix) are not efficient compared 
            ## with the cpu version given by SciPy. 
            ul = torch.matmul(s2fM, ul).cpu()
            preds = pdeasnet(ul)
            # preds = pdeasnet(ul.cpu())
            preds = preds.to(targets.device)
            val = -loss_residual(preds, targets)
            loss_res_L[ii] = val
        
        ## torch.logsumexp is used to avoid possible instability of computations
        lnZ = lnZ + torch.logsumexp(loss_res_L, 0)  \
                  - torch.log(torch.tensor(L, dtype=torch.float32)).to(device)
                  
    if with_hyper_prior == True:
        Sy = train_dataset_Sy[rand_idx, :, :].to(torch.device(device))
        Sy = normalize_train_Sy.encode(Sy)
        Sy = Sy.reshape(len(rand_idx), equ_nx+1, equ_nx+1, 1)
        output_FNO = nnprior_mean(Sy).reshape(len(rand_idx), equ_nx+1, equ_nx+1)
        output_FNO = normalize_model_params.decode(output_FNO)
        output_FNO = (output_FNO.reshape(len(rand_idx), -1))[:, d2v]
        
        # prior1 = hyper_prior_mean.evaluate_CM_norm_batch(output_FNO)
        # prior1 = hyper_prior_mean.evaluate_CM_inner(prior.mean_vec_torch0)
        prior1 = hyper_prior_mean(output_FNO, hyper_prior_mean_)
        
        # prior1 = 0
        # for ii in range(len(rand_idx)):
        #     prior1 += hyper_prior_mean(output_FNO[ii, :], hyper_prior_mean_)
            
        # prior1 = hyper_prior_mean(prior.mean_vec_torch, hyper_prior_mean_)
        prior1 += hyper_prior_loglam(prior.log_lam)
        nlogP = -weight_of_lnZ*lnZ + prior1 + prior0
    else:
        nlogP = -weight_of_lnZ*lnZ
    
    nn_params = [x for name, x in nnprior_mean.named_parameters()]
    nn.utils.clip_grad_norm_(nn_params, 1e4)
    nlogP.backward()
    
    # grad_norm = eva_grad_norm(nnprior_mean).cpu().detach().numpy()
            
    optimizer.step() 
    loss_list.append(nlogP.item())
    # end = time.time()
    
    if kk % 15 == 0 and kk > 1:
        # for g in optimizer.param_groups:
        #     g["lr"] = g["lr"]*0.5
        if optimizer.param_groups[0]["lr"] > 1e-5:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5
        else:
            optimizer.param_groups[0]["lr"] = 1e-5
        if optimizer.param_groups[1]["lr"] > 1e-4:
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"]*0.9
        else:
            optimizer.param_groups[1]["lr"] = 1e-4
    
    if kk % 10 == 0:
        end = time.time()
        print("-"*50)
        print("Time: %.2f, lr_u: %.6f, lr_loglam: %.6f"  %  \
              (end-start, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
              )
        start = time.time()
        
        Sy = train_dataset_Sy[batch, :, :].to(torch.device(device))
        Sy = normalize_train_Sy.encode(Sy)
        Sy = Sy.reshape(batch_size, equ_nx+1, equ_nx+1, 1)
        output_FNO = nnprior_mean(Sy).reshape(batch_size, equ_nx+1, equ_nx+1)
        output_FNO = normalize_model_params.decode(output_FNO)
        train_error = loss_lp(output_FNO.cpu(), train_model_params[batch, :, :])
        
        Sy = test_dataset_Sy.to(torch.device(device))
        n_test = Sy.shape[0]
        Sy = normalize_train_Sy.encode(Sy)
        Sy = Sy.reshape(n_test, equ_nx+1, equ_nx+1, 1)
        output_FNO = nnprior_mean(Sy).reshape(-1, equ_nx+1, equ_nx+1)
        output_FNO = normalize_model_params.decode(output_FNO)
        test_error = loss_lp(output_FNO.cpu(), test_model_params)
        
        print_str = "Iter = %4d/%d, nlogP = %.4f, term1 = %.4f, prior1 = %.4f," 
        print_str += "train_error = %.4f, test_error = %.4f" 
        print(print_str  % (kk, max_iter, nlogP.item(), (-weight_of_lnZ*lnZ).item(), \
                            prior1.item(), train_error.item(), test_error.item()))
        
        draw_est = output_FNO[0,:,:].cpu().detach().numpy()
        draw_truth = test_model_params[0, :, :]
        # fun = fe.Function(domain.function_space)
        # fun.vector()[:] = np.array()
        # equ_solver = EquSolver(domain_equ=domain, points=points, m=fun, f=f)
        # sol_est = equ_solver.forward_solver()
        # sol_fun_est = fe.Function(domain.function_space)
        # sol_fun_est.vector()[:] = np.array(sol_est)
        
        plt.figure(figsize=(17, 5)) 
        plt.subplot(1,3,1)
        # fig = fe.plot(u_meta_fun1, vmin=0, vmax=3.4)
        fig = plt.imshow(draw_truth)
        plt.colorbar(fig)
        plt.title("Meta u")
        plt.subplot(1,3,2)
        # fig = fe.plot(fun)
        fig = plt.imshow(draw_est)
        plt.colorbar(fig)
        plt.title("Estimated Mean")   
        plt.subplot(1,3,3)
        plt.plot(prior.log_lam.cpu().detach().numpy(), linewidth=3, label="learned loglam")
        plt.plot(loglam, linewidth=1, label="loglam")
        plt.legend()
        plt.title("Log-Lam")

        if with_hyper_prior == True:
            os.makedirs(meta_results_dir + env + "_figs_prior/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs_prior/" + 'fig' + str(kk) + '.png')
        else:
            os.makedirs(meta_results_dir + env + "_figs/", exist_ok=True)
            plt.savefig(meta_results_dir + env + "_figs/" + 'fig' + str(kk) + '.png')
        plt.close()


## save results
if with_hyper_prior == True:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss_prior", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_lam_prior", 
        prior.log_lam.cpu().detach().numpy()
        )
else:
    torch.save(
        nnprior_mean.state_dict(), meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"
        )
    np.save(meta_results_dir + env + str(equ_nx) + "_meta_FNO_loss", loss_list)
    np.save(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_lam", 
        prior.log_lam.cpu().detach().numpy()
        )



























