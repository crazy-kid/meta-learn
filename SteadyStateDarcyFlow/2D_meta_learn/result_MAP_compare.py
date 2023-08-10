#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 19:19:20 2022

@author: jjx323
"""

import numpy as np 
import fenics as fe
import torch 
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss


"""
In the following, we compare the estimated MAP results by Newton-CG optimization 
algorithm with different prior measures.
"""

"""
Preparations for construct appropriate prior measures 
"""
## dir meta data
meta_data_dir = "./DATA/"
meta_results_dir = "./RESULTS/"

env = 'complex'
with_hyper_prior = True
# with_hyper_prior = False
# max_iter = int(5e3)
device = "cpu"
# device = "cuda"

noise_level = 0.01

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(meta_data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)

## domain for solving PDE
equ_nx = 50
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)
## d2v is used to transfer the grid coordinates. 
d2v = np.array(fe.dof_to_vertex_map(domain.function_space), dtype=np.int64)
v2d = np.array(fe.vertex_to_dof_map(domain.function_space), dtype=np.int64)
coor_transfer = {"d2v": d2v, "v2d": v2d}

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

## Transfer discrete observation points to the functions defined on domain
dis2fun = Dis2Fun(domain=domain, points=test_dataset_x[0, :], alpha=0.01)

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

f = load_expre(meta_data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

small_n = equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.01, beta=100, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)

nnprior_mean = FNO2d(
    modes1=15, modes2=15, width=7, coordinates=domain.mesh.coordinates()
    ).to(device)

## load results
if with_hyper_prior == True:
    dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean_prior"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    loss_list = np.load(meta_results_dir + env + str(50) + "_meta_FNO_loss_prior.npy")
    # prior_log_lam = np.load(
    #     meta_results_dir + env + str(50) + "_meta_FNO_log_lam_prior.npy", 
    #     )
else:
    dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    loss_list = np.load(meta_results_dir + env + str(50) + "_meta_FNO_loss.npy")
    # prior_log_lam = np.save(
    #     meta_results_dir + env + str(50) + "_meta_FNO_log_lam.npy", 
    #     )

forward_nn = ForwardProcessNN(
    nn_model=nnprior_mean, coor_transfer=coor_transfer, 
    normalize_data=normalize_train_Sy, normalize_param=normalize_model_params, 
    device=device
    )

loss_lp = LpLoss()

Sy = test_dataset_Sy.to(torch.device(device))
output_FNO = forward_nn(Sy)[:, coor_transfer["v2d"]].reshape(-1, equ_nx+1, equ_nx+1)
test_error = loss_lp(output_FNO.cpu(), test_model_params)
test_error = test_error.cpu().detach().numpy()


"""
Evaluate MAP estimates for test examples with unlearned prior measures 
"""

prior_unlearn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.01, beta=100, s=2
    )
prior_unlearn.calculate_eigensystem()

estimates_unlearn = []
final_error_unlearn = []

for idx in range(n_test):
    
    equ_solver = EquSolver(
        domain_equ=domain, m=fe.Function(domain.function_space), f=f, 
        points=test_dataset_x[idx,:].cpu().detach().numpy()
        )
    d_noisy = test_dataset_y[idx, :]
    
    noise_level_ = noise_level*max(test_dataset_y[idx,:])
    noise = NoiseGaussianIID(dim=len(test_dataset_y[idx,:]))
    noise.set_parameters(variance=noise_level_**2)
    
    ## setting the Model
    model = ModelDarcyFlow(
        d=d_noisy, domain_equ=domain, prior=prior_unlearn, 
        noise=noise, equ_solver=equ_solver
        )
    
    ## set optimizer NewtonCG
    newton_cg = NewtonCG(model=model)
    max_iter = 200
    
    newton_cg.re_init(prior_unlearn.mean_vec)
    
    loss_pre = model.loss()[0]
    for itr in range(max_iter):
        newton_cg.descent_direction(cg_max=30, method='cg_my')
        # newton_cg.descent_direction(cg_max=30, method='bicgstab')
        # print(newton_cg.hessian_terminate_info)
        newton_cg.step(method='armijo', show_step=False)
        if newton_cg.converged == False:
            break
        loss = model.loss()[0]
        # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        if np.abs(loss - loss_pre) < 1e-3*loss:
            # print("Iteration stoped at iter = %d" % itr)
            break 
        loss_pre = loss
        
    # m_newton_cg = fe.Function(domain.function_space)
    # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())
    
    estimates_unlearn.append(np.array(newton_cg.mk.copy()))
    final_error_unlearn.append(loss)
    print("unlearn idx: ", idx)


np.save(meta_results_dir + "estimates_unlearn_MAP", (estimates_unlearn, final_error_unlearn))


"""
Evaluate MAP estimates for test examples with learned prior measures 
"""

prior_learn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.01, beta=100, s=2
    )
prior_learn.calculate_eigensystem()

estimates_learn = []
final_error_learn = []

for idx in range(n_test):

    param = test_model_params[idx].cpu().detach().numpy().flatten()
    param = param[coor_transfer["d2v"]]
    prior_learn.update_mean_fun(param)
    # prior_learn.set_log_lam(prior_log_lam)
    
    equ_solver = EquSolver(
        domain_equ=domain, m=fe.Function(domain.function_space), f=f, 
        points=test_dataset_x[idx,:].cpu().detach().numpy()
        )
    d_noisy = test_dataset_y[idx, :]
    
    noise_level_ = noise_level*max(test_dataset_y[idx,:])
    noise = NoiseGaussianIID(dim=len(test_dataset_y[idx,:]))
    noise.set_parameters(variance=noise_level_**2)
    
    ## setting the Model
    model = ModelDarcyFlow(
        d=d_noisy, domain_equ=domain, prior=prior_learn, 
        noise=noise, equ_solver=equ_solver
        )
    
    ## set optimizer NewtonCG
    newton_cg = NewtonCG(model=model)
    max_iter = 200
    
    newton_cg.re_init(prior_learn.mean_vec)
    
    loss_pre = model.loss()[0]
    for itr in range(max_iter):
        newton_cg.descent_direction(cg_max=30, method='cg_my')
        # newton_cg.descent_direction(cg_max=30, method='bicgstab')
        # print(newton_cg.hessian_terminate_info)
        newton_cg.step(method='armijo', show_step=False)
        if newton_cg.converged == False:
            break
        loss = model.loss()[0]
        # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        if np.abs(loss - loss_pre) < 1e-3*loss:
            # print("Iteration stoped at iter = %d" % itr)
            break 
        loss_pre = loss
    
    # m_newton_cg = fe.Function(domain.function_space)
    # m_newton_cg.vector()[:] = np.array(newton_cg.mk.copy())

    estimates_learn.append(np.array(newton_cg.mk.copy()))
    final_error_learn.append(loss)
    print("learn idx: ", idx)
        

np.save(meta_results_dir + "estimates_learn_MAP", (estimates_learn, final_error_learn))



























