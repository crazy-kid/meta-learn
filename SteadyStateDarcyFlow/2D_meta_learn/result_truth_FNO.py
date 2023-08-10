#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:15:05 2022

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
from core.misc import load_expre

from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss


"""
In the following, we plot the truth parameters and the estimated mean function by 
the trained FNO2d. 
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
equ_nx = 60
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

small_n = 40 # equ_nx
domain_ = Domain2D(nx=small_n, ny=small_n, mesh_type='P', mesh_order=1)
prior = GaussianFiniteRankTorch(
    domain=domain_, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior.calculate_eigensystem()
loglam = prior.log_lam.copy()
prior.trans2torch(device=device)

nnprior_mean = FNO2d(
    modes1=12, modes2=12, width=32, coordinates=domain.mesh.coordinates()
    ).to(device)

## load results
if with_hyper_prior == True:
    dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean_prior"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    loss_list = np.load(meta_results_dir + env + str(50) + "_meta_FNO_loss_prior.npy")
    prior_log_lam = np.load(
        meta_results_dir + env + str(50) + "_meta_FNO_log_lam_prior.npy", 
        )
else:
    dir_nn = meta_results_dir + env + str(50) + "_meta_FNO_mean"
    nnprior_mean.load_state_dict(torch.load(dir_nn))
    loss_list = np.load(meta_results_dir + env + str(50) + "_meta_FNO_loss.npy")
    prior_log_lam = np.save(
        meta_results_dir + env + str(50) + "_meta_FNO_log_lam.npy", 
        )

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


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 0.4

examples = [0, 1]
X, Y = np.meshgrid(np.linspace(0, 1, equ_nx+1), np.linspace(0, 1, equ_nx+1))
fig, ax = plt.subplots(2, 2, figsize=(13, 13), squeeze=False) 
param_truth = test_model_params[examples[0]].cpu().detach().numpy()
tu = ax[0, 0].pcolor(X, Y, param_truth)
plt.colorbar(tu, ax=ax[0,0])
ax[0,0].set_title("Truth parameter")

draw_est = output_FNO[examples[0],:,:].cpu().detach().numpy()
tu = ax[0,1].pcolor(X, Y, draw_est)
plt.colorbar(tu, ax=ax[0,1])
ax[0,1].set_title("Estimated mean")

param_truth = test_model_params[examples[1]].cpu().detach().numpy()
tu = ax[1, 0].pcolor(X, Y, param_truth)
plt.colorbar(tu, ax=ax[1,0])
ax[1,0].set_title("Truth parameter")
    
draw_est = output_FNO[examples[1],:,:].cpu().detach().numpy()
tu = ax[1,1].pcolor(X, Y, draw_est)
plt.colorbar(tu, ax=ax[1,1])
ax[1,1].set_title("Estimated mean")

plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2) 













