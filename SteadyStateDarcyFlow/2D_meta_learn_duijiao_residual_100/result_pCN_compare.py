#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:59:04 2022

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
from core.sample import pCN

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import Dis2Fun, UnitNormalization, GaussianFiniteRankTorch, \
    FNO2d, ForwardProcessNN, LpLoss


"""
In the following, we compare the estimated results by pCN sampling 
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


beta = 0.005
length_total = np.int64(2e5)

"""
Evaluate pCN estimates for test examples with unlearned prior measures 
"""

prior_unlearn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior_unlearn.calculate_eigensystem()

accept_rates = []

global num_

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
    
    ## define the function phi used in pCN
    def phi(u_vec):
        model.update_m(u_vec.flatten(), update_sol=True)
        return model.loss_residual()
    
    ## set the path for saving results
    samples_file_path = meta_results_dir + "pCN_unlearn/" + str(idx) + 'beta_' + str(beta) + '/samples/'
    os.makedirs(samples_file_path, exist_ok=True)
    draw_path = meta_results_dir + "pCN_unlearn/" + str(idx) + 'beta_' + str(beta) + '/draws/'
    os.makedirs(draw_path, exist_ok=True)
    
    ## set pCN 
    pcn = pCN(prior=model.prior, phi=phi, beta=beta, save_num=np.int64(1e3), path=samples_file_path)
    
    ## extract information from the chain to see how the algorithm works
    num_ = 0
    class CallBack(object):
        def __init__(self, idx=idx, num_=0, function_space=domain.function_space, 
                     save_path=draw_path, length_total=length_total, phi=phi):
            self.num_ = num_
            self.fun = fe.Function(function_space)
            # self.truth = truth
            self.save_path = save_path
            self.num_fre = 1000
            self.length_total = length_total
            self.phi = phi
            self.idx = idx
            
        def callback_fun(self, params):
            # params = [uk, iter_num, accept_rate, accept_num]
            num = params[1]
            if num % self.num_fre == 0:
                print("-"*70)
                print('unlearn, Idx = %d, Iteration number = %d/%d' % (self.idx, num, self.length_total), end='; ')
                print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
                print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
                self.num_ = params[3]
                print('Phi = %4.4f' % self.phi(params[0]))
            
                # self.fun.vector()[:] = params[0]
                # fe.plot(self.truth, label='Truth')
                # fe.plot(self.fun, label='Estimation')
                # plt.legend()
                # plt.show(block=False)
                # plt.savefig(self.save_path + 'fig' + str(num) + '.png')
                # plt.close()
     
    callback = CallBack()
    
    ## start sampling
    acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=length_total, 
                                                        callback=callback.callback_fun)
    
    accept_rates.append(acc_rate)
    
    
np.save(meta_results_dir + "pCN_unlearn/acc_rate_unlearn", accept_rates)



"""
Evaluate pCN estimates for test examples with learned prior measures 
"""

prior_learn = GaussianFiniteRank(
    domain=domain, domain_=domain_, alpha=0.1, beta=10, s=2
    )
prior_learn.calculate_eigensystem()

accept_rates = []

for idx in range(n_test):
    
    param = test_model_params[idx].cpu().detach().numpy().flatten()
    param = param[coor_transfer["d2v"]]
    prior_learn.update_mean_fun(param)
    prior_learn.set_log_lam(prior_log_lam)
    
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
    
    ## define the function phi used in pCN
    def phi(u_vec):
        model.update_m(u_vec.flatten(), update_sol=True)
        return model.loss_residual()
    
    ## set the path for saving results
    samples_file_path = meta_results_dir + "pCN_learn/" + str(idx) + 'beta_' + str(beta) + '/samples/'
    os.makedirs(samples_file_path, exist_ok=True)
    draw_path = meta_results_dir + "pCN_learn/" + str(idx) + 'beta_' + str(beta) + '/draws/'
    os.makedirs(draw_path, exist_ok=True)
    
    ## set pCN 
    pcn = pCN(prior=model.prior, phi=phi, beta=beta, save_num=np.int64(1e3), path=samples_file_path)
    
    ## extract information from the chain to see how the algorithm works
    num_ = 0
    class CallBack(object):
        def __init__(self, idx=idx, num_=0, function_space=domain.function_space, 
                     save_path=draw_path, length_total=length_total, phi=phi):
            self.num_ = num_
            self.fun = fe.Function(function_space)
            # self.truth = truth
            self.save_path = save_path
            self.num_fre = 1000
            self.length_total = length_total
            self.phi = phi
            self.idx = idx
            
        def callback_fun(self, params):
            # params = [uk, iter_num, accept_rate, accept_num]
            num = params[1]
            if num % self.num_fre == 0:
                print("-"*70)
                print('learn, Idx = %d, Iteration number = %d/%d' % (self.idx, num, self.length_total), end='; ')
                print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
                print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
                self.num_ = params[3]
                print('Phi = %4.4f' % self.phi(params[0]))
            
                # self.fun.vector()[:] = params[0]
                # fe.plot(self.truth, label='Truth')
                # fe.plot(self.fun, label='Estimation')
                # plt.legend()
                # plt.show(block=False)
                # plt.savefig(self.save_path + 'fig' + str(num) + '.png')
                # plt.close()
     
    callback = CallBack()
    
    ## start sampling
    acc_rate, samples_file_path, _ = pcn.generate_chain(length_total=length_total, 
                                                        callback=callback.callback_fun)
    
    accept_rates.append(acc_rate)
    
    
np.save(meta_results_dir + "pCN_learn/acc_rate_unlearn", accept_rates)























