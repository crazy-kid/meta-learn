#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:14:48 2022

@author: Junxiong Jia
"""

"""
Basic settings: $\Omega:=[0,1]^2 \subset\mathbb{R}^2$. The model parameters are 
    complex version with two branches. 

We compare the computational results by pCN algorithm when the prior are chosen: 
    (1) The Gaussian measure $\mathcal{N}(0, \mathcal{C}_0)$, 
        where $\mathcal{C}_0 := (Id - \Delta)^{-1}$;
    (2) The learned Gaussian measure $\mathcal{N}(f(S;\theta), \mathcal{C}_0)$,
        where $f(S;\theta)$ is obtained by MLL (maximum likelihood) method;
    (3) The learned Gaussian measure $\mathcal{N}(f(S;\theta), \mathcal{C}_0)$,
        where $f(S;\theta)$ is obtained by MAP (maximum a posteriori) estimate. 
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import pickle
import copy
import torch

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain1D
from core.probability import GaussianElliptic2
from core.sample import pCN
from core.noise import NoiseGaussianIID

from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from BackwardDiffusion.meta_common import GaussianFiniteRank
from NN_library import FNO1d, d2fun


## set data and result dir
data_dir = './DATA/'
results_dir = './RESULTS/pCN/'
meta_results_dir = './RESULTS/'
env = "complex"
noise_level = 0.05

## set the step length of MCMC
beta = 0.01
## set the total number of the sampling
length_total = 1e5

## domain for solving PDE
equ_nx = 200
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
num_gamma = 100
domain_ = Domain1D(n=num_gamma, mesh_type='P', mesh_order=1)
## save the mesh information 
os.makedirs(data_dir, exist_ok=True)
file_mesh = fe.File(data_dir + env + '_saved_mesh_meta_pCN.xml')
file_mesh << domain.function_space.mesh()

d2v = fe.dof_to_vertex_map(domain.function_space)
## gridx contains coordinates that are match the function values obtained by fun.vector()[:]
## More detailed illustrations can be found in Subsections 5.4.1 to 5.4.2 of the following book:
##     Hans Petter Langtangen, Anders Logg, Solving PDEs in Python: The FEniCS Tutorial I. 
gridx = domain.mesh.coordinates()[d2v]
## transfer numpy.arrays to torch.tensor that are used as part of the input of FNO 
gridx_tensor = torch.tensor(gridx, dtype=torch.float32)

"""
load model parameters; load test samples 
"""
## u_meta_fun_test1: first branch of the random model parameters 
## u_meta_fun_test2: second branch of the random model parameters 
with open(data_dir + env + "_meta_parameters_test", 'rb') as f: 
    u_meta_test = pickle.load(f)
u_meta_test = np.array(u_meta_test)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
u_meta_fun_test1 = fe.Function(V_meta)
idx_p = 0
u_meta_fun_test1.vector()[:] = np.array(u_meta_test[idx_p])
u_meta_fun_test2 = fe.Function(V_meta)
idx_n = 1
u_meta_fun_test2.vector()[:] = np.array(u_meta_test[idx_n])

## load the test data pairs 
with open(data_dir + env + "_meta_data_x_test", 'rb') as f: 
    meta_data_x_test = pickle.load(f)
with open(data_dir + env + "_meta_data_y_test", 'rb') as f: 
    meta_data_y_test = pickle.load(f)
T, num_steps = np.load(data_dir + env + "_equation_parameters_test.npy")
num_steps = np.int64(num_steps)

## construct different equ_solvers for different model parameters that with 
## different measurement data 
coordinates_test = meta_data_x_test[idx_p]
equ_solver1 = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                        points=np.array([coordinates_test]).T, m=u_meta_fun_test1)
coordinates_test = meta_data_x_test[idx_n]
equ_solver2 = EquSolver(domain_equ=domain, T=T, num_steps=num_steps, \
                        points=np.array([coordinates_test]).T, m=u_meta_fun_test1)

## load results of f(S;\theta) obtained without hyperprior
without_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"

nnprior_mean0 = FNO1d(
    modes=32, width=64
    )   
nnprior_mean0.load_state_dict(torch.load(without_dir))
nnprior_mean0.eval()

Sy1 = d2fun(meta_data_y_test[idx_p], equ_solver1)
mean_fS1 = fe.Function(domain.function_space)
mean_fS1.vector()[:] = np.array(nnprior_mean0(Sy1, gridx_tensor).reshape(-1).detach().numpy())
Sy2 = d2fun(meta_data_y_test[idx_n], equ_solver2)
mean_fS2 = fe.Function(domain.function_space)
mean_fS2.vector()[:] = np.array(nnprior_mean0(Sy2, gridx_tensor).reshape(-1).detach().numpy())

## load results of f(S;\theta) obtained with hyperprior
with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"

nnprior_mean1 = FNO1d(
    modes=32, width=64
    )
nnprior_mean1.load_state_dict(torch.load(with_dir)) 
nnprior_mean1.eval()

mean_fS1_prior = fe.Function(domain.function_space)
mean_fS1_prior.vector()[:] = np.array(
    nnprior_mean1(Sy1, gridx_tensor).reshape(-1).detach().numpy()
    )
mean_fS2_prior = fe.Function(domain.function_space)
mean_fS2_prior.vector()[:] = np.array(
    nnprior_mean1(Sy2, gridx_tensor).reshape(-1).detach().numpy()
    )

"""
Construct different priors for testing 
"""
alpha_prior = 0.1
# a_fun_prior = 1.0
beta_prior = 1.0
boundary_condition = "Neumann"
## prior $\mathcal{N}(0,\mathcal{C}_0)$
prior_measure = GaussianFiniteRank(
    domain=domain, domain_=domain, num_gamma=equ_nx, alpha=alpha_prior, 
    beta=beta_prior, s=2
    )
prior_measure.calculate_eigensystem()  
# prior_measure = GaussianElliptic2(
#     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
#     boundary=boundary_condition 
#     )

## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS1
prior_measure_fS1 = GaussianFiniteRank(
    domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    s=2, mean=mean_fS1
    )
prior_measure_fS1.calculate_eigensystem()  
log_gamma_fS1 = np.load(meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma.npy")
prior_measure_fS1.log_gamma[:len(log_gamma_fS1)] = log_gamma_fS1
# prior_measure_fS1 = GaussianElliptic2(
#     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
#     boundary=boundary_condition, mean_fun=mean_fS1
#     )

## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS2
prior_measure_fS2 = GaussianFiniteRank(
    domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    s=2, mean=mean_fS2
    )
prior_measure_fS2.calculate_eigensystem()  
log_gamma_fS2 = np.load(meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma.npy")
prior_measure_fS2.log_gamma[:len(log_gamma_fS2)] = log_gamma_fS2
# prior_measure_fS2 = GaussianElliptic2(
#     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
#     boundary=boundary_condition, mean_fun=mean_fS2
#     )

## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS1_prior
prior_measure_fS1_prior = GaussianFiniteRank(
    domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    s=2, mean=mean_fS1_prior
    )
prior_measure_fS1_prior.calculate_eigensystem()  
log_gamma_fS1_prior = np.load(
    meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior.npy"
    )
prior_measure_fS1_prior.log_gamma[:len(log_gamma_fS1_prior)] = log_gamma_fS1_prior
# prior_measure_fS1_prior = GaussianElliptic2(
#     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
#     boundary=boundary_condition, mean_fun=mean_fS1_prior
#     )

## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS2_prior
prior_measure_fS2_prior = GaussianFiniteRank(
    domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    s=2, mean=mean_fS2_prior
    )
prior_measure_fS2_prior.calculate_eigensystem()  
log_gamma_fS2_prior = np.load(
    meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior.npy"
    )
prior_measure_fS2_prior.log_gamma[:len(log_gamma_fS2_prior)] = log_gamma_fS2_prior
# prior_measure_fS2_prior = GaussianElliptic2(
#     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
#     boundary=boundary_condition, mean_fun=mean_fS2_prior
#     )

"""
There will be two test datasets corresponding to two branches of the random model parameters
"""
d1 = meta_data_y_test[idx_p] ## data of the model parameter that is above the x-axis
d2 = meta_data_y_test[idx_n] ## data of the model parameter that is below the x-axis


## Set the noise 
noise_level_ = noise_level
noise = NoiseGaussianIID(dim=len(d1))
noise.set_parameters(variance=noise_level_**2)

"""
There will be two models corresponding to two branches of the random model parameters
"""
## model1 with prior measure $\mathcal{N}(0, \mathcal{C}_0)$ and data d1
model1 = ModelBackwarDiffusion(
    d=d1, domain_equ=domain, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver1
    )
## define the function phi used in pCN
def phi1(u_vec):
    model1.update_m(u_vec.flatten(), update_sol=True)
    return model1.loss_residual()
## set the path for saving results
samples_file_path1 = results_dir + '1_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path1, exist_ok=True)
draw_path1 = results_dir + '1_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path1, exist_ok=True)

## model2 with prior measure $\mathcal{N}(0, \mathcal{C}_0)$ and data d2
model2 = ModelBackwarDiffusion(
    d=d2, domain_equ=domain, prior=prior_measure, 
    noise=noise, equ_solver=equ_solver2
    )
## define the function phi used in pCN
def phi2(u_vec):
    model2.update_m(u_vec.flatten(), update_sol=True)
    return model2.loss_residual()
## set the path for saving results
samples_file_path2 = results_dir + '2_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path2, exist_ok=True)
draw_path2 = results_dir + '2_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path2, exist_ok=True)

## model_fS1: d1, equ_solver1, prior_measure_fS1
model_fS1 = ModelBackwarDiffusion(
    d=d1, domain_equ=domain, prior=prior_measure_fS1, 
    noise=noise, equ_solver=equ_solver1
    )
## define the function phi used in pCN
def phi_fS1(u_vec):
    model_fS1.update_m(u_vec.flatten(), update_sol=True)
    return model_fS1.loss_residual()
## set the path for saving results
samples_file_path_fS1 = results_dir + 'fS1_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS1, exist_ok=True)
draw_path_fS1 = results_dir + 'fS1_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS1, exist_ok=True)

## model_fS2: d2, equ_solver2, prior_measure_fS2
model_fS2 = ModelBackwarDiffusion(
    d=d2, domain_equ=domain, prior=prior_measure_fS2, 
    noise=noise, equ_solver=equ_solver2
    )
## define the function phi used in pCN
def phi_fS2(u_vec):
    model_fS2.update_m(u_vec.flatten(), update_sol=True)
    return model_fS2.loss_residual()
## set the path for saving results
samples_file_path_fS2 = results_dir + 'fS2_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS2, exist_ok=True)
draw_path_fS2 = results_dir + 'fS2_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS2, exist_ok=True)

## model_fS1_prior: d1, equ_solver1, prior_measure_fS1_prior
model_fS1_prior = ModelBackwarDiffusion(
    d=d1, domain_equ=domain, prior=prior_measure_fS1_prior, 
    noise=noise, equ_solver=equ_solver1
    )
## define the function phi used in pCN
def phi_fS1_prior(u_vec):
    model_fS1_prior.update_m(u_vec.flatten(), update_sol=True)
    return model_fS1_prior.loss_residual()
## set the path for saving results
samples_file_path_fS1_prior = results_dir + 'fS1_prior_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS1_prior, exist_ok=True)
draw_path_fS1_prior = results_dir + 'fS1_prior_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS1_prior, exist_ok=True)

## model_fS2_prior: d2, equ_solver2, prior_measure_fS2_prior
model_fS2_prior = ModelBackwarDiffusion(
    d=d2, domain_equ=domain, prior=prior_measure_fS2_prior, 
    noise=noise, equ_solver=equ_solver2
    )
## define the function phi used in pCN
def phi_fS2_prior(u_vec):
    model_fS2_prior.update_m(u_vec.flatten(), update_sol=True)
    return model_fS2_prior.loss_residual()
## set the path for saving results
samples_file_path_fS2_prior = results_dir + 'fS2_prior_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS2_prior, exist_ok=True)
draw_path_fS2_prior = results_dir + 'fS2_prior_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS2_prior, exist_ok=True)

"""
We need to define six different pCN objects
"""
pcn1 = pCN(
    prior=model1.prior, phi=phi1, beta=beta, save_num=np.int64(1e3), 
    path=samples_file_path1
    )
pcn2 = pCN(
    prior=model2.prior, phi=phi2, beta=beta, save_num=np.int64(1e3), 
    path=samples_file_path2
    )
pcn_fS1 = pCN(
    prior=model_fS1.prior, phi=phi_fS1, beta=beta, save_num=np.int64(1e3), 
    path=samples_file_path_fS1
    )
pcn_fS2 = pCN(
    prior=model_fS2.prior, phi=phi_fS2, beta=beta, save_num=np.int64(1e3), 
    path=samples_file_path_fS2
    )
pcn_fS1_prior = pCN(
    prior=model_fS1_prior.prior, phi=phi_fS1_prior, beta=beta, 
    save_num=np.int64(1e3), path=samples_file_path_fS1_prior
    )
pcn_fS2_prior = pCN(
    prior=model_fS2_prior.prior, phi=phi_fS2_prior, beta=beta, 
    save_num=np.int64(1e3), path=samples_file_path_fS2_prior
    )

"""
We also need to define six different callback functions 
"""
## extract information from the chain to see how the algorithm works
class CallBack(object):
    def __init__(self, num_=0, function_space=domain.function_space, truth=u_meta_fun_test1,
                 save_path=draw_path1, length_total=length_total, phi=phi1):
        self.num_ = num_
        self.fun = fe.Function(function_space)
        self.truth = truth
        self.save_path = save_path
        self.num_fre = 5000
        self.length_total = length_total
        self.phi = phi
        self.acc_data = []
        
    def callback_fun(self, params):
        # params = [uk, iter_num, accept_rate, accept_num]
        num = params[1]
        if num % self.num_fre == 0:
            print("-"*70)
            print('Iteration number = %d/%d' % (num, self.length_total), end='; ')
            print('Accept rate = %4.4f percent' % (params[2]*100), end='; ')
            print('Accept num = %d/%d' % (params[3] - self.num_, self.num_fre), end='; ')
            self.num_ = params[3]
            print('Phi = %4.4f' % self.phi(params[0]))
            
            self.acc_data.append([params[1], params[2], params[3]])
            
            self.fun.vector()[:] = params[0]
            fe.plot(self.truth, label='Truth')
            fe.plot(self.fun, label='Estimation')
            plt.legend()
            # plt.show(block=False)
            plt.savefig(self.save_path + 'fig' + str(num) + '.png')
            plt.close()


callback1 = CallBack(
    truth=u_meta_fun_test1, save_path=draw_path1, phi=phi1
    )
callback2 = CallBack(
    truth=u_meta_fun_test2, save_path=draw_path2, phi=phi2
    )
callback_fS1 = CallBack(
    truth=u_meta_fun_test1, save_path=draw_path_fS1, phi=phi_fS1
    )
callback_fS2 = CallBack(
    truth=u_meta_fun_test2, save_path=draw_path_fS2, phi=phi_fS2
    )
callback_fS1_prior = CallBack(
    truth=u_meta_fun_test1, save_path=draw_path_fS1_prior, phi=phi_fS1_prior
    )
callback_fS2_prior = CallBack(
    truth=u_meta_fun_test2, save_path=draw_path_fS2_prior, phi=phi_fS2_prior
    )


"""
Run the pCN algorithm with six different settings 
"""
## start sampling
print("------------------sampling case 1------------------")
acc_rate1, samples_file_path1, _ = pcn1.generate_chain(
    length_total=length_total, callback=callback1.callback_fun
    )
acc_data1 = copy.deepcopy(callback1.acc_data)
np.save(samples_file_path1[:-8], acc_data1)
## start sampling
print("------------------sampling case 2------------------")
acc_rate2, samples_file_path2, _ = pcn2.generate_chain(
    length_total=length_total, callback=callback2.callback_fun
    )
acc_data2 = copy.deepcopy(callback2.acc_data)
np.save(samples_file_path2[:-8], acc_data2)
## start sampling
print("------------------sampling case fS1------------------")
acc_rate_fS1, samples_file_path_fS1, _ = pcn_fS1.generate_chain(
    length_total=length_total, callback=callback_fS1.callback_fun
    )
acc_data_fS1 = copy.deepcopy(callback_fS1.acc_data)
np.save(samples_file_path_fS1[:-8], acc_data_fS1)
## start sampling
print("------------------sampling case fS2------------------")
acc_rate_fS2, samples_file_path_fS2, _ = pcn_fS2.generate_chain(
    length_total=length_total, callback=callback_fS2.callback_fun
    )
acc_data_fS2 = copy.deepcopy(callback_fS2.acc_data)
np.save(samples_file_path_fS2[:-8], acc_data_fS2)
## start sampling
print("------------------sampling case fS1_prior------------------")
acc_rate_fS1_prior, samples_file_path_fS1_prior, _ = pcn_fS1_prior.generate_chain(
    length_total=length_total, callback=callback_fS1_prior.callback_fun
    )
acc_data_fS1_prior = copy.deepcopy(callback_fS1_prior.acc_data)
np.save(samples_file_path_fS1_prior[:-8], acc_data_fS1_prior)
## start sampling
print("------------------sampling case fS2_prior------------------")
acc_rate_fS2_prior, samples_file_path_fS2_prior, _ = pcn_fS2_prior.generate_chain(
    length_total=length_total, callback=callback_fS2_prior.callback_fun
    )
acc_data_fS2_prior = copy.deepcopy(callback_fS2_prior.acc_data)
np.save(samples_file_path_fS2_prior[:-8], acc_data_fS2_prior)










