#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:34:27 2022

@author: Junxiong Jia
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
from core.optimizer import GradientDescent, NewtonCG
from core.noise import NoiseGaussianIID
from core.approximate_sample import LaplaceApproximate

from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from BackwardDiffusion.meta_common import GaussianFiniteRank
from NN_library import FNO1d, d2fun


## set data and result dir
data_dir = './DATA/'
# results_dir = './RESULTS/pCN/'
meta_results_dir = './RESULTS/'
env = "complex"
noise_level = np.load(data_dir  + "noise_level.npy")

## domain for solving PDE
equ_nx = 70
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
num_gamma = 60
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

def relative_error(m1, m2, domain):
    m1 = fe.interpolate(m1, domain.function_space)
    m2 = fe.interpolate(m2, domain.function_space) 
    fenzi = fe.assemble(fe.inner(m1-m2, m1-m2)*fe.dx)
    fenmu = fe.assemble(fe.inner(m2, m2)*fe.dx)
    return fenzi/fenmu

error_posterior_mean = []
error_unlearned = []
for idx_ in range(20):

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
    idx_p = 2*idx_
    u_meta_fun_test1.vector()[:] = np.array(u_meta_test[idx_p])
    u_meta_fun_test2 = fe.Function(V_meta)
    idx_n = 2*idx_ + 1
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
    equ_solver1 = EquSolver(
        domain_equ=domain, T=T, num_steps=num_steps, 
        points=np.array([coordinates_test]).T, m=u_meta_fun_test1 ## test fun1
        )
    coordinates_test = meta_data_x_test[idx_n]
    equ_solver2 = EquSolver(
        domain_equ=domain, T=T, num_steps=num_steps,
        points=np.array([coordinates_test]).T, m=u_meta_fun_test2 ## test fun2 
        )
    
    ## idx_p, idx_n indicate different branches.
    ## Transfer measured data for the two branches into functions. 
    Sy1 = d2fun(meta_data_y_test[idx_p], equ_solver1)
    Sy2 = d2fun(meta_data_y_test[idx_n], equ_solver2)
    
    # ## load results of f(S;\theta) obtained without hyperprior
    # without_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"
    
    # nnprior_mean0 = FNO1d(
    #     modes=32, width=64
    #     )   
    # nnprior_mean0.load_state_dict(torch.load(without_dir))
    # nnprior_mean0.eval()
    
    # mean_fS1 = fe.Function(domain.function_space)
    # mean_fS1.vector()[:] = np.array(nnprior_mean0(Sy1, gridx_tensor).reshape(-1).detach().numpy())
    # mean_fS2 = fe.Function(domain.function_space)
    # mean_fS2.vector()[:] = np.array(nnprior_mean0(Sy2, gridx_tensor).reshape(-1).detach().numpy())
    
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
    ## 1. The prior measure without learning, set it as the initial prior measure 
    ##    in the learning stage. 
    
    ## alpha_prior and beta_prior are set as in the training stage
    alpha_prior = 0.1
    beta_prior = 1
    boundary_condition = "Neumann"
    ## prior $\mathcal{N}(0,\mathcal{C}_0)$
    prior_measure = GaussianFiniteRank(
        domain=domain, domain_=domain_, num_gamma=equ_nx, alpha=alpha_prior, 
        beta=beta_prior, s=2  ## set it as in the training stage
        )
    prior_measure.calculate_eigensystem()  
    # prior_measure = GaussianElliptic2(
    #     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
    #     boundary=boundary_condition 
    #     )
    
    ## 2. The prior measure with learned mean and variance by MLL(maximum likelihood) 
    ##    algorithm that without hyper-prior.
    
    # ## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS1
    # prior_measure_fS1 = GaussianFiniteRank(
    #     domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    #     s=2, mean=mean_fS1
    #     )
    # prior_measure_fS1.calculate_eigensystem()  
    # log_gamma_fS1 = np.load(meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma.npy")
    # prior_measure_fS1.log_gamma[:len(log_gamma_fS1)] = log_gamma_fS1
    # # prior_measure_fS1 = GaussianElliptic2(
    # #     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
    # #     boundary=boundary_condition, mean_fun=mean_fS1
    # #     )
    
    # ## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS2
    # prior_measure_fS2 = GaussianFiniteRank(
    #     domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
    #     s=2, mean=mean_fS2
    #     )
    # prior_measure_fS2.calculate_eigensystem()  
    # log_gamma_fS2 = np.load(meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma.npy")
    # prior_measure_fS2.log_gamma[:len(log_gamma_fS2)] = log_gamma_fS2
    # # prior_measure_fS2 = GaussianElliptic2(
    # #     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
    # #     boundary=boundary_condition, mean_fun=mean_fS2
    # #     )
    
    ## 3. The prior measure with learned mean and variance by MAP with hyper-prior.
    
    ## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS1_prior
    prior_measure_fS1_prior = GaussianFiniteRank(
        domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
        s=2, mean=mean_fS1_prior  ## set mean function
        )
    prior_measure_fS1_prior.calculate_eigensystem()  
    log_gamma_fS1_prior = np.load(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior.npy"
        )
    ## set the learned variance
    # prior_measure_fS1_prior.log_gamma[:len(log_gamma_fS1_prior)] = log_gamma_fS1_prior
    # prior_measure_fS1_prior = GaussianElliptic2(
    #     domain=domain, alpha=alpha_prior, a_fun=a_fun_prior, theta=beta_prior, 
    #     boundary=boundary_condition, mean_fun=mean_fS1_prior
    #     )
    
    ## prior $\mathcal{N}(f(S;\theta),\mathcal{C}_0)$ with f(S;\theta) = mean_fS2_prior
    prior_measure_fS2_prior = GaussianFiniteRank(
        domain=domain, domain_=domain_, num_gamma=num_gamma, alpha=alpha_prior, beta=beta_prior, 
        s=2, mean=mean_fS2_prior  ## set mean function
        )
    prior_measure_fS2_prior.calculate_eigensystem()  
    log_gamma_fS2_prior = np.load(
        meta_results_dir + env + str(equ_nx) + "_meta_FNO_log_gamma_prior.npy"
        )
    ## set the learned variance
    # prior_measure_fS2_prior.log_gamma[:len(log_gamma_fS2_prior)] = log_gamma_fS2_prior
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
    
    ## model2 with prior measure $\mathcal{N}(0, \mathcal{C}_0)$ and data d2
    model2 = ModelBackwarDiffusion(
        d=d2, domain_equ=domain, prior=prior_measure, 
        noise=noise, equ_solver=equ_solver2
        )
    
    # ## model_fS1: d1, equ_solver1, prior_measure_fS1
    # model_fS1 = ModelBackwarDiffusion(
    #     d=d1, domain_equ=domain, prior=prior_measure_fS1, 
    #     noise=noise, equ_solver=equ_solver1
    #     )
    
    # ## model_fS2: d2, equ_solver2, prior_measure_fS2
    # model_fS2 = ModelBackwarDiffusion(
    #     d=d2, domain_equ=domain, prior=prior_measure_fS2, 
    #     noise=noise, equ_solver=equ_solver2
    #     )
    
    ## model_fS1_prior: d1, equ_solver1, prior_measure_fS1_prior
    model_fS1_prior = ModelBackwarDiffusion(
        d=d1, domain_equ=domain, prior=prior_measure_fS1_prior, 
        noise=noise, equ_solver=equ_solver1
        )
    
    ## model_fS2_prior: d2, equ_solver2, prior_measure_fS2_prior
    model_fS2_prior = ModelBackwarDiffusion(
        d=d2, domain_equ=domain, prior=prior_measure_fS2_prior, 
        noise=noise, equ_solver=equ_solver2
        )
    
    """
    Calculate the posterior mean (same as the MAP in this case)
    """
    def calculate_MAP(model):
        ## set optimizer NewtonCG
        # ff = fe.Function(model.domain_equ.function_space)
        # ff.vector()[:] = 0.0 
        # # ff = fe.interpolate(u_meta_fun_test1, model.domain_equ.function_space)
        # newton_cg = NewtonCG(model=model, mk=ff.vector()[:])
        
        newton_cg = NewtonCG(model=model)
        
        ## calculate the posterior mean 
        max_iter = 10
        loss_pre, _, _ = model.loss()
        for itr in range(max_iter):
            # newton_cg.descent_direction(cg_max=1000, method="cg_my")
            newton_cg.descent_direction(cg_max=1000, method="bicgstab")
            newton_cg.step(method='armijo', show_step=False)
            loss, _, _ = model.loss()
            print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if newton_cg.converged == False:
                break
            if np.abs(loss - loss_pre) < 1e-3*loss:
                print("Iteration stoped at iter = %d" % itr)
                break 
            loss_pre = loss
        mk = newton_cg.mk.copy()
            
        # gradient_descent = GradientDescent(model=model)
        # max_iter = 200
        # loss_pre, _, _ = model.loss()
        # for itr in range(max_iter):
        #     gradient_descent.descent_direction()
        #     gradient_descent.step(method='armijo', show_step=False)
        #     if gradient_descent.converged == False:
        #         break
        #     loss, _, _ = model.loss()
        #     if itr % 100 == 0:
        #         print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        #     if np.abs(loss - loss_pre) < 1e-5*loss:
        #         print("Iteration stoped at iter = %d" % itr)
        #         break 
        #     loss_pre = loss
        # mk = gradient_descent.mk.copy()
    
        return np.array(mk)         
    
    mean_unlearn_1 = fe.Function(domain.function_space)
    mean_unlearn_1.vector()[:] = np.array(calculate_MAP(model1))
    
        
    mean_learn_prior_1 = fe.Function(domain.function_space)
    mean_learn_prior_1.vector()[:] = np.array(calculate_MAP(model_fS1_prior)) 
    
    err = relative_error(mean_learn_prior_1, u_meta_fun_test1, domain) 
    error_posterior_mean.append(err)
    
    err = relative_error(mean_unlearn_1, u_meta_fun_test1, domain)
    error_unlearned.append(err)
    
    ## ------------------------------------------------------------------
    mean_unlearn_2 = fe.Function(domain.function_space)
    mean_unlearn_2.vector()[:] = np.array(calculate_MAP(model2))
    
        
    mean_learn_prior_2 = fe.Function(domain.function_space)
    mean_learn_prior_2.vector()[:] = np.array(calculate_MAP(model_fS2_prior)) 
    
    err = relative_error(mean_learn_prior_2, u_meta_fun_test2, domain) 
    error_posterior_mean.append(err)
    
    err = relative_error(mean_unlearn_2, u_meta_fun_test2, domain)
    error_unlearned.append(err)
    

print("unlearned error: %.6f" % (np.mean(error_unlearned)))
print("posterior error: %.6f" % (np.mean(error_posterior_mean)))

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
fe.plot(u_meta_fun_test1, label="truth")
fe.plot(mean_unlearn_1, label="MAP unlearn")
fe.plot(mean_learn_prior_1, label="MAP estimate")
fun = fe.Function(domain.function_space)
fun.vector()[:] = np.array(model_fS1_prior.prior.mean_vec)
fe.plot(fun, label="prior mean")
plt.legend()
plt.subplot(1,2,2)
ff = fe.interpolate(u_meta_fun_test1, domain.function_space)
sol = equ_solver1.forward_solver(m_vec=ff.vector()[:])
dd1 = equ_solver1.S@sol
plt.plot(dd1, label="simulated data")
sol = equ_solver1.forward_solver(m_vec=mean_learn_prior_1.vector()[:])
dd2 = equ_solver1.S@sol
plt.plot(dd2, label="data posterior mean")
sol = equ_solver1.forward_solver(m_vec=model_fS1_prior.prior.mean_vec)
dd3 = equ_solver1.S@sol
plt.plot(dd3, label="data prior mean")
plt.legend()
        
# fun = fe.Function(domain.function_space)
# fun.vector()[:] = np.array(model_fS1_prior.prior.mean_vec)
# err1 = relative_error(fun, u_meta_fun_test1, domain)
# err2 = relative_error(mean_learn_prior_1, u_meta_fun_test1, domain)   
# print("Error (prior mean): %.6f" % err1)
# print("Error (posterior mean): %.6f" % err2)    
    
# print("Unlearned error: %.6f" % relative_error(mean_unlearn_1, u_meta_fun_test1, domain)) 
    
    
    
    
    
    
    
    
    
    
    
    















