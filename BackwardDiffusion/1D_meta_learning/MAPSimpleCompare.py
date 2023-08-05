#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:15:37 2023

@author: ubuntu
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
env = "simple"
noise_level = np.load(data_dir  + "noise_level.npy")

## domain for solving PDE
equ_nx = 70
domain = Domain1D(n=equ_nx, mesh_type='P', mesh_order=1)
# num_gamma = 60
# domain_ = Domain1D(n=num_gamma, mesh_type='P', mesh_order=1)
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

def relative_error(m1, m2, domain, err_type="L2"):
    if err_type == "L2":
        m1 = fe.interpolate(m1, domain.function_space)
        m2 = fe.interpolate(m2, domain.function_space) 
        fenzi = fe.assemble(fe.inner(m1-m2, m1-m2)*fe.dx)
        fenmu = fe.assemble(fe.inner(m2, m2)*fe.dx)
        return fenzi/fenmu
    elif err_type == "Max":
        m1 = fe.interpolate(m1, domain.function_space)
        m2 = fe.interpolate(m2, domain.function_space) 
        vertex_values_m1 = m1.vector()[:]
        vertex_values_m2 = m2.vector()[:]
        fenzi = np.max(np.abs(vertex_values_m1 - vertex_values_m2))
        fenmu = np.max(np.abs(vertex_values_m2))
        return fenzi/fenmu

error_posterior_mean = []

for idx_ in range(50):

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
    u_meta_fun_test = fe.Function(V_meta)
    u_meta_fun_test.vector()[:] = np.array(u_meta_test[idx_])
    
    ## load the test data pairs 
    with open(data_dir + env + "_meta_data_x_test", 'rb') as f: 
        meta_data_x_test = pickle.load(f)
    with open(data_dir + env + "_meta_data_y_test", 'rb') as f: 
        meta_data_y_test = pickle.load(f)
    T, num_steps = np.load(data_dir + env + "_equation_parameters_test.npy")
    num_steps = np.int64(num_steps)
    
    ## construct different equ_solvers for different model parameters that with 
    ## different measurement data 
    coordinates_test = meta_data_x_test[idx_]
    equ_solver = EquSolver(
        domain_equ=domain, T=T, num_steps=num_steps, 
        points=np.array([coordinates_test]).T, m=u_meta_fun_test ## test fun1
        )
    
    ## idx_p, idx_n indicate different branches.
    ## Transfer measured data for the two branches into functions. 
    Sy = d2fun(meta_data_y_test[idx_], equ_solver)
    
    ## load results of f(S;\theta) obtained with hyperprior
    with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean_prior"
    
    nnprior_mean_MAP = FNO1d(
        modes=32, width=64
        )
    nnprior_mean_MAP.load_state_dict(torch.load(with_dir)) 
    nnprior_mean_MAP.eval()
    
    mean_fS_MAP = fe.Function(domain.function_space)
    mean_fS_MAP.vector()[:] = np.array(
        nnprior_mean_MAP(Sy, gridx_tensor).reshape(-1).detach().numpy()
        )
    
    with_dir = meta_results_dir + env + str(equ_nx) + "_meta_FNO_mean"
    
    nnprior_mean_MLL = FNO1d(
        modes=32, width=64
        )
    nnprior_mean_MLL.load_state_dict(torch.load(with_dir)) 
    nnprior_mean_MLL.eval()
    
    mean_fS_MLL = fe.Function(domain.function_space)
    mean_fS_MLL.vector()[:] = np.array(
        nnprior_mean_MLL(Sy, gridx_tensor).reshape(-1).detach().numpy()
        )
        
    """
    Construct different priors for testing 
    """
    ## 1. The prior measure without learning, set it as the initial prior measure 
    ##    in the learning stage. 
    
    ## alpha_prior and beta_prior are set as in the training stage
    alpha_prior = 0.01
    aa = 1.0
    # beta_prior = 1
    boundary_condition = "Neumann"
    ## prior $\mathcal{N}(0,\mathcal{C}_0)$
    # prior_measure = GaussianFiniteRank(
    #     domain=domain, domain_=domain_, num_gamma=equ_nx, alpha=alpha_prior, 
    #     beta=beta_prior, s=2  ## set it as in the training stage
    #     )
    # prior_measure.calculate_eigensystem()  
    prior_measure = GaussianElliptic2(
        domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0, 
        boundary=boundary_condition 
        )
    
    ## 2. The prior measure with learned mean and variance by MLL(maximum likelihood) 
    ##    algorithm that without hyper-prior.
    temp = np.load(meta_results_dir + env + str(equ_nx) + "_meta_mean_prior.npy")
    mean_fS = fe.Function(domain.function_space)
    mean_fS.vector()[:] = np.array(temp)
    prior_measure_MAP = GaussianElliptic2(
        domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0, 
        boundary=boundary_condition, mean_fun=mean_fS
        )
    
    temp = np.load(meta_results_dir + env + str(equ_nx) + "_meta_mean_prior.npy")
    mean_fS_MLL = fe.Function(domain.function_space)
    mean_fS_MLL.vector()[:] = np.array(temp)
    prior_measure_MLL = GaussianElliptic2(
        domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0, 
        boundary=boundary_condition, mean_fun=mean_fS_MLL
        )
    
    ## 3. The prior measure with learned mean and variance by MAP with hyper-prior.
    
    prior_measure_fS_MAP = GaussianElliptic2(
        domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0, 
        boundary=boundary_condition, mean_fun=mean_fS_MAP
        )
    
    prior_measure_fS_MLL = GaussianElliptic2(
        domain=domain, alpha=alpha_prior, a_fun=fe.Constant(aa), theta=1.0, 
        boundary=boundary_condition, mean_fun=mean_fS_MLL
        )
        
    """
    There will be two test datasets corresponding to two branches of the random model parameters
    """
    dd = meta_data_y_test[idx_] ## data of the model parameter that is above the x-axis
    
    ## Set the noise 
    noise_level_ = noise_level
    noise = NoiseGaussianIID(dim=len(dd))
    noise.set_parameters(variance=noise_level_**2)
       
    """
    There will be two models corresponding to two branches of the random model parameters
    """
    ## model1 with prior measure $\mathcal{N}(0, \mathcal{C}_0)$ and data d1
    model = ModelBackwarDiffusion(
        d=dd, domain_equ=domain, prior=prior_measure, 
        noise=noise, equ_solver=equ_solver
        )
    
    model_MAP = ModelBackwarDiffusion(
        d=dd, domain_equ=domain, prior=prior_measure_MAP, 
        noise=noise, equ_solver=equ_solver
        )
    
    model_MLL = ModelBackwarDiffusion(
        d=dd, domain_equ=domain, prior=prior_measure_MLL, 
        noise=noise, equ_solver=equ_solver
        )
    
    model_fS_MAP = ModelBackwarDiffusion(
        d=dd, domain_equ=domain, prior=prior_measure_fS_MAP, 
        noise=noise, equ_solver=equ_solver
        )
    
    model_fS_MLL = ModelBackwarDiffusion(
        d=dd, domain_equ=domain, prior=prior_measure_fS_MLL, 
        noise=noise, equ_solver=equ_solver
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
        max_iter = 4
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
    
    mean_unlearn = fe.Function(domain.function_space)
    mean_unlearn.vector()[:] = np.array(calculate_MAP(model))
    
        
    mean_learn_MAP = fe.Function(domain.function_space)
    mean_learn_MAP.vector()[:] = np.array(calculate_MAP(model_MAP))
    
    mean_learn_MLL = fe.Function(domain.function_space)
    mean_learn_MLL.vector()[:] = np.array(calculate_MAP(model_MLL))
    
    mean_learn_fS_MAP = fe.Function(domain.function_space)
    mean_learn_fS_MAP.vector()[:] = np.array(calculate_MAP(model_fS_MAP))
                                      
    mean_learn_fS_MLL = fe.Function(domain.function_space)
    mean_learn_fS_MLL.vector()[:] = np.array(calculate_MAP(model_fS_MLL))
    
    err_L2 = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="L2") 
    err_Max = relative_error(mean_unlearn, u_meta_fun_test, domain, err_type="Max")
    
    err_learn_MAP_L2 = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="L2") 
    err_learn_MAP_Max = relative_error(mean_learn_MAP, u_meta_fun_test, domain, err_type="Max")
    err_learn_MLL_L2 = relative_error(mean_learn_MLL, u_meta_fun_test, domain, err_type="L2") 
    err_learn_MLL_Max = relative_error(mean_learn_MLL, u_meta_fun_test, domain, err_type="Max")
    
    err_learn_fS_MAP_L2 = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="L2") 
    err_learn_fS_MAP_Max = relative_error(mean_learn_fS_MAP, u_meta_fun_test, domain, err_type="Max")
    err_learn_fS_MLL_L2 = relative_error(mean_learn_fS_MLL, u_meta_fun_test, domain, err_type="L2") 
    err_learn_fS_MLL_Max = relative_error(mean_learn_fS_MLL, u_meta_fun_test, domain, err_type="Max")
    
    
    error_posterior_mean.append([
        err_L2, err_Max, 
        err_learn_MAP_L2, err_learn_MAP_Max, err_learn_MLL_L2, err_learn_MLL_Max, 
        err_learn_fS_MAP_L2, err_learn_fS_MAP_Max, err_learn_fS_MLL_L2, 
        err_learn_fS_MLL_Max
        ])
    
error_posterior_mean = np.array(error_posterior_mean)

tmp1 = error_posterior_mean[:,0]
tmp2 = error_posterior_mean[:,1]
print("unlearned error L2: %.6f" % (np.mean(tmp1)))
print("unlearned error Max: %.6f" % (np.mean(tmp2)))
tmp1 = error_posterior_mean[:,2]
tmp2 = error_posterior_mean[:,3]
print("learn MAP L2: %.6f" % (np.mean(tmp1)))
print("learn MAP Max: %.6f" % (np.mean(tmp2)))
tmp1 = error_posterior_mean[:,4]
tmp2 = error_posterior_mean[:,5]
print("learn MLL L2: %.6f" % (np.mean(tmp1)))
print("learn MLL Max: %.6f" % (np.mean(tmp2)))
tmp1 = error_posterior_mean[:,6]
tmp2 = error_posterior_mean[:,7]
print("learn fS MAP L2: %.6f" % (np.mean(tmp1)))
print("learn fS MAP Max: %.6f" % (np.mean(tmp2)))
tmp1 = error_posterior_mean[:,8]
tmp2 = error_posterior_mean[:,9]
print("learn fS MLL L2: %.6f" % (np.mean(tmp1)))
print("learn fS MLL Max: %.6f" % (np.mean(tmp2)))

###----------------------------------------------------------------------------
plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
with open(data_dir + env + "_meta_parameters_test", 'rb') as f: 
    env_samples = pickle.load(f)
env_samples = np.array(env_samples)
mesh_meta_test = fe.Mesh(data_dir + env + '_saved_mesh_meta_test.xml')
V_meta = fe.FunctionSpace(mesh_meta_test, 'P', 1)
tmp_fun = fe.Function(V_meta)
for idx in range(env_samples.shape[0]):
    tmp_fun.vector()[:] = np.array(env_samples[idx])
    fe.plot(tmp_fun)
plt.ylim([-10.5, 10.5])
plt.title("Environment samples")
plt.subplot(1,2,2)
fe.plot(u_meta_fun_test, label="Truth")
fe.plot(mean_unlearn, label="Estimate (unlearned prior)")
fe.plot(mean_learn_MAP, label="Estimate (data-independent prior)")
fe.plot(mean_learn_fS_MAP, label="Estimate (data-dependent prior)")
plt.legend()
plt.title("Estimates with different prior")

# plt.figure(figsize=(13, 5))
# plt.subplot(1,2,1)
# fe.plot(u_meta_fun_test1, label="truth")
# fe.plot(mean_unlearn_1, label="MAP unlearn")
# fe.plot(mean_learn_prior_1, label="MAP estimate")
# fun = fe.Function(domain.function_space)
# fun.vector()[:] = np.array(model_fS1_prior.prior.mean_vec)
# fe.plot(fun, label="prior mean")
# plt.legend()
# plt.subplot(1,2,2)
# ff = fe.interpolate(u_meta_fun_test1, domain.function_space)
# sol = equ_solver1.forward_solver(m_vec=ff.vector()[:])
# dd1 = equ_solver1.S@sol
# plt.plot(dd1, label="simulated data")
# sol = equ_solver1.forward_solver(m_vec=mean_learn_prior_1.vector()[:])
# dd2 = equ_solver1.S@sol
# plt.plot(dd2, label="data posterior mean")
# sol = equ_solver1.forward_solver(m_vec=model_fS1_prior.prior.mean_vec)
# dd3 = equ_solver1.S@sol
# plt.plot(dd3, label="data prior mean")
# plt.legend()