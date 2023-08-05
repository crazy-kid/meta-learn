#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:33:59 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from core.misc import eval_min_max

from BackwardDiffusion.common import EquSolver, ModelBackwarDiffusion
from NN_library import FNO1d, d2fun


## set data and result dir
data_dir = './DATA/'
results_dir = './RESULTS/pCN/'
meta_results_dir = './RESULTS/'
env = "complex"

## set the step length of MCMC
beta = 0.01
## set the total number of the sampling
length_total = 1e5

## load the mesh used in the inversion 
mesh_inv = fe.Mesh(data_dir + env + '_saved_mesh_meta_pCN.xml')
V_inv = fe.FunctionSpace(mesh_inv, 'P', 1)

## load model parameters; load test samples 
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

## set the path for saving results
samples_file_path1 = results_dir + '1_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path1, exist_ok=True)
draw_path1 = results_dir + '1_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path1, exist_ok=True)

## set the path for saving results
samples_file_path2 = results_dir + '2_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path2, exist_ok=True)
draw_path2 = results_dir + '2_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path2, exist_ok=True)

## set the path for saving results
samples_file_path_fS1 = results_dir + 'fS1_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS1, exist_ok=True)
draw_path_fS1 = results_dir + 'fS1_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS1, exist_ok=True)

## set the path for saving results
samples_file_path_fS2 = results_dir + 'fS2_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS2, exist_ok=True)
draw_path_fS2 = results_dir + 'fS2_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS2, exist_ok=True)

## set the path for saving results
samples_file_path_fS1_prior = results_dir + 'fS1_prior_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS1_prior, exist_ok=True)
draw_path_fS1_prior = results_dir + 'fS1_prior_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS1_prior, exist_ok=True)

## set the path for saving results
samples_file_path_fS2_prior = results_dir + 'fS2_prior_beta_' + str(beta) + '/samples/'
os.makedirs(samples_file_path_fS2_prior, exist_ok=True)
draw_path_fS2_prior = results_dir + 'fS2_prior_beta_' + str(beta) + '/draws/'
os.makedirs(draw_path_fS2_prior, exist_ok=True)

## load the accept rate data
acc1 = np.load(samples_file_path1[:-8] + ".npy")[:,1]
acc2 = np.load(samples_file_path2[:-8] + ".npy")[:,1]
acc_fS1 = np.load(samples_file_path_fS1[:-8] + ".npy")[:,1]
acc_fS2 = np.load(samples_file_path_fS2[:-8] + ".npy")[:,1]
acc_fS1_prior = np.load(samples_file_path_fS1_prior[:-8] + ".npy")[:,1]
acc_fS2_prior = np.load(samples_file_path_fS2_prior[:-8] + ".npy")[:,1]

## calculate the posterior mean function

def cal_posterior_mean(path_dir, num_start=0):
    num_total = np.int64(len(os.listdir(path_dir)))
    mean_vector = 0
    
    for i in range(num_start, num_total):
        params = np.load(path_dir + 'sample_' + str(i) + '.npy')
        mean_ = np.sum(params, axis=0)
        mean_vector += mean_/params.shape[0]
    
    mean_vector = mean_vector/num_total
    
    return np.array(mean_vector)


def cal_trace(path_dir, num_start=0, idx=0):
    num_total = np.int64(len(os.listdir(path_dir)))
    
    trace = np.load(path_dir + 'sample_' + str(num_start) + '.npy')[:,idx]
    
    for i in range(num_start+1, num_total):
        params = np.load(path_dir + 'sample_' + str(i) + '.npy')
        trace = np.concatenate((trace, params[:,idx]))
    
    trace = np.array(trace)
    
    return trace


num_start = 30

mean_fun1 = fe.Function(V_inv) 
mean_fun1.vector()[:] = cal_posterior_mean(samples_file_path1, num_start)
mean_fun2 = fe.Function(V_inv) 
mean_fun2.vector()[:] = cal_posterior_mean(samples_file_path2, num_start)
mean_fun_fS1 = fe.Function(V_inv) 
mean_fun_fS1.vector()[:] = cal_posterior_mean(samples_file_path_fS1, num_start)
mean_fun_fS2 = fe.Function(V_inv) 
mean_fun_fS2.vector()[:] = cal_posterior_mean(samples_file_path_fS2, num_start)
mean_fun_fS1_prior = fe.Function(V_inv) 
mean_fun_fS1_prior.vector()[:] = cal_posterior_mean(samples_file_path_fS1_prior, num_start)
mean_fun_fS2_prior = fe.Function(V_inv) 
mean_fun_fS2_prior.vector()[:] = cal_posterior_mean(samples_file_path_fS2_prior, num_start)

# num_start = 0
# idx = 150

# trace1 = cal_trace(samples_file_path1, num_start=num_start, idx=idx)
# trace_fS1 = cal_trace(samples_file_path_fS1, num_start=num_start, idx=idx)
# trace_fS1_prior = cal_trace(samples_file_path_fS1_prior, num_start=num_start, idx=idx)
# trace2 = cal_trace(samples_file_path2, num_start=num_start, idx=idx)
# trace_fS2 = cal_trace(samples_file_path_fS2, num_start=num_start, idx=idx)
# trace_fS2_prior = cal_trace(samples_file_path_fS2_prior, num_start=num_start, idx=idx)

# len_trace = trace1.shape[0]
# trace_start, trace_end = len_trace-2000, len_trace 
# trace1 = trace1[trace_start:trace_end]
# trace2 = trace2[trace_start:trace_end]
# trace_fS1 = trace_fS1[trace_start:trace_end]
# trace_fS2 = trace_fS2[trace_start:trace_end]
# trace_fS1_prior = trace_fS1_prior[trace_start:trace_end]
# trace_fS2_prior = trace_fS2_prior[trace_start:trace_end]


## draw the figures
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 0.4
plt.figure(figsize=(13, 13))

plt.subplot(2, 2, 1)
fe.plot(u_meta_fun_test1, label="Model parameter")
fe.plot(mean_fun1, linewidth=1.5, linestyle=":", label="Unlearned prior")
fe.plot(mean_fun_fS1, alpha=0.8, linestyle="--", label="Learned prior (no hyperprior)")
fe.plot(mean_fun_fS1_prior, linewidth=2, linestyle="-.", 
         label="Learned prior (with hyperprior)")
plt.legend()
min_val, max_val = eval_min_max([
    u_meta_fun_test1.vector()[:], mean_fun1.vector()[:], 
    mean_fun_fS1.vector()[:], mean_fun_fS1_prior.vector()[:]
    ])
plt.ylim([min_val-0.1, max_val+3])
plt.title("(a) True and estimated model parameters")

plt.subplot(2, 2, 2)
plt.plot(acc1, label="Zero mean prior")
plt.plot(acc_fS1, linestyle="--", label="Learned mean by MLL")
plt.plot(acc_fS1_prior, linestyle="-.", label="Learned mean by MAP")
plt.legend()
min_val, max_val = eval_min_max([
    acc1, acc_fS1, acc_fS1_prior
    ])
plt.ylim([min_val-0.01, max_val+0.05])
plt.title("(b) Acceptance rates")

# plt.subplot(2, 3, 3)
# plt.plot(trace1, label="Zero mean prior")
# plt.plot(trace_fS1, label="Learned mean by MLL")
# plt.plot(trace_fS1_prior, label="Learned mean by MAP")
# min_val, max_val = eval_min_max([
#     trace1, trace_fS1, trace_fS1_prior
#     ])
# plt.legend()
# plt.ylim([min_val-0.1, max_val+0.1])
# plt.title("(c) Traces of the first component")

plt.subplot(2, 2, 3)
fe.plot(u_meta_fun_test2, label="Model parameter")
fe.plot(mean_fun2, linewidth=1.5, linestyle=":", label="Unlearned prior")
fe.plot(mean_fun_fS2, alpha=0.8, linestyle="--", label="Learned prior (no hyperprior)")
fe.plot(mean_fun_fS2_prior, linewidth=2, linestyle="-.", 
         label="Learned prior (with hyperprior)")
plt.legend()
min_val, max_val = eval_min_max([
    u_meta_fun_test2.vector()[:], mean_fun2.vector()[:], 
    mean_fun_fS2.vector()[:], mean_fun_fS2_prior.vector()[:]
    ])
plt.legend()
plt.ylim([min_val-0.1, max_val+3])
plt.title("(c) True and estimated model parameters")

plt.subplot(2, 2, 4)
plt.plot(acc2, label="Zero mean prior")
plt.plot(acc_fS2, linestyle="--", label="Learned mean by MLL")
plt.plot(acc_fS2_prior, linestyle="-.", label="Learned mean by MAP")
plt.legend()
min_val, max_val = eval_min_max([
    acc2, acc_fS2, acc_fS2_prior
    ])
plt.ylim([min_val-0.01, max_val+0.05])
plt.title("(d) Acceptance rates")

# plt.subplot(2, 3, 6)
# plt.plot(trace2, label="Zero mean prior")
# plt.plot(trace_fS2, label="Learned mean by MLL")
# plt.plot(trace_fS2_prior, label="Learned mean by MAP")
# min_val, max_val = eval_min_max([
#     trace2, trace_fS2, trace_fS2_prior
#     ])
# plt.legend()
# plt.ylim([min_val-0.1, max_val+0.1])
# plt.title("(c) Traces of the first component")

plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2) 



















