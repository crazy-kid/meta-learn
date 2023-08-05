#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:04:36 2022

@author: jjx323
"""

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from collections import OrderedDict
import fenics as fe
import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
sys.path.append(os.pardir)

from core.misc import trans2spnumpy, spnumpy2sptorch, trans2sptorch, sptorch2spnumpy, \
    construct_measurement_matrix





class AinvB(torch.autograd.Function):
    """
    Calculate x = A^{-1}B
    input: B; (batch_size, vector_dim, )
    output: A^{-1}b
    """
    @staticmethod
    def forward(ctx, input, A):
        A = spnumpy2sptorch(A)
        ctx.save_for_backward(input, A)
        
        m_vec = np.array(input.detach(), dtype=np.float32)
        
        ## solve forward equation
        v_n = m_vec[:].copy()
        
        for itr in range(equ_solver.num_steps):
            rhs = equ_solver.M@v_n
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[equ_solver.bc_idx] = 0.0 
            v = spsl.spsolve(equ_solver.M + equ_solver.dt*equ_solver.A, rhs)
            v_n = v.copy()   
        
        output = torch.tensor(equ_solver.S@v_n, dtype=torch.float32)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, M, A, S, bc_idx, dt, num_steps = ctx.saved_tensors 
        M = sptorch2spnumpy(M)
        A = sptorch2spnumpy(A)
        S = S.numpy()
        bc_idx = np.int64(bc_idx) 
        dt = dt.numpy()
        
        v_n = -spsl.spsolve(M, (S.T)@np.array(grad_output))
        
        for itr in range(num_steps):
            rhs = M@v_n 
            ## To keep the homogeneous Dirichlet boundary condition, it seems crucial to 
            ## apply boundary condition to the right-hand side for the elliptic equations. 
            rhs[bc_idx] = 0.0 
            v = spsl.spsolve(M + dt*A, rhs)
            v_n = v.copy()    

        val = torch.tensor(-v_n, dtype=torch.float32)
        return val, None

