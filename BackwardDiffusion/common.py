#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 21:04:22 2022

@author: jjx323
"""

from core.misc import construct_measurement_matrix, trans2spnumpy
from core.model import ModelBase
import numpy as np
# import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
# import dolfin as dl

import sys
import os
sys.path.append(os.pardir)

###########################################################################


class EquSolver(object):
    '''
    The forward equation: 
        \frac{\partial v(x,t)}{\partial_t} - \Delta v(x,t) = 0  in \Omega
        v(x, t) = 0 on \partial\Omega\times(0, T), 
        v(x, 0) = m(x) in \Omega

    The adjoint equation (transform s=T-t):
        \frac{\partial w(x,s)}{\partial_s} - \Delta w(x,s) = 0
        w(x, s) = 0 on \partial\Omega\times(0, T)
        w(x, 0) = -\mathcal{B}^*(\mathcal{B}v(x,T) - d)  in \Omega
    where $T$ is a fixed positive number, $\mathcal{B}$ is the measurement operator, and 
    $d$ is the noisy data. The definition of $\mathcal{B}$ is as follow:
        \mathcal{B}(v(x, T)) = (v(x_1, T), \ldots, v(x_{N_d}, T))^T. 

    The incremental forward equation is the same as the forward equation.

    The incremental adjoint eqution (transform s=T-t):  
        \frac{\partial w(x,s)}{\partial_s} - \Delta w(x,s) = 0
        w(x, s) = 0 on \partial\Omega\times(0, T)
        w(x, 0) = -\mathcal{B}^*\mathcal{B}\tilde{v}(T,x)  in \Omega
    where $\tilde{v}$ is the solution of incremental forward equation. 
    '''

    def __init__(self, domain_equ, T, num_steps, points, m=None):
        self.domain = domain_equ
        self.T = T
        self.num_steps = num_steps
        self.dt = self.T/self.num_steps
        self.points = points.copy()
        if m is None:
            m = fe.Constant('0.0')
        self.m_param: fe.Function = fe.interpolate(
            m, self.domain.function_space)
        self.m_vec_: fe.Function = fe.interpolate(
            m, self.domain.function_space).vector()

        u_, v_ = fe.TrialFunction(self.domain.function_space), fe.TestFunction(
            self.domain.function_space)
        self.M_ = fe.assemble(fe.inner(u_, v_)*fe.dx)
        self.A_ = fe.assemble(fe.inner(fe.grad(u_), fe.grad(v_))*fe.dx)
        self.S = construct_measurement_matrix(
            self.points, self.domain.function_space)

        def boundary(x, on_boundary):
            return on_boundary
        bc = fe.DirichletBC(self.domain.function_space,
                            fe.Constant('0.0'), boundary)
        bc.apply(self.A_)
        bc.apply(self.M_)

        # specify which element will be set zero for force term
        # Here we ignore the internal mechnisms of the FEniCS software.
        # If we know the details of the FEM software, there should be more clear way to specify self.bc_idx
        temp1 = fe.assemble(fe.inner(fe.Constant("1.0"), v_)*fe.dx)
        temp2 = temp1[:].copy()
        bc.apply(temp1)
        self.bc_idx = (temp2 != temp1)

        self.M = trans2spnumpy(self.M_)
        self.A = trans2spnumpy(self.A_)
        self.len_vec = self.A.shape[0]

    def update_m(self, m_vec):
        #        self.m_vec_.set_local(m_vec)
        #        self.m_param.vector().set_local(self.m_vec_)
        self.m_vec_[:] = np.array(m_vec[:])
        self.m_param.vector()[:] = np.array(self.m_vec_[:])

    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(
            self.points, self.domain.function_space)

    def forward_solver(self, m_vec=None):
        if m_vec is not None:
            self.update_m(m_vec)

        t = 0
        v_n = self.m_vec_[:].copy()

        for itr in range(self.num_steps):
            t += self.dt
            rhs = self.M@v_n  # @是numpy中的点乘
            # To keep the homogeneous Dirichlet boundary condition, it seems crucial to
            # apply boundary condition to the right-hand side for the elliptic equations.
            rhs[self.bc_idx] = 0.0
            v = spsl.spsolve(self.M + self.dt*self.A, rhs)
            v_n = v.copy()

        return np.array(v_n)

    def incremental_forward_solver(self, m_hat=None):
        # For linear problems, the incremental forward == forward
        if m_hat is None:
            m_hat = np.array(self.m_vec_[:])

        t = 0
        v_n = m_hat[:].copy()

        for itr in range(self.num_steps):
            t += self.dt
            rhs = self.M@v_n
            # To keep the homogeneous Dirichlet boundary condition, it seems crucial to
            # apply boundary condition to the right-hand side for the elliptic equations.
            rhs[self.bc_idx] = 0.0
            v = spsl.spsolve(self.M + self.dt*self.A, rhs)
            v_n = v.copy()

        return np.array(v_n)

    def adjoint_solver(self, res_vec):
        # res_vec = Sv - d
        t = 0
        v_n = -spsl.spsolve(self.M, (self.S.T)@res_vec)

        for itr in range(self.num_steps):
            t += self.dt
            rhs = self.M@v_n
            # To keep the homogeneous Dirichlet boundary condition, it seems crucial to
            # apply boundary condition to the right-hand side for the elliptic equations.
            rhs[self.bc_idx] = 0.0
            v = spsl.spsolve(self.M + self.dt*self.A, rhs)
            v_n = v.copy()

        return np.array(v_n)

    def incremental_adjoint_solver(self, vec, m_hat=None):
        t = 0
        v_n = -spsl.spsolve(self.M, self.S.T@vec)

        for itr in range(self.num_steps):
            t += self.dt
            rhs = self.M@v_n
            # To keep the homogeneous Dirichlet boundary condition, it seems crucial to
            # apply boundary condition to the right-hand side for the elliptic equations.
            rhs[self.bc_idx] = 0.0
            v = spsl.spsolve(self.M + self.dt*self.A, rhs)
            v_n = v.copy()

        return np.array(v_n)

    def construct_fun(self, f_vec):
        f = fe.Function(self.domain.function_space)
        f.vector().set_local(f_vec)
        return f

###########################################################################


class ModelBackwarDiffusion(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        super().__init__(d, domain_equ, prior, noise, equ_solver)

    def update_m(self, m_vec, update_sol=True):
        # print(np.array(m_vec).shape, self.m.vector()[:].shape)
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(self.m.vector())
        if update_sol is True:
            self.p.vector()[:] = self.equ_solver.forward_solver()

    def loss_residual(self):
        temp = (self.S@self.p.vector()[:] - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*temp

    def loss_residual_L2(self):
        temp = (self.S@self.p.vector()[:] - self.d)
        temp = temp@temp
        return 0.5*temp

    def eval_grad_residual(self, m_vec):
        self.equ_solver.update_m(m_vec)
        self.p.vector()[:] = self.equ_solver.forward_solver()
        res_vec = spsl.spsolve(self.noise.covariance,
                               self.S@(self.p.vector()[:]) - self.d)
        # print(res_vec)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        g = -g_.vector()[:]
        return np.array(g)

    def eval_hessian_res_vec(self, dm):
        self.equ_solver.update_m(dm)
        self.p.vector()[:] = self.equ_solver.forward_solver()
        res_vec = spsl.spsolve(self.noise.covariance,
                               self.S@(self.p.vector()[:]))
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        HM = -g_.vector()[:]
        return np.array(HM)


###########################################################################
# class PosteriorSamples(object):
#     '''
#     Using a group of samples to calculate the posterior information
#     '''
#     def __init__(self, V, samples):
#         ## sample_list is a list in Python
#         self.V = V  # V is the function space of samples
#         self.samples_vec = samples
#         self.length = len(samples)
#         self.mean_vec = np.mean(np.array(self.samples_vec), axis=0)
#         self.cov_fun = None
#         self.mean_fun = fe.Function(self.V)
#         self.mean_fun.vector()[:] = self.mean_vec.copy()

#     def eval_mean(self, x_vec):
#         S = construct_measurement_matrix(x_vec, self.V)
#         return S@self.mean_fun.vector()[:]

#     def eval_std(self, x_vec):
#         std = 0
#         S = construct_measurement_matrix(x_vec, self.V)
#         for i in range(self.length):
#             temp = S@self.samples_vec[i] - S@self.mean_fun.vector()[:]
#             std = std + temp*temp
#         std = std/self.length
#         return std

#     def eval_cov(self, x_vec, y_vec):
#         cov = 0
#         Sx = construct_measurement_matrix(x_vec, self.V)
#         Sy = construct_measurement_matrix(y_vec, self.V)
#         for i in range(self.length):
#             temp_x = Sx@self.samples_vec[i] - Sx@self.mean_fun.vector()[:]
#             temp_y = Sy@self.samples_vec[i] - Sy@self.mean_fun.vector()[:]
#             cov = cov + temp_x*temp_y
#         cov = cov/self.length
#         return cov
