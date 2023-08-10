#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:14:52 2022

@author: jjx323
"""

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import torch
import cupy as cp
import cupyx.scipy.sparse as cpss
import cupyx.scipy.sparse.linalg as cpssl

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.model import Domain2D
from core.probability import GaussianElliptic2, GaussianFiniteRank
from core.noise import NoiseGaussianIID
from core.optimizer import GradientDescent, NewtonCG
from core.misc import load_expre, smoothing, print_my

from SteadyStateDarcyFlow.common import EquSolver, ModelDarcyFlow
from SteadyStateDarcyFlow.MLcommon import GaussianElliptic2Torch, \
    GaussianFiniteRankTorch, PDEFun, PDEasNet, LossResidual


## set data and result dir
data_dir = './DATA/'

# device = "cuda"
device = "cpu"

noise_level = 0.01

## domain for solving PDE
equ_nx = 200
domain = Domain2D(nx=equ_nx, ny=equ_nx, mesh_type='P', mesh_order=1)

model_params = np.load(data_dir + "model_params.npy")
dataset_x = np.load(data_dir + "dataset_x.npy")
dataset_y = np.load(data_dir + "dataset_y.npy")

## loading the mesh information of the model_params
mesh_truth = fe.Mesh(data_dir + 'saved_mesh_truth.xml')
V_truth = fe.FunctionSpace(mesh_truth, 'P', 1)
truth_fun = fe.Function(V_truth, data_dir + 'truth_fun.xml')

truth_fun.vector()[:] = np.array(model_params[0,:])
truth_fun = fe.interpolate(truth_fun, domain.function_space)

points = dataset_x[0, :]
d_noisy = dataset_y[0, :]

## define the prior measure 
# domain_ = Domain2D(nx=50, ny=50, mesh_type='P', mesh_order=1)
# gaussian_measure = GaussianFiniteRankTorch(
#     domain=domain, domain_=domain_, alpha=0.01, beta=1, s=2
#     )
# gaussian_measure.calculate_eigensystem()
# gaussian_measure.trans2torch(device=device)

f = load_expre(data_dir + 'f_2D.txt')
f = fe.interpolate(fe.Expression(f, degree=3), domain.function_space)

equ_solver = EquSolver(domain_equ=domain, m=fe.Constant(0.0), f=f, points=points)

pdefun = PDEFun.apply
pdeasnet = PDEasNet(pdefun, equ_solver)

## setting the noise
noise_level_ = noise_level*max(d_noisy)
noise = NoiseGaussianIID(dim=len(points))
noise.set_parameters(variance=noise_level_**2)

max_iter = 2000

criterion_res = LossResidual(noise)
criterion_res.to(device)

dim = equ_solver.M.shape[0]

import scipy.sparse as sps
import scipy.sparse.linalg as spsl

def spsolve_lu(L, U, b, perm_c=None, perm_r=None):
    """ an attempt to use SuperLU data to efficiently solve
        Ax = Pr.T L U Pc.T x = b
         - note that L from SuperLU is in CSC format solving for c
           results in an efficiency warning
        Pr . A . Pc = L . U
        Lc = b      - forward solve for c
         c = Ux     - then back solve for x
    """
    if perm_r is not None:
        perm_r_rev = np.argsort(perm_r)
        b = b[perm_r_rev]

    try:    # unit_diagonal is a new kw
        # c = spsl.spsolve_triangular(L, b, lower=True, unit_diagonal=True)
        c = spsl.spsolve(L, b, permc_spec="NATURAL")
    except TypeError:
        c = spsl.spsolve_triangular(L, b, lower=True)
    # px = spsl.spsolve_triangular(U, c, lower=False)
    px = spsl.spsolve(U, c, permc_spec="NATURAL")
    if perm_c is None:
        return px
    return px[perm_c]

import time
start = time.time()
lu = spsl.splu(equ_solver.K.tocsc(), permc_spec="NATURAL")
end = time.time()
print("splu: ", end - start)

start = time.time()
sol1 = lu.solve(equ_solver.F)
end = time.time()
print("lu.solve: ", end - start)

start = time.time()
sol2 = spsl.spsolve(equ_solver.K, equ_solver.F)
end = time.time()
print("spsolve: ", end - start)

L = sps.csr_matrix(lu.L)
U = sps.csr_matrix(lu.U)
start = time.time()
sol3 = spsolve_lu(L, U, equ_solver.F, perm_c=lu.perm_c, perm_r=lu.perm_r)
end = time.time()
print("spsolve_lu: ", end - start)

class SuperLU():

    def __init__(self, shape, L, U, perm_r, perm_c, nnz):
        """LU factorization of a sparse matrix.
        Args:
            obj (scipy.sparse.linalg.SuperLU): LU factorization of a sparse
                matrix, computed by `scipy.sparse.linalg.splu`, etc.
        """

        self.shape = shape
        self.nnz = nnz
        self.perm_r = cp.array(perm_r)
        self.perm_c = cp.array(perm_c)
        self.L = cp.sparse.csr_matrix(L.tocsr())
        self.U = cp.sparse.csr_matrix(U.tocsr())

        # self._perm_r_rev = cp.argsort(self.perm_r)
        # self._perm_c_rev = cp.argsort(self.perm_c)
        self._perm_r_rev = cp.array(np.argsort(perm_r))
        self._perm_c_rev = cp.array(np.argsort(perm_c))

    def solve(self, rhs, trans='N'):
        """Solves linear system of equations with one or several right-hand sides.
        Args:
            rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
                ``(M)`` or ``(M, K)``.
            trans (str): 'N', 'T' or 'H'.
                'N': Solves ``A * x = rhs``.
                'T': Solves ``A.T * x = rhs``.
                'H': Solves ``A.conj().T * x = rhs``.
        Returns:
            cupy.ndarray:
                Solution vector(s)
        """
        if not isinstance(rhs, cp.ndarray):
            raise TypeError('ojb must be cupy.ndarray')
        if rhs.ndim not in (1, 2):
            raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.
                             format(rhs.ndim))
        if rhs.shape[0] != self.shape[0]:
            raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'
                             .format(self.shape, rhs.shape))
        if trans not in ('N', 'T', 'H'):
            raise ValueError('trans must be \'N\', \'T\', or \'H\'')
        if not cp.cusparse.check_availability('csrsm2'):
            raise NotImplementedError

        x = rhs.astype(self.L.dtype)
        if trans == 'N':
            if self.perm_r is not None:
                x = x[self._perm_r_rev]
            cp.cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            cp.cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            if self.perm_c is not None:
                x = x[self.perm_c]
        else:
            if self.perm_c is not None:
                x = x[self._perm_c_rev]
            cp.cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            cp.cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            if self.perm_r is not None:
                x = x[self.perm_r]

        if not x._f_contiguous:
            # For compatibility with SciPy
            x = x.copy(order='F')
        return x
    
start = time.time()
lu_gpu = SuperLU(
    shape=lu.shape, L=lu.L, U=lu.U, perm_c=lu.perm_c, perm_r=lu.perm_r, 
    nnz=lu.nnz 
    )
sol4 = lu_gpu.solve(cp.array(equ_solver.F))
end = time.time()
print("lu_gpu: ", end - start)


# target = torch.tensor(d_noisy, dtype=torch.float32, requires_grad=False).to(device)
# init_fun = smoothing(truth_fun, alpha=0.1)
# u = torch.tensor(
#     init_fun.vector()[:], dtype=torch.float32, requires_grad=True, 
    
#     device=target.device
#     )
# optimizer = torch.optim.Adam([u], lr=1e-2)

# pdeasnet.to(device)

# import time
# start = time.time()
# for itr in range(max_iter):
#     optimizer.zero_grad()
#     preds = pdeasnet(u)
#     loss_res = criterion_res(preds, target) 
#     loss_reg = gaussian_measure.evaluate_CM_inner(u)
#     loss = loss_res + loss_reg
#     loss.backward() 
#     for g in optimizer.param_groups:
#         g["lr"] = min(1e-1/torch.linalg.norm(u), 1e-1)
#     optimizer.step() 
    
#     if itr % 100 == 0:
#         print("Iter = %d/%d, loss_res = %.4f, loss_reg = %.4f, loss = %.4f" 
#               % (itr, max_iter, loss_res.item(), loss_reg.item(), loss.item()))
# end = time.time()
# print_my("Consume time(" + device + "): ", end - start, color="red")

# if u.device.type == "cuda":
#     m_vec = np.array(u.cpu().detach())
# elif u.device.type == "cpu":
#     m_vec = np.array(u.detach())
# noise.to_numpy()
# fun = fe.Function(domain.function_space)
# fun.vector()[:] = m_vec

# equ_solver.update_m(m_vec)
# sol1 = equ_solver.forward_solver()
# d_est = equ_solver.S@sol1

# plt.figure(figsize=(18, 5))
# plt.subplot(1, 3, 1)
# fig = fe.plot(truth_fun)
# plt.colorbar(fig)
# plt.title("Truth Function")
# plt.subplot(1, 3, 2)
# fig = fe.plot(fun)
# plt.colorbar(fig)
# plt.title("Estimated Function")
# plt.subplot(1, 3, 3)
# plt.plot(d_est, label='d_est')
# plt.plot(d_noisy, label='d')
# plt.legend()
# plt.title('Gradient Descent')



# ###############################################################################
# domain_ = Domain2D(nx=50, ny=50, mesh_type='P', mesh_order=1)
# gaussian_measure = GaussianFiniteRankTorch(
#     domain=domain, domain_=domain_, alpha=0.1, beta=1, s=2
#     )
# gaussian_measure.calculate_eigensystem()

# gaussian_measure.trans2torch()
# sample_torch = gaussian_measure.generate_sample(num_sample=5)

# fun = fe.interpolate(truth_fun, domain.function_space)
# a = gaussian_measure.evaluate_CM_inner(fun.vector()[:])

# gaussian_measure.trans2numpy()
# sample_numpy = gaussian_measure.generate_sample(num_sample=5)
# b = gaussian_measure.evaluate_CM_inner(fun.vector()[:])

# fun1 = fe.Function(domain.function_space)
# fun1.vector()[:] = np.array(sample_torch.detach().numpy()[:, 0])
# fun2 = fe.Function(domain.function_space)
# fun2.vector()[:] = np.array(sample_numpy[:, 0])

###############################################################################
# ## setting the prior measure
# gaussian_measure = GaussianElliptic2Torch(
#     domain=domain, alpha=1, a_fun=1, theta=0.1, boundary='Neumann',
#     mean_fun=fe.Constant(0.0)
#     )

# gaussian_measure.trans2torch()
# sample_torch = gaussian_measure.generate_sample(num_sample=5)

# fun = fe.interpolate(truth_fun, domain.function_space)
# a = gaussian_measure.evaluate_CM_inner_torch(fun.vector()[:])

# gaussian_measure.trans2numpy()
# sample_numpy = gaussian_measure.generate_sample(num_sample=5)
# b = gaussian_measure.evaluate_CM_inner(fun.vector()[:])

# fun1 = fe.Function(domain.function_space)
# fun1.vector()[:] = sample_torch.detach().numpy()[:,0]
# fun2 = fe.Function(domain.function_space)
# fun2.vector()[:] = sample_numpy[:,0]




