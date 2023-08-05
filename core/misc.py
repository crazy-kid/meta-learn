#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:50:14 2022

@author: Junxiong Jia
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import torch


#########################################################################
'''
条件数 condition num : 描述问题的适定型 越小适定型越好
preconditioner : 降低条件数的算子/矩阵
ILU Incomplete LU factorization: preconditioner的一种 寻找 LU～A 而不是 LU=A
'''
# FEniCS interfaces several linear algebra packages, called linear algebra backends in FEniCS terminology.
fe.parameters['linear_algebra_backend'] = 'Eigen'  # ? 这个Eigen是C++的Eigen库吗


def trans2spnumpy(M, dtype=np.float64):
    '''
    This function transfer the sparse matrix generated by FEniCS 
    into numpy sparse array. 
    Example: 
    A_low_level = fe.assemble(fe.inner(u, v)*fe.dx)
    A = trans2spnumpy(A_low_level)
    '''
    row, col, val = fe.as_backend_type(M).data()
    return sps.csr_matrix((val, col, row), dtype=dtype)


def trans2sptorch(M, dtype=torch.float32):
    '''
    Converting a sparse matrix generated by FEniCS through fe.assemble 
    to a sparse matrix in pytorch with csr format.

    Parameters
    ----------
    M : A sparse matrix generated by FEniCS through fe.assemble()

    Returns
    -------
    The sparse matrix of type torch.sparse_csr_tensor

    More information on torch.spare: https://pytorch.org/docs/stable/sparse.html
    '''
    row, col, val = fe.as_backend_type(M).data()
    return torch.sparse_csr_tensor(row, col, val, dtype=dtype)


def spnumpy2sptorch(M, dtype=torch.float32, device="cpu"):
    '''
    The following codes are found in the webset: 
    https://www.jianshu.com/p/eb10322be38b
    '''
    # sparse_mx = M.tocoo().astype(np.float32)
    sparse_mx = M.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    # values = torch.from_numpy(sparse_mx.data)
    values = torch.tensor(sparse_mx.data, dtype=dtype)
    shape = torch.Size(sparse_mx.shape)
    if device == "cuda":
        values = values.cuda()
        indices = indices.cuda()
    M = torch.sparse.FloatTensor(indices, values, shape)
    return M.to_sparse_csr()


def sptorch2spnumpy(M, dtype=np.float64):
    row = np.array(M.crow_indices()).astype(np.int64)
    col = np.array(M.col_indices()).astype(np.int64)
    value = np.array(M.values(), dtype=dtype)
    return sps.csr_matrix((value, col, row))


def sptensor2cuda(M):
    row = M.crow_indices().cuda()
    col = M.col_indices().cuda()
    value = M.values().cuda()
    return torch.sparse_csr_tensor(row, col, value)


#########################################################################
def construct_measurement_matrix(xs, V):
    '''
    This function generate measurement matrix 
    xs: measurement points
    V:  function space generated by FEniCS
    Example: 
    Let V be a function space generated by FEniCS
    u is a function genrated by FEniCS based on function space V
    points = np.array([(0,0), (0.5,0.5)])
    S = construct_measurement_matrix(ponits, V)
    S@u.vector()[:] is a vector (u[0, 0], u[0.5, 0.5])
    '''
    nx, dim = xs.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    dolfin_element = V.dolfin_element()
    dofmap = V.dofmap()
    bbt = mesh.bounding_box_tree()
    sdim = dolfin_element.space_dimension()
    v = np.zeros(sdim)
    rows = np.zeros(nx*sdim, dtype='int')
    cols = np.zeros(nx*sdim, dtype='int')
    vals = np.zeros(nx*sdim)
    for k in range(nx):
        # Loop over all interpolation points
        x = xs[k, :]
        if dim > 1:
            p = fe.Point(x[0], x[1])
        elif dim == 1:
            p = fe.Point(x)
        # Find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)
        # Vertex coordinates for the cell
        xvert = coords[cells[cell_id, :], :]
        # Evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x, xvert, cell_id)
        jj = np.arange(sdim*k, sdim*(k+1))
        rows[jj] = k
        # Find the dofs for the cell
        cols[jj] = dofmap.cell_dofs(cell_id)
        vals[jj] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    M = sps.csr_matrix((vals, ij), shape=(nx, V.dim()))
    return M


############################################################################
def save_expre(filename, contents):
    fh = open(filename, 'w')
    fh.write(contents)
    fh.close()

############################################################################


def load_expre(filename):
    fh = open(filename, 'r')
    a = fh.read()
    fh.close()
    return a

##########################################################################
# the project command in FEniCS seems has some bug, it occupy memory and make the program out of memory!


def my_project(fun, V=None, flag='only_vec'):
    if V is None:
        V = fun.function_space()
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    a = fe.assemble(fe.inner(u, v)*fe.dx)
    b = fe.assemble(fe.inner(fun, v)*fe.dx)
    A = trans2spnumpy(a)
    sol = fe.Function(V)
    sol.vector()[:] = spsl.spsolve(A, b[:])
    if flag == 'only_vec':
        return sol.vector()[:]
    elif flag == 'only_fun':
        return sol
    else:
        return (sol, sol.vector()[:])

# the above function may also lead to memory problem


class MY_Project(object):
    def __init__(self, V):
        self.V = V
        self.u = fe.TrialFunction(V)
        self.v = fe.TestFunction(V)
        A_ = fe.assemble(fe.inner(self.u, self.v)*fe.dx)
        self.A = trans2spnumpy(A_)

    def project(self, fun):
        b_ = fe.assemble(fe.inner(fun, self.v)*fe.dx)
        return spsl.spsolve(self.A, b_[:])

############################################################################


def make_symmetrize(A):
    return 0.5*(A.T + A)

############################################################################


def smoothing(fun, bc=None, alpha=0.1):
    V = fun.function_space()
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    alpha = fe.Constant(str(alpha))
    A = fe.assemble(
        (alpha*fe.inner(fe.grad(u), fe.grad(v)) + fe.inner(u, v))*fe.dx)
    b = fe.assemble(fun*v*fe.dx)
    if bc is not None:
        bc.apply(A, b)
    sol = fe.Function(V)
    fe.solve(A, sol.vector(), b)
    return sol

###########################################################################


def relative_error(domain, u, u_truth):
    if type(u) is np.ndarray and type(u_truth) is np.ndarray:
        a = fe.Function(domain.function_space)
        b = fe.Function(domain.function_space)
        a.vector()[:], b.vector()[:] = u, u_truth
    else:
        u = fe.interpolate(u, domain.function_space)
        u_truth = fe.interpolate(u_truth, domain.function_space)
        a, b = u, u_truth
    fenzi = fe.assemble(fe.inner(a-b, a-b)*fe.dx)
    fenmu = fe.assemble(fe.inner(b, b)*fe.dx)
    return fenzi/fenmu

############################################################################


def print_my(*string, end=None, color=None):
    if color == 'red':
        print('\033[1;31m', end='')
        if end == None:
            print(*string)
        else:
            print(*string, end=end)
        print('\033[0m', end='')
    elif color == None:
        print(*string, end=end)
    elif color == 'blue':
        print('\033[1;34m', end='')
        if end == None:
            print(*string)
        else:
            print(*string, end=end)
        print('\033[0m', end='')
    elif color == 'green':
        print('\033[1;32m', end='')
        if end == None:
            print(*string)
        else:
            print(*string, end=end)
        print('\033[0m', end='')

############################################################################


def generate_points(x, y):
    '''
    If x = (x_1,x_2,\cdots, x_n),  y = (y_1,y_2,\cdots, y_m), 
    then points = gene_points(x, y) will be 
    ((x_1, y_1), (x_1, y_2), \cdots, (x_1, y_m), (x_2, y_1), \cdots (x_2, y_m), 
    \cdots, (x_n, y_m)))
    '''
    points = []
    for xx in x:
        for yy in y:
            points.append((xx, yy))
    return np.array(points)


############################################################################
def eval_min_max(x):
    len_x = len(x)
    min_val, max_val = min(x[0]), max(x[0])
    for itr in range(1, len_x):
        min_ = min(x[itr])
        max_ = max(x[itr])
        if min_ < min_val:
            min_val = min_
        if max_ > max_val:
            max_val = max_

    return min_val, max_val
