# META-LEARNING
硕士毕业项目

## 记录
### 2023.8.5
    meta_common.py - Gaussian1DFiniteDifference
                  |- GaussianElliptic2Learn

#### class scipy.sparse.coo_matrix
A sparse matrix in coordinate format.

##### function
1. coo_matrix(arg1, shape=None, dtype=None, copy=False) 
   
    构造函数
    > https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix
2. to_csr() 
   
    合并相同坐标上的值
    > https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html#scipy.sparse.coo_matrix.tocsr

**example**
```python 
nx, ny = (5, 5)
rows = np.arange(nx)
cols = rows
a_fun = 2.0 * np.ones(nx)
matrix = sps.coo_matrix((a_fun, (rows, cols)), shape=[nx, ny])
print(matrix)
```
