# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Mathieu Blondel, Tom Dupre la Tour
# License: BSD 3 clause

cimport cython
cimport openmp
import numpy as np
import multiprocessing
#from numpy import linalg as LA
from libc.math cimport fabs, sqrt
from cython.parallel import parallel, prange

@cython.boundscheck(False)
def _update_FISTA_fast(double[:, ::1] W, double[:, ::1] dW, 
                      double[:, :] HHt, double[:, :] XHt,
                      double[::1] old_diagonal,
                      double[::1] theta,
                      Py_ssize_t[::1] permutation):
					  
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef double grad, y_slice, pg, hess, beta, old_theta
    cdef Py_ssize_t i, r, s, t, num_threads


    for s in range(n_components):
            t = permutation[s]
		    # Hessian
            hess = HHt[t,t]

            # update diagnal & root
            a = old_diagonal[t]/hess * theta[t]
            old_diagonal[t] = hess

            # update theta & beta
            old_theta = theta[t]
            theta[t] = ( -a + sqrt(a*a + 4*a) ) /2
            beta = theta[t] * ( 1/old_theta - 1 )

            with nogil, parallel(num_threads=4):            
              for i in prange(n_samples):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]
                y_slice = W[i,t] + beta * dW[i,t]

                for r in prange(n_components):
                    grad += HHt[t, r] * W[i, r]
                

                # projected gradient
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)

                if hess != 0:
                    W[i, t] = max(y_slice - grad / hess, 0.)


    #old_diagonal = diagonal
 
    return violation
