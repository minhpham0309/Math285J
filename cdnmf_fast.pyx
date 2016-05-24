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
from libc.math cimport fabs
from cython.parallel import parallel, prange

@cython.boundscheck(False)
def _update_cdnmf_fast(double[:, ::1] W, double[:, :] HHt, double[:, :] XHt,
                       Py_ssize_t[::1] permutation):
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef double grad, pg, hess
    cdef Py_ssize_t i, r, s, t, num_threads

    #cdef double[:] W_slice = np.zeros[n_samples]
    #cdef double[:] gradient = np.zeros[n_samples]

    #with nogil, parallel(num_threads=4):
        #num_threads = openmp.omp_get_num_threads()
    for s in range(n_components):
            t = permutation[s]
		    # Hessian
            hess = HHt[t, t]

            with nogil, parallel(num_threads=4):            
              for i in prange(n_samples):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in prange(n_components):
                    grad += HHt[t, r] * W[i, r]
                

                # projected gradient
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)

                if hess != 0:
                    W[i, t] = max(W[i, t] - grad / hess, 0.)

    #with nogil:
        #num_threads = openmp.omp_get_num_threads()
    #    for s in prange(n_components, schedule='static'):
    #        t = permutation[s]
    #        W_slice = W(:,t);
    #        signW = sign(W(:,t));

		    # Hessian
    #        hess = HHt[t, t]
            
            # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
    #        gradient = -XHt[:, t] + np.dot(W,HHt[t,:])



            # projected gradient
    #        pg = signW.*gradient +  (1-signW).* min(0., gradient);
    #        norm_pg = LA.norm(pg,1);
            
    #        if hess != 0:
    #            W(:,t) = max(W_slice - coef * gradient/hess,0.);
					
    return violation
