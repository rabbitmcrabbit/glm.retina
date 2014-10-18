from __future__ import division

import cPickle
import os
import sys
import weakref
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from numpy import fft, shape, zeros, ones, eye, dot, exp, log, sqrt, diag
from numpy.linalg import inv, det, norm, eigh
from numpy.random import normal, multivariate_normal, poisson, random_integers
from numpy.fft import rfft, irfft, fftfreq

from scipy.linalg import block_diag
from scipy.io import loadmat
from scipy.optimize import fmin_l_bfgs_b, fmin_ncg

try:
    __IPYTHON__
    from IPython.core.debugger import Tracer 
    tracer = Tracer() 
except NameError:
    pass

import progress
from autocr import AutoReloader, AutoCacherAndReloader, cached
import bunch

A = np.array
na = np.newaxis
conc = np.concatenate
ls = os.listdir
join = os.path.join

np.seterr(all='ignore')
plt.rcParams['image.interpolation'] = 'nearest'


"""
================
Helper functions
================
"""

def logdet(X):
    """ Log determinant of a matrix. """
    return np.linalg.slogdet(X)[1]

def ldiv(A, b):
    """ Left division of matrix A by b; i.e. A \ b in matlab notation. """
    return np.linalg.solve(A, b)

def mdot(*a):
    """ Dot product of all vectors/matrices. 
    
    >>> mdot( X, Y, Z )
    gives X.Y.Z where matrices are sizes X(a, b), Y(b, c), Z(c, d).
    
    """
    return reduce( lambda x, y: dot(x, y), a )

def maxabs( x ):
    """ Max of the absolute values of elements of an array. """
    if len(x) == 0:
        return 0
    return np.max(np.abs(x))

def is_even( N ):
    """ Boolean: is `N` an even integer? """
    return ( int(N // 2) * 2 == N )

def nanmean(x):
    return np.mean( x[~np.isnan(x)] )

def nanstd(x):
    return np.std( x[~np.isnan(x)] )



"""
================
Helper classes
================
"""

class Bunch( bunch.Bunch ):

    """ Extends Bunch, but only prints keys. Useful for data exploration."""

    def __repr__(self):
        k = self.keys()
        k.sort()
        return k.__repr__().replace('[', '<').replace(']', '>')

    def copy(self):
        #return Bunch(super(Bunch, self).copy())
        return Bunch( self.toDict().copy() )

    @property
    def __shapes__(self):
        max_len_k = max([len(str(k)) for k in self.keys()])
        for k in np.sort(self.keys()):
            k_str = str(k)
            k_str += ' ' * (max_len_k - len(k_str)) + ' : '
            k_str += str(shape(self[k]))
            print '    ', k_str

    @property
    def __summary__(self):
        max_len_k = max([len(str(k)) for k in self.keys()])
        for k in np.sort(self.keys()):
            k_str = str(k)
            k_str += ' ' * (max_len_k - len(k_str)) + ' : '
            v = self[k]
            if np.iterable(v):
                k_str += str(shape(v)) + ' array'
            else:
                k_str += str(v)
            print '    ', k_str
