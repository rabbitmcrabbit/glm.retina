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



"""
==============
Helper classes
==============
"""

class ConvergedUpToFtol( Exception ):

    """ Exception class: when fitting successful up to desired tolerance. """

    pass


class Announcer( AutoReloader ):

    """ Class for managing verbose output. """

    def __init__( self, callers_to_suppress=None, verbose=True ):
        """ Create Announcer, for managing verbose output.

        Optional keywords:

        - `callers_to_suppress` : list of calling function names whose 
            announcements are suppressed. These can be further changed with the 
            `suppress` and `allow` methods.

        - `verbose` : whether to allow announcements (True)

        """
        if callers_to_suppress is None:
            callers_to_suppress = []
        self.callers_to_suppress = set(callers_to_suppress)
        self.verbose = verbose
        
    def suppress( self, *func_names ):
        """ Ignore further announcements from each `func_name`. """
        for func_name in func_names:
            self.callers_to_suppress.add( func_name )

    def allow( self, *func_names ):
        """ Allow announcements from each `func_name`. """
        for func_name in func_names:
            try:
                self.callers_to_suppress.remove( func_name )
            except KeyError:
                pass

    def thresh_allow( self, verbose, val, *func_names ):
        """ verbose >= val: allow. verbose < val: suppress. None => pass. """
        if verbose is None:
            return
        elif verbose >= val:
            self.allow( *func_names )
        elif verbose < val:
            self.suppress( *func_names )
        else:
            raise ValueError('invalid `verbose` value')

    def get_caller_name( self, n_steps=0 ):
        """ Return the name of the calling function, `n_steps` back. """
        f = sys._getframe()
        for i in range( n_steps + 3 ):
            f = f.f_back
        return f.f_code.co_name

    def announce( self, string, prefix=None, **kw ):
        """ Print `string` to stdout, with prefix.

        If prefix is not provided, this defaults to the name of the 
        calling function. If prefix == '', then no prefixing is set.

        """
        # whether the output is suppressed
        if not self.verbose:
            return
        # prefix
        if prefix is None:
            prefix = self.get_caller_name( **kw )
        if prefix in self.callers_to_suppress:
            return
        # create announcement
        if not prefix == '':
            prefix = '[%s]: ' % prefix
        msg = '%s%s\n' % ( prefix, string )
        # print out
        sys.stdout.write( msg )
        sys.stdout.flush()


class GridSearcher( AutoReloader ):

    """ Class for grid search through hyperparameter settings. """

    def __init__( self, bounds, spacing, initial, strategy, grid_size ):
        # parse input
        self.strategy = strategy
        self.grid_size = grid_size
        # check inputs
        self.N_dims = N_dims = len(bounds)
        if not ( N_dims == len(grid_size) ):
            raise ValueError('bounds and grid_size do not match')
        if N_dims > 2:
            pass
            #raise NotImplementedError('only for 1/2D grids at the moment')
        if np.min( grid_size ) <= 0:
            raise ValueError('grid size must be at least 1 on each axis')
        if strategy not in ['1D', '1D one pass', '1D two pass', 
                '1D three pass', 'full']:
            raise ValueError("strategy must be one of " + 
                    " '1D' / '1D one pass' / '1D two pass' / " + 
                    "'1D three pass' / 'full'" )
        # total count
        if strategy == '1D':
            if not( N_dims == 1 ):
                raise ValueError("'1D' strategy only available for 1D grids")
            max_count = np.sum(grid_size)
        if strategy == '1D one pass':
            max_count = np.sum(grid_size) - len(grid_size) + 1
        elif strategy == '1D two pass':
            max_count = 2 * (np.sum(grid_size) - len(grid_size)) + 1
        elif strategy == '1D three pass':
            max_count = 3 * (np.sum(grid_size) - len(grid_size)) + 1
        elif strategy == 'full':
            max_count = np.prod(grid_size)
        self.max_count = max_count
        # calculate ordinate values
        self.ordinates = ordinates = []
        for i in range(N_dims):
            s = grid_size[i]
            b = bounds[i]
            if spacing[i] == 'linear':
                pts = np.linspace( b[0], b[1], s )
            elif spacing[i] == 'log':
                pts = np.logspace( np.log10(b[0]), np.log10(b[1]), s )
            else:
                raise ValueError('spacing must be log or linear')
            ordinates.append( pts )
        # initial start point
        self.start_idx = start_idx = zeros( N_dims, dtype=int )
        for i in range(N_dims):
            if spacing[i] == 'linear':
                d = np.abs( ordinates[i] - initial[i] )
            elif spacing[i] == 'log':
                d = np.abs( np.log10(ordinates[i]) - np.log10(initial[i]) )
            start_idx[i] = np.argmin(d)
        # construct executed dictionary
        self.evidence_values = {}
        self.is_finished = False
        self.start()

    @property
    def current_count( self ):
        return len( self.evidence_values )

    def start( self ):
        # grid size is only 1D
        if self.N_dims == 1:
            self.stack = [(i,) for i in np.arange( self.grid_size[0] )]
        # 1D: create stack
        elif self.strategy in ['1D one pass', '1D two pass', '1D three pass', 
                '1D'] and self.N_dims == 2:
            self.axis_passes = [(self.start_idx[0], None)]
            self.stack = [(self.start_idx[0], i) 
                    for i in np.arange( self.grid_size[1] )]
        # full: create stack
        elif self.strategy == 'full':
            self.stack = zip(*[i.flatten() for i in np.meshgrid( *[
                np.arange(s) for s in self.grid_size], indexing='ij' )])
        # 1D in high dimensions
        elif self.N_dims > 2 and self.strategy.startswith('1D'):
            axis_pass = list(self.start_idx)
            axis_pass[-1] = None
            self.axis_passes = [ tuple(axis_pass) ]
            self.stack = [ tuple( axis_pass[:-1] + [i]) 
                    for i in np.arange(self.grid_size[-1]) ]
        else:
            raise ValueError('unknown strategy')

    @property
    def current_idx( self ):
        return self.stack[0]

    @property
    def current_theta( self ):
        idx = self.current_idx
        return [ self.ordinates[i][idx[i]] for i in range(len(idx)) ]

    def next( self, evidence ):
        # save evidence
        if evidence is not None:
            self.evidence_values[ self.current_idx ] = evidence
        # pop off the stack
        self.stack = self.stack[1:]
        # if the stack is empty
        if len( self.stack ) == 0:
            if self.N_dims == 1:
                self.is_finished = True
            elif self.strategy == 'full':
                self.is_finished = True
            elif ( self.strategy == '1D one pass' 
                    and len(self.axis_passes) == self.N_dims ):
                self.is_finished = True
            elif ( self.strategy == '1D two pass' 
                    and len(self.axis_passes) == (self.N_dims * 2) ):
                self.is_finished = True
            elif ( self.strategy == '1D three pass' 
                    and len(self.axis_passes) == (self.N_dims * 3) ):
                self.is_finished = True
            elif self.N_dims <= 2:
                last_axis_pass = self.axis_passes[-1]
                if last_axis_pass[0] is None:
                    # find best row
                    evs = A([ 
                        self.evidence_values[ (i, last_axis_pass[1])] 
                        for i in range(self.grid_size[0]) ])
                    row_idx = np.argmax( evs )
                    self.axis_passes += [(row_idx, None)]
                    # construct next column
                    self.stack = [(row_idx, i) 
                            for i in np.arange( self.grid_size[1] )]
                else:
                    # find best column
                    evs = A([ 
                        self.evidence_values[ (last_axis_pass[0], j )] 
                        for j in range(self.grid_size[1]) ])
                    col_idx = np.argmax( evs )
                    self.axis_passes += [(None, col_idx)]
                    # construct next row
                    self.stack = [(i, col_idx) 
                            for i in np.arange( self.grid_size[0] )]
            else:
                last_axis_pass = self.axis_passes[-1]
                axis_idx = [ i for i, a in enumerate(last_axis_pass) 
                        if a == None ][0]
                last_axis_list = []
                for i in range( self.grid_size[axis_idx] ):
                    a = A( last_axis_pass )
                    a[ axis_idx ] = i
                    last_axis_list.append( tuple(a) )
                evs = A([ self.evidence_values[a] for a in last_axis_list ])
                new_pos = np.argmax(evs)
                next_axis_pass = [a for a in last_axis_pass]
                next_axis_pass[ axis_idx ] = new_pos
                axis_idx -= 1
                if axis_idx < 0:
                    axis_idx = self.N_dims - 1
                next_axis_pass[ axis_idx ] = None
                self.axis_passes += [ tuple(next_axis_pass) ]
                for i in np.arange( self.grid_size[ axis_idx ] ):
                    next_axis_pass[ axis_idx ] = i
                    self.stack.append( tuple(next_axis_pass) )
        # check if the top one has been done
        if len( self.stack ) > 0:
            if self.evidence_values.has_key( self.current_idx ):
                self.next( None )

