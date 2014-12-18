from common import *
from numpy import cos, pi

import glm


class SimulatedData( glm.Data ):

    """ Simulated dataset, for testing Solvers. """

    def __init__( self, fs=100, N_sec=60, 
            X_std=1, add_constant=True, nonlinearity='exp', **kw ):
        """ Create a simulated data set. 
        
        Creates a dataset produced with a fixed 1D kernel. This can then be 
        fitted with various Solvers.

        Size of the dataset:

        - `fs` : sampling frequency, i.e. bins per second
        - `N_sec` : number of seconds

        The keyword `nonlinearity` can be 'exp' or 'soft'.
            
        Keywords for `k`:

        - `X_std` : standard deviation of X. `k` is fixed, so this determines
            how much the stimulus-driven factor fluctuates from sample to 
            sample.
        - `add_constant` : whether to include a constant term in X

        """
        # save parameters
        self.fs = fs
        self.N_sec = N_sec
        self.T = np.floor( self.fs * self.N_sec )
        # create the kernel
        k_true = self.k_true = self.make_k_true()
        D = self.D = len( k_true )
        # stimulus and response
        X = self.make_X( X_std=X_std )
        if add_constant:
            z = dot( X, k_true[:-1] )
        else:
            z = dot( X, k_true )
        if nonlinearity == 'exp':
            mu = exp( z )
        elif nonlinearity == 'soft':
            mu = log( 1 + exp(z) ) / log(2)
        else:
            raise ValueError( 'unknown nonlinearity: %s' % nonlinearity )
        y = poisson( mu )
        # continue initialising
        super( SimulatedData, self ).__init__( 
                X, y, add_constant=add_constant, **kw )

    def make_X( self, X_std ):
        """ Generate stimulus matrix for simulation. """
        return normal( size = (self.T, self.D-1) ) * X_std

    def make_k_true( self ):
        """ Generate stimulus-response relationship for simulation. """
        # dimensions
        D = 101
        # create kernel
        i = np.arange(D, dtype=float)
        k_true = ( exp( -((i - D // 2) / 12.5)** 2) 
                * cos(2 * pi * i / 20) * 0.25 )
        return k_true


