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
        k_stim_true = self.k_stim_true = self.make_k_stim_true()
        k_const_true = self.k_const_true = self.make_k_const_true()
        k_true = self.k_true = np.concatenate([ k_stim_true, k_const_true ])
        D = self.D = len( k_true )
        # stimulus and response
        X_stim = self.make_X_stim( X_std=X_std )
        z = dot( X_stim, k_stim_true )
        if add_constant:
            z += k_const_true
        if nonlinearity == 'exp':
            mu = exp( z )
        elif nonlinearity == 'soft':
            mu = log( 1 + exp(z) ) / log(2)
        else:
            raise ValueError( 'unknown nonlinearity: %s' % nonlinearity )
        y = poisson( mu )
        # continue initialising
        super( SimulatedData, self ).__init__( 
                X_stim, y, add_constant=add_constant, **kw )

    def make_X_stim( self, X_std ):
        """ Generate stimulus matrix for simulation. """
        return normal( size = (self.T, self.D-1) ) * X_std

    def make_k_stim_true( self ):
        """ Generate stimulus-response relationship for simulation. """
        # dimensions
        D = 101
        # create kernel
        i = np.arange(D, dtype=float)
        k_true = ( exp( -((i - D // 2) / 12.5)** 2) 
                * cos(2 * pi * i / 20) * 0.25 )
        return k_true

    def make_k_const_true( self ):
        return A([0.5])
    



class SimulatedDataHistory( SimulatedData ):

    """ Simulated data, with spike history kernel. """ 

    def __init__( self, fs=100, N_sec=60, X_std=1, add_constant=True, 
            nonlinearity='exp', **kw ):
        # save parameters
        self.fs = fs
        self.N_sec = N_sec
        self.T = np.floor( self.fs * self.N_sec )
        # create the kernel        
        self.k_stim_true = k_stim = self.make_k_stim_true()
        self.k_history_true = k_history = self.make_k_history_true()
        self.k_const_true = k_const = self.make_k_const_true()
        self.k_true = k_true = np.concatenate([ k_stim, k_history, k_const ])
        # lengths
        self.D_stim = D_stim = len(k_stim)
        self.D_history = D_history = len(k_history)
        # stimulus driven firing rate
        X_stim = self.make_X_stim( X_std=X_std )
        z = dot( X_stim, k_stim )
        if nonlinearity == 'exp':
            mu = exp( z )
        elif nonlinearity == 'soft':
            mu = log( 1 + exp(z) ) / log(2)
        else:
            raise ValueError( 'unknown nonlinearity: %s' % nonlinearity )
        if add_constant:
            z += k_const
        # generate spikes (very inefficient)
        y = np.zeros_like(mu)
        for t in range( len(y) ):
            if t == 0:
                mu_history = 1.
            elif t <= self.D_history:
                prev_spikes = y[ : t ]
                mu_history = exp( dot( prev_spikes, k_history[ -t: ] ) )
            else:
                prev_spikes = y[ t - self.D_history : t ]
                mu_history = exp(dot( prev_spikes, k_history ))
            y[t] = poisson( mu[t] * mu_history )
        # construct X_history
        X_history = glm.construct_X_history( y, D_history )
        # construct X
        X = glm.combine_X( X_stim, X_history )
        # continue initialising
        super( SimulatedData, self ).__init__( 
                X, y, add_constant=add_constant, **kw )

    def make_k_history_true( self ):
        """ Generate history kernel for stimulation. """
        tt = np.arange(40.)
        k_history = -( np.sin(2 * np.pi * tt / (6 * np.sqrt(tt + 1))) 
                * np.exp(-0.5 * (tt/12.)**2 ) ) * 0.01
        k_history = k_history[::-1]
        return k_history



class SimulatedData2D( SimulatedData ):

    """ Simulated data, with 2D kernel. """ 
    
    def make_k_stim_true( self ):
        """ Generate stimulus-response relationship for simulation. """
        # dimensions
        self.D_x = D_x = 10
        self.D_y = D_y = 10
        self.D = D = D_x * D_y + 1 
        # create kernel
        i, j = np.meshgrid( np.arange(self.D_x), np.arange(self.D_y), 
                indexing='ij' )
        k_stim_true_2D = np.exp(-0.5 * ((j - 3) ** 2 + (i - 5) ** 2))
        return k_stim_true_2D.flatten()


class SimulatedData2DHistory( SimulatedDataHistory, SimulatedData2D ):

    """ Simulated data, with 2D kernel and history. """

    pass
