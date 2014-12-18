from common import *


"""
=============
Data class
=============
"""

class Data( AutoReloader ):

    """ Data structure, for a conditionally-Poisson response model. """

    def __init__( self, X__td, y__t, 
            normalise=False, whiten=False, add_constant=True, 
            announcer=None, verbose=False ):
        """ Construct a Data object. 
        
        Provide the stimulus `X__td`, which is size ( T x D ).
        Provide the spike count response `y__t`, which is size ( T ).

        Optional keywords:

        - `normalise` : transform each of the D dimensions of `X__td` to have
            zero mean and unitary standard deviation

        - `whiten` : transform so that each of the D dimensions of `X__td` has
            zero mean, and X^T . X = I

        - `add_constant` : append an extra dimension to `X__td` which is 
            constant. 

        Verbose output:

        - `announcer` : provide Announcer object. If not provided, this is
            constructed.

        - `verbose` : initialise announcer to be verbose (True)

        """
        # create announcer
        if announcer is None:
            announcer = Announcer( verbose=verbose )
        self._announcer = announcer
        # normalise, if required
        X = X__td
        T, D = self.T, self.D = shape( X )
        if normalise or whiten:
            self.announce('normalising', prefix='Data')
            Xm = np.mean( X, axis=0 )
            Xs = np.std( X, axis=0 )
            if np.min( Xs ) == 0:
                raise ValueError('`X` contains degenerate dimension')
            X = ( X - Xm[na, :] ) / Xs[na, :]
            self.normalisation_filter = Bunch({ 'X_mean': Xm, 'X_std': Xs })
        # whiten, if required
        if whiten:
            self.announce('whitening', prefix='Data')
            L, Q = eigh(dot(X.T, X))
            X = mdot( X, Q, diag(L ** -0.5), Q.T )
            self.whiten_filter = Bunch({ 'L':L, 'Q':Q })
        # add constant, if required
        if add_constant:
            X = np.hstack([ X, ones((T, 1)) ])
        # note these
        self.X_is_normalised = normalise
        self.X_is_whitened = whiten
        self.X_has_constant = add_constant
        # save the data
        self.X__td, self.y__t = X, y__t
        # sizes
        T, D = self.T, self.D = shape( self.X__td )
        if len( y__t ) != self.T:
            raise ValueError('# time bins in `X__td` and `y__t` do not match')

    def announce( self, *a, **kw ):
        return self._announcer.announce( *a, **kw )

    @property
    def X( self ):
        return self.X__td

    @property
    def y( self ):
        return self.y__t

    """ Plotting """

    def plot_y( self, mu=None, binsize_sec=None, 
            draw=True, new_figure=True, return_ax=False, return_fig=False, 
            **kw ):
        """ Plot the spike counts over time.

        This will also show the estimated spike counts from the posterior,
        if this is provided.

        Plotting keywords:

        - `posterior`
        - `binsize_sec` : Timescale over which to integrate spike counts

        Additional plotting keywords:

        - `draw`: Whether to call `plt.draw` at the end
        - `new_figure`: Whether to create a new figure instance
        - `return_ax`: Whether to return the axis object
        - `return_fig` : Whether to return the figure object

        """
        # create figure
        if new_figure:
            plt.figure(**kw)
        ax = plt.gca()
        # bin
        y = self.y__t
        if binsize_sec is None:
            i = np.arange( self.T )
        else:
            steps_per_block = int(np.round(self.fs * binsize_sec))
            T = len(y)
            n_blocks = int( np.floor( T // steps_per_block ) )
            T = int( n_blocks * steps_per_block )
            y = y[:T].reshape( n_blocks, steps_per_block )
            y = np.mean( y, axis=1 )
            i = np.arange( n_blocks )
            maxy = np.max(y)
        ax.plot(i, y, 'b', lw=1)
        # prediction
        if mu is not None:
            if binsize_sec is not None:
                y_hat = mu[:T].reshape( n_blocks, steps_per_block )
                y_hat = np.mean( y_hat, axis=1 )
            else:
                y_hat = mu
            ax.plot(i, y_hat, 'r', lw=3, alpha=0.7)
            maxy = np.max([y, y_hat])
        else:
            maxy = np.max(y)
        # aesthetics
        ax.set_ylim(0, maxy * 1.1)
        ax.set_xlim(0, len(y) - 1)
        ax.set_xticks([0, len(y)])
        ax.set_xticklabels([0, int(self.N_sec)])
        ax.set_yticklabels([])
        ax.set_xlabel('time (sec)', fontsize=14)
        # return
        if draw:
            plt.draw()
            plt.show()
        if return_ax:
            return ax
        if return_fig:
            return fig


"""
=============
Solver class
=============
"""

class Solver( AutoCacherAndReloader ):

    """ Superclass for GLM solvers.

    This should be subclassed to define certain structural priors over
    the variables.

    """

    # CONSTANTS
    # for dimensionality reduction: where to truncate the spectrum
    cutoff_lambda = 1e-12
    ftol_per_obs = 1e-5

    """ 
    ---------------
    Initialisation
    ---------------
    """

    def __init__( self, data, solve=False, nonlinearity='exp',
            testing_proportion=0, initial_conditions=None, 
            announcer=None, verbose=True, empty_copy=False, **kw ):
        """ Create a Solver object, for given data object, `data`.
        
        Keywords:

        - `solve`: whether to immediate solve (default: False)

        - `nonlinearity` : the output nonlinearity. Either 'exp' or 'soft'.
            * 'exp'  :  mu__t = exp( z__t )
            * 'soft' :  mu__t = log_2( 1 + exp( z__t ) )

        - `testing_proportion` : what fraction of the data should be set aside
            for cross-validation (between 0 and 1)

        - `initial_conditions`: initial values for parameters and/or 
            hyperparameters. Can be a dict or a Solver object.


        Verbosity:

        - `announcer` : provide Announcer object (constructed if not provided)
        - `verbose` : whether there are verbose announcements


        Other:

        - `empty_copy` : internal use. Helper for creating posterior objects.

        """
        # create announcer object. This outputs solving progress to the screen.
        if announcer is None:
            announcer = Announcer( verbose=verbose )
            announcer.suppress('cache')
        self._announcer = announcer
        # empty copy? (for posterior)
        if empty_copy:
            return
        # save data
        self._data = data
        # define training and testing regions of the data
        self.define_training_and_testing_regions( 
                testing_proportion=testing_proportion, **kw )
        # nonlinearity
        if nonlinearity in ['exp', 'soft']:
            self.nonlinearity = nonlinearity
        else:
            raise ValueError("`nonlinearity` must be 'exp' or 'soft'")
        # parse initial conditions
        self.parse_initial_conditions( initial_conditions )
        self.reset()
        # solve, if requested
        if solve:
            self.solve()

    """ Useful properties """

    @property
    def N_theta( self ):
        """ Number of hyperparameters. """
        return len( self.hyperparameter_names )

    @property
    def N_observations( self ):
        """ How many observed data points in the training set. """
        return self.T_training

    @cached
    def T( data ):
        """ Number of time bins. """
        return data.T

    @cached
    def D( data ):
        """ Dimensionality of the stimulus. """
        return data.D

    """ Announcements """

    def announce(self, *a, **kw):
        """ Verbose print. Sends call to the `self._announcer` object. """
        return self._announcer.announce(*a, **kw)

    def _caching_callback( self, name=None, arguments=None ):
        """ Callback method when a cached variable is computed. """
        return self.announce( 'computing %s' % name, prefix='cache' )

    """ Saving """

    def __getstate__( self ):
        """ For pickling, remove `data` and some baggage. """
        d = super( Solver, self ).__getstate__()
        # re-add the settable cache attributes
        for k in self._cache.keys():
            if getattr( self.__class__, k ).settable:
                d['_cache'][k] = self._cache[k]
        # get rid of undesirables
        if self.is_posterior:
            d['parent'] = None
        keys_to_del = ['_data']
        for k in keys_to_del:
            if d.has_key(k):
                del d[k]
        return d

    def __setstate__( self, d ):
        # create weak references
        for k in [k for k in d.keys() if k.startswith('posterior_')]:
            d[k].parent = weakref.proxy( self )
        if d.has_key('_reducer'):
            d['_reducer'].parent = weakref.proxy( self )
        # save
        self.__dict__ = d

    """ Plotting """

    def plot_y( self, posterior=None, **kw ):
        """ Plot spike counts. See `Data.plot_y` for docstring. 
        
        By default, this plots `self.posterior`, unless an alternative
        posterior is provided.
        
        """
        if posterior is None:
            posterior = self.posterior

        return self.data.plot_y( posterior, **kw )

    """
    ===================
    Initial conditions
    ===================

    An object may be supplied with initial conditions for the parameters (`k`)
    and/or hyperparameters (`theta`). These methods define how to parse these 
    initial conditions, or what `k` and `theta` need to default to when 
    nothing is provided.

    """

    def parse_initial_conditions( self, initial_conditions ):
        """ Sets up the initial conditions for the model. """
        # initialise `k` and `theta` to global defaults
        self.initial_conditions = ics = Bunch()
        ics['k__d'] = zeros( self.D )
        ics['theta'] = self.default_theta0
        # parse initial conditions: None
        if initial_conditions == None:
            pass
        # parse initial_conditions: dict
        elif isinstance( initial_conditions, dict ):
            for k in ics.keys():
                ics[k] = initial_conditions.get( k, ics[k] )
        # parse initial_conditions: Solver object
        else: 
            # copy posterior
            if hasattr( initial_conditions, 'posterior' ):
                ics['k__d'] = initial_conditions.posterior.k__d
            # recast values of `theta`
            try:
                ics['theta'] = self.recast_theta( initial_conditions )
            except TypeError:
                pass
        # replace any invalid values
        for i in range( self.N_theta ):
            if (ics['theta'][i] is None) or (ics['theta'][i] == np.nan):
                ics['theta'] = self.default_theta0[i]
    
    def recast_theta( self, ic ):
        """ Process results of previous Solver to determine initial `theta`.

        When provided with an initial condition `ic` which is another Solver
        object, this extracts `ic.theta` and reconfigures it to be compatible
        with the current class. 

        Here, if initial conditions come from the same Prior class, then
        inherit the fitted theta. Any other behaviour requires subclassing.

        Arguments:
        - `ic` : previously solved version of model
        
        """
        c1 = ic.Prior_class
        c2 = self.Prior_class
        if c1 == c2:
            return ic.posterior.theta
        else:
            err_str = 'subclass %s to define how to recast theta from %s to %s '
            err_str = err_str % ( c2.__name__, c1.__name__, c2.__name__ )
            raise TypeError( err_str )

    def Prior_class( self ):
        """ Returns the superclass of `self` that defines the prior covariance. 

        This checks the method resolution order, and finds the definition of
        the covariance matrix `C__dd`, or its diagonal `l__d`.
        
        """
        return [ c for c in self.__class__.mro() 
                if c.__dict__.has_key('l__d') 
                or c.__dict__.has_key('C__dd') ][ 0 ]

    """
    ==========================
    Resetting `v` and `theta`
    ==========================

    At initialisation, and sometimes afterwards, we need to reset the
    parameter vector (`v`), and the hyperparameters (`theta`). The `reset`
    methods wipe the current values, and retrieve the values from the saved 
    posterior object (if available), else the initial conditions. The 
    `initialise` methods only do this if there are no current values of `v` 
    and/or `theta` (or if these are not valid, e.g. are of the wrong 
    dimensionality.)

    """

    def initialise_theta( self ):
        """ If `theta` is not set, set it from initial conditions. """
        if hasattr( self, 'theta' ):
            return 
        elif hasattr( self, 'posterior' ):
            self.theta = self.posterior.theta
        else:
            self.theta = self.initial_conditions.theta

    def initialise_v( self ):
        """ If `v` is not set (validly), set from initial conditions. """
        # check current value is a valid size
        if hasattr( self, 'v' ):
            v = self.v
            if len( v ) == self.required_v_length:
                return
            else:
                delattr( self, v )
        # recast `v` from posterior or initial conditions
        if hasattr( self, 'posterior' ):
            self.v = self.reproject_to_v( posterior=self.posterior )
        else:
            self.v = self.reproject_to_v( posterior=self.initial_conditions )
    
    def reset_theta( self ):
        """ Force reset of `theta`. """
        if hasattr( self, 'theta' ):
            delattr( self, 'theta' )
        self.initialise_theta()

    def reset_v( self ):
        """ Force reset of `v`. """
        if hasattr( self, 'v' ):
            delattr( self, 'v' )
        self.initialise_v()

    def reset( self ):
        """ Sets the values of parameters to the initial conditions. """
        ic = self.initial_conditions
        # delete any posteriors
        if hasattr( self, 'posterior' ):
            delattr( self, 'posterior' )
        # reset hyperparameters
        self.reset_theta()
        # reset latent variables
        self.reset_v()
        # initialise posterior
        p = self.posterior = self.create_posterior( theta=self.theta, v=self.v )
        p.is_point_estimate = True

    """
    =============
    Optimisation
    =============

    Optimisation involves calculating the posterior distribution on `k` given 
    the data and the hyperparameters (`theta`):

        P( k | data, theta )

    Here, we use a Laplace approximation. First, we solve for the posterior 
    mode (stored in `v` / `k_star` / `k`). Then we approximate the posterior 
    covariance from the curvature of the log likelihood around the mode (stored 
    in `Lambda`).

    When the posterior is calculated, it is saved in `self.posterior`. This 
    attribute is an object of the same class as `self`, but has a flag
    `is_posterior` set to True. Also the cache of the posterior object is 
    immutable, to prevent accidental changes.

    There is also the question of how to choose the hyperparameters, `theta`. 
    This is determined by calculating the marginal likelihood (i.e. the 
    evidence) of the data given `theta`, and then optimising this value for 
    `theta`. This is done in two stages: first a coarse grid search, then a 
    gradient ascent. For the best value of `theta`, the posterior is retained
    as `self.posterior`.

    """

    def solve( self, grid_search_theta=True, verbose=1, **kw ):
        """ Solve for all hyperparameters and parameters. 

        First solves for the hyperparameters `theta`. Then solves for
        the posterior on `k` given the max marginal likelihood values of 
        `theta`.

        On the first iteration of the solver, a grid search is run on `theta`
        if `grid_search_theta` is True. If this is done, it will override the
        current value of `theta`. 
        
        Keywords:

        - `grid_search_theta` : boolean (True). Run a grid search on `theta`
        first.

        Verbosity levels:

        - 0 : print nothing
        - 1 : print evidence steps only
        - 2 : print evidence and posterior steps
        - None : change nothing

        """
        # verbosity
        self._announcer.thresh_allow( verbose, 1, 'solve', 'solve_theta' )
        self._announcer.thresh_allow( verbose, 2, 'calc_posterior' )
        # get initial posterior 
        self.announce('Initial posterior')
        self.calc_posterior( verbose=None )
        # extract starting points for first cycle
        self.reset_v()
        # solve for theta
        self.announce( 'solving for theta' )
        self.solve_theta( verbose=None, grid_search=grid_search_theta, **kw )
        # ensure no more grid searches are run
        grid_search_theta = False
        # restore verbosity
        self._announcer.allow( 
                'solve', 'solve_theta', 'calc_posterior' )

    """ 
    ------------------
    Posterior objects 
    ------------------
    """

    # by default, `self` is not a posterior, unless specified otherwise
    is_posterior = False

    def create_posterior( self, **kw ):
        """ Create a posterior object of the same class as self. 

        This will place a copy of the current cache in the posterior, and
        set the flag `is_posterior` to be `True`. Any keywords provided will
        also be set as attributes; this will happen *first* so that it
        does not wipe the cache.
        
        The posterior object will contain weak references to the parent.
        
        """
        # create weak references
        data = weakref.proxy( self.data )
        announcer = weakref.proxy( self._announcer )
        parent = weakref.proxy( self )
        # create object
        p = self.__class__( data=None, empty_copy=True, announcer=announcer )
        # label it as a posterior object
        p.is_posterior = True
        p.parent = parent
        # note the training
        p.training_slices = self.training_slices
        p.testing_slices = self.testing_slices
        # copy the nonlinearity
        p.nonlinearity = self.nonlinearity 
        # set any remaining attributes
        for k, v in kw.items():
            setattr(p, k, v)
        # copy the current cache
        return p

    @property
    def data( self ):
        """ Retain a single shared copy of the data object. """
        if hasattr( self, '_data' ):
            return self._data
        else:
            return self.parent.data

    def _clear_descendants( self, name ):
        """ Posteriors should be immutable. """
        if self.is_posterior:
            for d in self._descendants.get( name, () ):
                if self._cache.has_key( d ):
                    raise TypeError('cannot change the cache of a posterior')
        else:
            return super( Solver, self )._clear_descendants( name )

    """
    --------------------------------------
    Solving for the parameter vector, `v`
    --------------------------------------
    """
    
    def calc_posterior( self, xtol=None, ftol_per_obs=None, verbose=1 ):
        """ Calculate posterior on `v`, via gradient descent. 

        Tolerance is given in terms of two quantities:
        
        - `xtol` : the desired precision in the value of `v`. If this is 
        not provided, it defaults to to `self.xtol`, which might be defined 
        in the class. Otherwise this defaults to 1e-8.  

        - `ftol_per_obs` : the desired precision in the log posterior, per
        observation (i.e. per time bin). If this is not provided, it defaults 
        to `self.ftol_per_obs`, which must be defined in the class. 

        """
        # parse tolerance
        if xtol is None:
            if hasattr( self, 'xtol' ):
                xtol = self.xtol
            else:
                xtol = 1e-8
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        ftol = ftol_per_obs * self.N_observations
        # parse verbosity
        self._announcer.thresh_allow( verbose, 1, 'calc_posterior' )
        # functions to minimise
        # (these are interface functions to the cacher)
        eq = np.array_equal
        f = self.cfunction( eq, 'negLP_objective', 'v' )
        df = self.cfunction( eq, 'negLP_jacobian', 'v' )
        d2f = self.cfunction( eq, 'negLP_hessian', 'v' )
        # initialise starting value of v
        self.initialise_v()
        # reporting during each step
        def callback( v, assess_convergence=True ):
            # set the current value of `v` (if required)
            self.csetattr( 'v', v )
            # run the callback function
            self.negLP_callback( prefix='calc_posterior' )
            # have we converged
            last_negLP = self._last_negLP
            this_negLP = self.negLP_objective
            improvement = -(this_negLP - last_negLP)
            # if we have converged, break out of the loop
            if assess_convergence and (improvement < ftol):
                raise ConvergedUpToFtol()
            self._last_negLP = this_negLP
        # initial condition
        v0 = self.v
        # if pre-optimisation is available, run it
        preoptim_func = '_preoptimise_v_for_LP_objective'
        if hasattr( self, preoptim_func ):
            preoptim_func = getattr( self, preoptim_func )
            v0 = preoptim_func( v0 )
        # initial announcement
        self._last_negLP = np.inf
        callback( v0, assess_convergence=False )
        # solve
        v_hat = v0
        last_LP = -f(v_hat)
        if not len(v_hat) == 0:
            try:
                v_hat = fmin_ncg( f, v_hat, df, fhess=d2f, 
                        disp=False, callback=callback, avextol=xtol )
            except ConvergedUpToFtol:
                v_hat = self.v
        # create posterior object
        self.create_and_save_posterior( v_hat )
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 'calc_posterior' )

    def create_and_save_posterior( self, v_hat ):
        # ensure we are set to the optimum
        self.csetattr( 'v', v_hat )
        # create posterior object
        p = self.posterior = self.create_posterior( theta=self.theta, v=v_hat )
        # again, ensure that this posterior is set to the posterior mode
        p.csetattr( 'v', v_hat )

    """
    ----------------------------------------------
    Solving for the hyperparameter vector, `theta`
    ----------------------------------------------
    """

    def solve_theta( self, grid_search=False, ftol_per_obs=None, 
            max_iterations=10, verbose=1, **kw ):
        """ Optimise for `theta` by maximum marginal likelihood. 
        
        Convergence is measured by `ftol_per_obs`. This is multiplied by
        `self.N_observations` to give a step improvement criterion for 
        continuing the optimisation. This defaults to `self.ftol_per_obs`
        if not provided.
        
        """
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # do we actually have a theta to solve
        if self.N_theta == 0:
            return
        # parse verbosity
        self._announcer.thresh_allow( verbose, 1, 'solve_theta' )
        self._announcer.thresh_allow( verbose, 2, 'calc_posterior' )
        # initialise
        self.initialise_theta()
        best_evidence = -np.inf
        best_posterior = None
        # calculate posterior
        self.calc_posterior( verbose=None, **kw )
        # current value
        if hasattr(self, 'posterior' ):
            try:
                best_posterior = self.posterior
                best_evidence = self.posterior.evidence
                announce_str = 'evidence existing:     %.3f' % best_evidence
                self.announce( announce_str, prefix='solve_theta' )
            except (AttributeError, np.linalg.LinAlgError):
                pass
        # check that we can actually grid search
        grid_search = ( grid_search and self.grid_search_theta_available )

        # case 1: set initial conditions by grid search
        if grid_search:
            # announce
            self.announce( 'theta grid search starting', prefix='solve_theta' )
            # construct grid searching object, `gs`
            grid_kws = self.grid_search_theta_parameters.copy()
            grid_kws.update( **kw )
            gs = GridSearcher( **grid_kws )
            # run through the grid search
            while not gs.is_finished:
                # current `theta` value
                theta = gs.current_theta
                self.theta = theta
                # set the initial value of `v`
                if best_posterior is not None:
                    self.v = self.reproject_to_v( posterior=best_posterior )
                # calculate posterior on `v` for this `theta`
                self.calc_posterior( verbose=None, **kw )
                p = self.posterior
                # evaluate the evidence for this `theta`
                this_evidence = self.posterior.evidence
                # prepare announcement
                announce_str = 'grid search [%d/%d]: ('
                announce_str = announce_str % (gs.current_count, gs.max_count)
                announce_str += ('%d,' * len(theta))
                announce_str = announce_str[:-1] + ')'
                announce_str = announce_str % tuple(theta)
                max_len = 30 + 2*len(theta)
                if len(announce_str) < max_len:
                    announce_str += ' '*(max_len - len(announce_str))
                announce_str += ('= %.1f' % this_evidence)
                # if the evidence is better, keep
                if this_evidence > best_evidence:
                    best_posterior = p
                    best_evidence = this_evidence
                    announce_str += '    *'
                # announce
                self.announce( announce_str, prefix='solve_theta' )
                # continue
                gs.next( this_evidence )
            # we are finished grid search
            self.announce( 'theta grid search finished', prefix='solve_theta' )

        # case 2: initial condition is current `theta`
        elif ( hasattr( self, 'posterior' ) and 
                np.array_equal( self.posterior.theta, self.theta ) ):
            pass

        # case 3: calculate
        else:
            self.calc_posterior( verbose=None, **kw )
            best_posterior = self.posterior
            best_evidence = best_posterior.evidence
                
        # announce the evidence
        announce_str = 'evidence initial:     %.3f' % best_evidence
        self.announce( announce_str, prefix='solve_theta' )
        # save the posterior
        self.posterior = best_posterior

        # solve: cycle of maximising local LE
        for i in range( max_iterations ):
            # optimise local evidence
            self.get_next_theta_n()
            new_theta = self.theta
            # check that theta has changed
            old_theta = self.posterior.theta
            if np.array_equal( new_theta, old_theta ):
                break
            # calculate posterior here
            self.calc_posterior( verbose=None, **kw )
            # evaluate evidence here
            new_posterior = self.posterior
            new_evidence = self.posterior.evidence
            # report progress
            announce_str = 'evidence iteration %d: %.3f' % ( i, new_evidence )
            self.announce( announce_str, prefix='solve_theta' )
            # if we have made an improvement, keep it
            evidence_improvement = new_evidence - best_evidence
            if evidence_improvement > 0:
                best_posterior = new_posterior
                best_evidence = new_evidence
            # if the improvement has been negative, restore, then end
            if evidence_improvement < 0:
                self.posterior = best_posterior
                new_theta = best_posterior.theta
                new_v = best_posterior.v
                self.csetattr( 'theta', new_theta )
                self.csetattr( 'v', new_v )
                break
            # if the improvement has been too small, end here
            if evidence_improvement < (ftol_per_obs * self.N_observations):
                self.posterior = best_posterior
                break

        # announce the evidence
        announce_str = 'evidence final:       %.3f' % best_evidence
        self.announce( announce_str, prefix='solve_theta' )
        if best_evidence < -1e10:
            tracer()
        # restore verbosity
        self._announcer.thresh_allow( 
                verbose, -np.inf, 'solve_theta', 'calc_posterior' )
    
    def get_next_theta_n( self, factr=1e10, ftol_per_obs=None ):
        """ Single step for local evidence approx algorithm.

        In the Park & Pillow (2012) approximation, one walks towards an
        optimal `theta` by approximating the objective function near
        `theta_n` as `psi(theta)`, then maximise this approx objective
        function. The solution becomes the next `theta_n`. This method
        performs one step in this optimisation procedure, finding, in effect
        `theta_(n+1)` from `theta_(n)`.

        Keywords:

        - `factr` : convergence factor for 'grad'-based optimisation

        """
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        ftol = ftol_per_obs * self.N_observations
        # functions to minimise
        # (these are interface functions to the cacher)
        f = self.cfunction( np.array_equal, 'LE_objective', 'theta' )
        df = self.cfunction( np.array_equal, 'LE_jacobian', 'theta' )
        # a copy is necessary as fmin_l_bfgs_b makes changes in place
        g = lambda x: f( x.copy() )
        dg = lambda x: df( x.copy() )
        # reporting during each step
        def callback( theta, assess_convergence=True ):
            # set the current value of `theta` (if required)
            self.csetattr( 'theta', theta )
            # have we converged
            last_LE = self._last_LE
            this_LE = getattr( self, 'LE_objective' )
            improvement = -(this_LE - last_LE)
            # if we have converged, break out of the loop
            if assess_convergence and (improvement < ftol):
                raise ConvergedUpToFtol()
            self._last_LE = this_LE
        # initial condition
        theta0 = np.array( self.posterior.theta )
        # boundary conditions
        bounds = self.bounds_theta
        self._last_LE = g( theta0 ) #np.inf
        # run
        try:
            theta = fmin_l_bfgs_b( 
                    g, theta0, fprime=dg, approx_grad=False, bounds=bounds, 
                    factr=factr, disp=False, callback=callback )[0]
        except ConvergedUpToFtol:
            theta = self.theta
        # save
        g(theta)

    @property
    def grid_search_theta_available( self ):
        """ Can we grid search to find `theta`. """
        return hasattr( self, 'grid_search_theta_parameters' )

    """
    ================
    Cross-validation
    ================

    This section contains methods for cross-validating. 
    
    At initialisation, the data are split up into training and testing 
    datasets. These are currently implemented as slices: e.g., for a 100 sec
    recording, the training dataset comprises the contiguous regions from 
    0-8, 10-18, 20-28 secs, ... while the testing dataset comprises the
    remaining contiguous regions from 8-10, 18-20, 28-30 secs, ... This is
    just an implementation detail at present, and can be changed so long as
    the methods `has_testing_regions`, `T_training`, `T_testing`, 
    `slice_by_training`, `slice_by_testing`, `zero_during_testing` are
    replaced.

    """
    
    def define_training_and_testing_regions( 
            self, testing_proportion, testing_block_size_smp=100, **kw ):
        """ Partitions the data for cross-validation. 
        
        Arguments:

        - `testing_proportion` : how much of the data to put aside for
            testing purposes. Should be between 0 and 0.4 (recommended: 0.2)

        - `testing_block_size_smp` : max duration of the contiguous blocks in
            the testing data set. This is defined in samples.
        
        """
        # what to do if no cross-validation
        if testing_proportion == 0:
            self.training_slices = [ slice(0, self.T, None ) ]
            self.testing_slices = []
            return
        # check input
        if not 0 <= testing_proportion <= 0.4:
            raise ValueError('testing_proportion should be between 0 and 0.4')
        # how many testing regions
        T = self.T
        testing_block_size_smp = int( testing_block_size_smp )
        testing_total_smp = int( np.ceil( T * testing_proportion ) )
        if testing_block_size_smp > testing_total_smp:
            testing_block_size_smp = testing_total_smp
        N_testing = int( np.ceil( 
            testing_total_smp / float(testing_block_size_smp) ) )
        # how long is each testing region
        testing_lengths = ones(N_testing, dtype=int) * testing_block_size_smp
        testing_lengths[-1] -= ( np.sum(testing_lengths) - testing_total_smp  )
        testing_lengths = np.random.permutation( testing_lengths )
        # construct minimal training regions
        N_training = N_testing
        training_lengths = zeros(N_training, dtype=int)
        training_lengths += testing_block_size_smp # minimum size
        # random assignment of remaining time to training regions
        remaining_smps = T - np.sum(training_lengths) - testing_total_smp
        if remaining_smps < 0:
            raise ValueError(
                    'cannot fit training and testing blocks in, using ' +
                    'these parameters')
        proportions = np.random.exponential( size=N_training )
        proportions /= np.sum(proportions)
        training_lengths += np.round( proportions * remaining_smps ).astype(int)
        remaining_smps = T - np.sum(training_lengths) - testing_total_smp
        if remaining_smps > 0:
            idx = random_integers(
                    low=0, high=N_training-1, size=remaining_smps)
            for i in idx:
                training_lengths[i] += 1
        elif remaining_smps < 0:
            idx = random_integers(
                    low=0, high=N_training-1, size=-remaining_smps)
            for i in idx:
                training_lengths[i] -= 1
        # define regions
        N_blocks = N_training + N_testing 
        block_lengths = zeros(N_blocks, dtype=int)
        block_is_training = zeros(N_blocks, dtype=bool)
        block_lengths[0::2] = training_lengths
        block_lengths[1::2] = testing_lengths
        block_is_training[0::2] = True
        block_star__et_idx = np.cumsum(np.hstack([[0], block_lengths]))[:-1]
        block_end_idx = np.cumsum(block_lengths)
        # create slices
        block_slices = A([ slice(idx_i, idx_j, None) 
                for idx_i, idx_j in zip(block_star__et_idx, block_end_idx) ])
        training_slices = block_slices[ block_is_training ]
        testing_slices = block_slices[ ~block_is_training ]
        # randomly rotate
        offset = random_integers( low=0, high=T-1 )
        training_slices = [ 
                slice( (s.start + offset) % T , (s.stop + offset) % T, None )
                for s in training_slices ]
        testing_slices = [ 
                slice( (s.start + offset) % T, (s.stop + offset) % T, None )
                for s in testing_slices ]
        for s in training_slices:
            if s.stop < s.start:
                training_slices.remove(s)
                training_slices += [
                        slice(s.start, T, None), slice(0, s.stop, None)]
        for s in testing_slices:
            if s.stop < s.start:
                testing_slices.remove(s)
                testing_slices += [
                        slice(s.start, T, None), slice(0, s.stop, None)]
        training_slices.sort()
        testing_slices.sort()
        # save
        self.training_slices = training_slices
        self.testing_slices = testing_slices

    @cached
    def has_testing_regions( testing_slices ):
        """ Returns True if there are testing regions. """
        return len( testing_slices ) > 0

    @cached
    def T_training( training_slices ):
        """ Number of training data points. """
        return int(sum([ s.stop - s.start for s in training_slices ]))

    @cached
    def T_testing( testing_slices ):
        """ Number of testing data points. """
        return int(sum([ s.stop - s.start for s in testing_slices ]))

    def slice_by_training( self, sig ):
        """ Retrieves the training slices of a signal. 
        
        It is assumed that the 0th dimension in the signal is time.
        
        """
        if not self.has_testing_regions:
            return sig
        new_shape = A(sig.shape)
        new_shape[0] = self.T_training
        new_sig = zeros( new_shape, dtype=sig.dtype )
        count = 0
        for s in self.training_slices:
            s_len = s.stop - s.start
            new_sig[ count : count + s_len, ... ] = sig[ s, ... ]
            count += s_len
        return new_sig

    def slice_by_testing( self, sig ):
        """ Retrieves the testing slices of a signal. 
        
        It is assumed that the 0th dimension in the signal is time.

        """
        if not self.has_testing_regions:
            return A([])
        new_shape = A(sig.shape)
        new_shape[0] = self.T_testing
        new_sig = zeros( new_shape, dtype=sig.dtype )
        count = 0
        for s in self.testing_slices:
            s_len = s.stop - s.start
            new_sig[ count : count + s_len, ... ] = sig[ s, ... ]
            count += s_len
        return new_sig

    def zero_during_testing( self, sig ):
        """ Sets the signal components during testing regions to be zero. 
        
        It is assumed that the 0th dimension in the signal is time.
        
        """
        if not self.has_testing_regions:
            return sig
        sig = sig.copy()
        for s in self.testing_slices:
            sig[ s, ... ] = 0
        return sig

    @cached
    def y_testing__t( data, slice_by_testing ):
        """ Spike counts for the testing dataset. """
        return slice_by_testing( data.y__t )

    @cached
    def LL_testing( slice_by_testing, mu__t, log_mu__t, y_testing__t ):
        """ Log likelihood on the testing dataset. """
        mu__t = slice_by_testing(mu__t)
        log_mu__t = slice_by_testing(log_mu__t)
        return -np.sum( mu__t ) + dot( y_testing__t, log_mu__t )

    @cached
    def LL_training_per_observation( LL_training, T_training ):
        """ Log likelihood per observation on the training dataset. """
        return LL_training / T_training

    @cached
    def LL_testing_per_observation( LL_testing, T_testing ):
        """ Log likelihood per observation on the testing dataset. """
        return LL_testing / T_testing

    """
    ===================
    Derivative checking
    ===================

    These are methods to ensure that the provided analytical derivatives
    are correct, by comparing them with empirically-calculated derivatives.

    """
    
    def check_LP_derivatives( self, eps=1e-6, debug=False, 
            error_if_fail=False, error_threshold=0.05 ):
        """ Check derivatives of the log posterior.

        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective attributes `negLP_jacobian` and 
        `negLP_hessian`, and compares with empirical values, obtained 
        from finite differences method on `negLP_objective` and 
        `negLP_jacobian` respectively.

        Typically, this prints out the size of the error between analytic
        and empirical estimates, as a norm. For example, for the Jacobian 
        (analytic `aJ` and empirical `eJ` respectively), this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        This tests the derivatives at the current value of the 
        hyperparameters `theta` and the parameters `v`.
        
        Keywords on testing procedure:

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.
        
        """
        # check that we are not at a posterior mode
        if hasattr( self, 'posterior' ):
            p = self.posterior
            eq = np.array_equal
            same_theta = eq( self.theta, p.theta )
            same_v = eq( self.v, p.v )
            if same_theta and same_v:
                err_str = ( 'LP checking cannot run at a posterior mode; ' +
                        'try changing `v` to something else' )
                raise ValueError(err_str)
        # initial conditions
        v0 = self.v
        N_v = len( v0 )
        # helper function: a step along coordinate `i`
        def dv(i):
            z = zeros( N_v )
            z[i] = eps
            return z
        # *** JACOBIAN ***
        # evaluate analytic jacobian
        self.v = v0
        LP = self.negLP_objective
        dLP = self.negLP_jacobian
        # evaluate empirical jacobian
        edLP = np.zeros( N_v )
        for i in progress.dots( range(N_v), 'empirical jacobian' ):
            self.v = v0 + dv(i)
            edLP[i] = ( self.negLP_objective - LP ) / eps
        # restore `v`
        self.v = v0
        # print
        err1 = norm( dLP - edLP ) / norm( dLP )
        print ' '
        print 'dLP norm deviation : %.6f' % err1
        print ' '
        # raise error?
        if error_if_fail and (err1 > error_threshold):
            raise ValueError('Jacobian of LP failed at %.6f' % err1 )
        # HESSIAN
        # evaluate analytic
        d2LP = self.negLP_hessian
        # evaluate empirical
        def empirical_d2LP(i):
            self.v = v0 + dv(i)
            return ( self.negLP_jacobian - dLP ) / eps
        ed2LP = A([ empirical_d2LP(i) 
            for i in progress.dots( range(N_v), 'empirical hessian' ) ])
        # restore
        self.v = v0
        # print
        err2 = norm( d2LP - ed2LP ) / norm( d2LP )
        print ' '
        print 'd2LP norm deviation: %.6f' % err2
        print ' '
        # raise error?
        if error_if_fail and (err2 > error_threshold):
            raise ValueError('Hessian of LP failed at %.6f' % err2 )
        # debug
        if debug:
            tracer()

    def check_LE_derivatives( self, eps=1e-6, debug=False, 
            error_if_fail=False, error_threshold=0.05, **kw ):
        """ Check derivatives of the local evidence wrt `theta`.
        
        For the Jacobian, evaluates the analytic derivatives provided by
        the attribute `LE_jacobian` and compares these with the empirical
        values, obtained from finite differences method on `LE_objective`.

        Typically, this prints out the size of the error between analytic (aJ)
        and empirical estimates (eJ), as a norm. In particular, this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        Keywords:

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.
        
        """
        N_theta = self.N_theta
        # evaluate analytic
        LE = self.LE_objective
        a = dLE = self.LE_jacobian
        # evaluate empirical
        theta0 = np.array( self.theta ).astype(float)
        e = edLE = np.zeros( N_theta )
        for i in range( N_theta ):
            theta = theta0.copy()
            theta[i] += eps
            self.theta = theta
            edLE[i] = ( self.LE_objective - LE) / eps
        # check Jacobian
        a[ np.abs(a) < 1e-20 ] = 1e-20
        e[ np.abs(e) < 1e-20 ] = 1e-20
        err1 = np.nanmax( np.abs( (a-e)/e ) )
        err2 = norm( a - e ) / norm( e )
        print 'LE_jacobian : '
        print ' '
        print '   max deviation : %.6f ' % err1
        print '   norm deviation : %.6f ' % err2
        print ' '
        sys.stdout.flush()
        # raise error?
        if error_if_fail and (err2 > error_threshold):
            raise ValueError('Jacobian of LE failed at %.6f' % err2 )
        # debug
        if debug:
            tracer()

    def check_l_derivatives( self, eps=1e-6, debug=False,
            error_if_fail=False, error_threshold=0.05, hessian=False ):
        """ Check derivatives of diagonal of prior covariance wrt `theta`. 
        
        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective attributes `dl_dtheta__id` and 
        `d2l_dtheta2__iid`, and compares with empirical values, obtained from 
        finite differences method on `l__d` and `dl_dtheta__id` respectively.
        
        Typically, this prints out the size of the error between analytic
        and empirical estimates, as a norm. For example, for the Jacobian 
        (analytic `aJ` and empirical `eJ` respectively), this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        This tests the derivatives at the current value of `theta`.

        Keywords:

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.

        - `hessian` : boolean. whether to check hessian

        NOTE: Hessian testing not currently implemented.

        """
        # can only run this if C is diagonal
        if not self.C_is_diagonal:
            raise TypeError('`C` is not diagonal: check `l` derivs instead.')
        # initial condition
        theta0 = self.theta
        N_theta = self.N_theta
        D = self.D
        # calculate l
        l = self.l__d
        # JACOBIAN
        # analytic Jacobian
        dl = A( self.dl_dtheta__id )
        # helper function
        def dth(i):
            z = zeros( N_theta )
            z[i] = eps
            return z
        # empirical Jacobian
        edl = np.zeros( (N_theta, D) )
        for i in range(N_theta):
            self.theta = theta0 + dth(i)
            edl[i] = ( self.l__d - l ) / eps
        # print error
        err1 = norm( dl - edl ) / norm( dl )
        print ' '
        print 'dl norm deviation: %.6f' % err1
        print ' '
        # raise error?
        if error_if_fail and (err1 > error_threshold):
            raise ValueError('Jacobian of l failed at %.6f' % err1 )
        # HESSIAN
        if hessian:
            raise NotImplementedError()
        # debug
        if debug:
            tracer()

    def check_C_derivatives( self, eps=1e-6, debug=False,
            error_if_fail=False, error_threshold=0.05, hessian=False ):
        """ Check derivatives of the prior covariance matrix wrt `theta`. 
        
        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective methods `dC_dtheta__idd` and 
        `d2C_dtheta2__iidd`, and compares with empirical values, obtained from 
        finite differences method on `C__dd` and `dC_dtheta__iidd` respectively.
        
        Typically, this prints out the size of the error between analytic
        and empirical estimates, as a norm. For example, for the Jacobian 
        (analytic `aJ` and empirical `eJ` respectively), this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        This tests the derivatives at the current value of `theta`.

        Keywords:

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.

        - `hessian` : whether to test the same for the hessian

        NOTE: Hessian testing not currently implemented.
        
        """
        # can only run this if C is not diagonal
        if self.C_is_diagonal:
            raise TypeError('`C` is diagonal: check `L` derivatives instead')
        # initial condition
        theta0 = self.theta
        N_theta = self.N_theta
        D = self.D
        # calculate C
        C = self.C__dd
        # JACOBIAN
        # analytic Jacobian
        dC = A( self.dC_dtheta__idd )
        # helper function
        def dth(i):
            z = zeros( N_theta )
            z[i] = eps
            return z
        # empirical Jacobian
        edC = np.zeros( (N_theta, D, D) )
        for i in range(N_theta):
            self.theta = theta0 + dth(i)
            edC[i] = ( self.C__idd - C ) / eps
        # print error
        err1 = norm( dC - edC ) / norm( dC )
        print ' '
        print 'dC norm deviation: %.6f' % err1
        print ' '
        # raise error?
        if error_if_fail and (err1 > error_threshold):
            raise ValueError('Jacobian of C failed at %.6f' % err1 )
        # HESSIAN
        if hessian:
            raise NotImplementedError()
        # debug
        if debug:
            tracer()

    

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for the parameter vector, `v`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    The goal is produce a posterior over the weights, `k`. When we put a prior
    over `k`, this might suppress some dimensions on `k` enough that it
    becomes worthwhile to project out these dimensions entirely. The 
    optimisation then operates over a reduced-dimensionality vector, `k_star`.

    We use the symbol `v` to refer to the vector that is being directly 
    optimised. Generally, `v = k_star`.

    In what follows, we define the components of the objective function over
    `v`, and then how to optimise. This will typically be a convex objective.

    Also note that this section makes extensive use of cached variables.

    """

    @cached
    def k_star__e( v ):
        """ Extract the dim-reduced `k` from `v`. Can change in subclasses. """
        return v

    """
    ================
    Log Prior on `k`
    ================

    We assume that the prior on `k` is a zero-mean multivariate Gaussian, in 
    which case we need to specify its (D x D) covariance matrix, `C__dd`. 
    This covariance matrix will be a function of the hyperparameters, `theta`. 
    In the most general form, `C__dd` will be an arbitrary covariance matrix; 
    however, in some models it might have specific structure. For notational 
    simplicity, we distinguish here between two forms: where it is a diagonal 
    matrix, in which case the diagonal vector is referred to as `l__d`; and 
    where it is a non-diagonal matrix, in which case it is referred to as 
    `C__dd`. In this latter case, `l__d` is still used as the diagonal in the 
    eigendecomposition of `C__dd`.
    
    The definition of `C__dd` or `l__d` must be given in a subclass. If `l__d` 
    is given, then the attribute `C_is_diagonal` must be set to False. In this
    case, it is assumed that the parametric form of the covariance matrix
    ensures it is *always* diagonal.

    The forms of the prior covariance matrix (and its derivatives wrt `theta`)
    must be defined in subclasses. If they are not defined, accessing these
    attributes will throw a TypeError.
    
    """

    def _raise_must_subclass( self, f_str ):
        """ Helper function """
        raise TypeError( 'must subclass to define `%s` behaviour' % f_str )

    @cached
    def C__dd( C_is_diagonal ):
        """ Prior covariance on `k`. """
        if C_is_diagonal:
            return None
        else:
            self._raise_must_subclass('C__dd')
        
    @cached
    def dC_dtheta__idd():
        """ Jacobian of prior covariance wrt `theta`. """
        self._raise_must_subclass('dC_dtheta')

    @cached
    def C_eig( C__d, C_is_diagonal ):
        """ Eigendecomposition of the prior covariance matrix. """
        if C_is_diagonal:
            return None
        try:
            l, Q = eigh( C )
        except np.linalg.LinAlgError:
            l, Q = eigh( C * 1e4 )
            l /= 1e4
        return {'l__d':l, 'Q__dd':Q}

    @cached( cskip = ('C_is_diagonal', True, 'C_eig') )
    def l__d( C_is_diagonal, C_eig ):
        """ Return the diagonal covariance (in the rotated space if req'd) """
        if C_is_diagonal:
            self._raise_must_subclass('l__d')
        else:
            return C_eig['l__d']
       
    @cached
    def dl_dtheta__id():
        """ Jacobian of the diagonal prior covariance wrt `theta`. """
        self._raise_must_subclass('dl_dtheta')

    @cached
    def dims( l__d, cutoff_lambda ):
        """ Boolean vector of dimensions to keep, from the eigenspectrum. 
        
        If any eigenvalues of the prior covariance are sufficiently small,
        the corresponding eigenvectors are projected out.
        
        """
        return ( l__d/maxabs(l__d) > cutoff_lambda )

    @cached
    def D_star( dims ):
        """ How many dimensions in the reduced space. """
        return np.sum( dims )

    @cached
    def required_v_length( D_star ):
        """ Number of parameters in `v`, based on `len(k_star__e)`. """
        return D_star

    @cached
    def R__de( C_eig, dims, C_is_diagonal ):
        """ Rotation and projection matrix: *-space to full-space. """
        return C_eig['Q'][ :, dims ].T

    @cached( cskip=('C_is_diagonal', True, ['C_eig', 'R__de', 'D_star']) )
    def R_is_identity( C_is_diagonal, C_eig, dims, R__de, D_star ):
        """ Is the rotation/projection operator the identity. """
        if C_is_diagonal:
            return np.all(dims) 
        else:
            return ( np.all(dims) and np.array_equal(R__de, eye(D_star)) )

    @cached( cskip=[ 
        ('R_is_identity', True, ['dims', 'R__de']),
        ('C_is_diagonal', True, 'R__de') ] )
    def X_star__te( R_is_identity, C_is_diagonal, data, dims, R__de ):
        """ Dimensionality-reduced matrix of regressors. """ 
        if R_is_identity:
            return data.X__td
        elif C_is_diagonal:
            return data.X__td[:, dims]
        else:
            return dot( data.X__td, R__de.T )

    @cached
    def l_star__e( l__d, dims ):
        """ Diagonalised prior covariance in *-space. """
        return l__d[ dims ]

    @cached
    def l_star_inv__e( l_star__e ):
        """ Inverse of diagonal of prior covariance in reduced space. """
        return 1. / l_star__e

    @cached
    def logdet_C_star( l_star__e ):
        """ Log determinant of reduced covariance matrix. """
        return np.sum(log(l_star__e))

    @cached( cskip=[
        ('R_is_identity', True, ['dims', 'R__de', 'D']),
        ('C_is_diagonal', True, ['R__de']) ])
    def k__d( k_star__e, R_is_identity, C_is_diagonal, dims, R__de, D ):
        """ Expand the *-space kernel to the full kernel. """
        if R_is_identity:
            return k_star__e
        elif C_is_diagonal:
            k__d = zeros( D )
            k__d[ dims ] = k_star__e
            return k__d
        else:
            return dot( R__de.T, k_star__e )

    def reproject_to_v( self, k__d=None, posterior=None ):
        """ Calculates `v` from `k__d`. Does not change object's state.

        Given a weight vector, `k__d`, this rotates and projects it into
        the *-space induced by the current prior (specified by `theta`), and
        returns the result.
        
        Provide exactly one of `k__d` *or* a posterior containing `k__d`.
        
        """
        # check inputs
        if [k__d, posterior].count(None) != 1:
            raise ValueError('either provide `k__d` or `posterior`')
        # provide a posterior
        elif posterior is not None:
            return self.reproject_to_v( k__d=posterior.k__d )
        # provide a `k__d` value
        elif k__d is not None:
            if self.R_is_identity:
                return k__d
            elif self.C_is_diagonal:
                return k__d[ self.dims ]
            else:
                return dot( self.R__de.T, k__d )

    """
    ====================
    Log posterior on `k`
    ====================
    """

    def _raise_bad_nonlinearity( self ):
        raise ValueError('unknown nonlinearity: %s' % self.nonlinearity )

    @cached
    def y_training__t( data, slice_by_training ):
        """ The training data. """
        return slice_by_training( data.y__t )

    @cached
    def z__t( X_star__te, k_star__e ):
        """ Argument to the nonlinearity. """
        return dot( X_star__te, k_star__e )

    @cached
    def ez__t( z__t ):
        """ Intermediate computation. """
        return exp( z__t )

    @cached
    def mu__t( ez__t, nonlinearity ):
        """ Expected firing rate """
        if nonlinearity == 'exp':
            return ez__t
        elif nonlinearity == 'soft':
            return log( 1 + ez__t ) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached( cskip = ('nonlinearity', 'exp', 'mu__t') )
    def log_mu__t( nonlinearity, z__t, mu__t ):
        """ Log of expected firing rate """
        if nonlinearity == 'exp':
            return z__t
        else:
            return log( mu__t )

    @cached
    def LL_training( y_training__t, slice_by_training, mu__t, log_mu__t ):
        """ Log likelihood on `k` given training data. """
        # select the training data only
        mu__t = slice_by_training( mu__t )
        log_mu__t = slice_by_training( log_mu__t )
        # compute
        return -np.sum( mu__t ) + dot( y_training__t, log_mu__t )

    @cached
    def LPrior( logdet_C_star, k_star__e, l_star_inv__e ):
        """ Log prior on `k`. """
        return sum([
            -0.5 * logdet_C_star, 
            -0.5 * dot( k_star__e, l_star_inv__e * k_star__e ) ])

    @cached
    def LP_training( LL_training, LPrior ):
        """ Log posterior on `k` given training data. """
        return LL_training + LPrior

    """ Jacobian """

    @cached
    def resid__t( mu__t, data, zero_during_testing ):
        """ Residual spike counts. Zeroed outside of training data. """
        return zero_during_testing( data.y__t - mu__t )

    @cached
    def dF__t( nonlinearity, ez__t ):
        if nonlinearity == 'exp':
            return ez__t
        elif nonlinearity == 'soft':
            return ez__t / (1 + ez__t) / log(2)
        else:
            raise self._raise_bad_nonlinearity()

    @cached( cskip = ('nonlinearity', 'exp', ['dF__t', 'mu__t'] ) )
    def dF_on_F__t( nonlinearity, dF__t, mu__t ):
        if nonlinearity == 'exp':
            return 1
        else:
            return dF__t / mu__t

    @cached( cskip = ('nonlinearity', 'exp', 'dF_on_F__t') )
    def dLL_training( nonlinearity, resid__t, X_star__te, dF_on_F__t ):
        """ Jacobian of training log likelihood wrt `k_star__e`. """
        if nonlinearity == 'exp':
            return dot( X_star__te.T, resid__t )
        else:
            return dot( X_star__te.T, resid__t * dF_on_F__t )

    @cached
    def dLPrior( l_star_inv__e, k_star__e ):
        """ Jacobian of log prior wrt `k_star__e`. """
        return -l_star_inv__e * k_star__e

    @cached
    def dLP_training( dLL_training, dLPrior ):
        """ Jacobian of training log posterior wrt `k_star__e`. """
        return dLL_training + dLPrior

    """ Hessian """

    @cached
    def d2F__t( nonlinearity, ez__t ):
        if nonlinearity == 'exp':
            return ez__t
        elif nonlinearity == 'soft':
            return ez__t / ( (1 + ez__t)**2 ) / log(2)
        else:
            raise self._raise_bad_nonlinearity()

    @cached( cskip = 
            ('nonlinearity', 'exp', ['dF_on_F__t', 'resid__t', 'd2F__t']) )
    def d2LP_training( mu__t, D_star, slice_by_training, X_star__te, 
            l_star_inv__e, nonlinearity, dF_on_F__t, d2F__t, resid__t,
            y_training__t):
        """ Hessian of training log prob wrt `k_star__e`. """
        # training data only
        mu__t = slice_by_training( mu__t )
        X_star__te = slice_by_training( X_star__te )
        # evaluate d2LL
        if nonlinearity == 'exp':
            diag__t = -mu__t
        else:
            resid__t = slice_by_training( resid__t )
            d2F__t = slice_by_training( d2F__t )
            dF_on_F__t = slice_by_training( dF_on_F__t )
            diag__t = (
                    (resid__t * d2F__t / mu__t ) - 
                    (y_training__t * dF_on_F__t**2) )
        d2LP = dot( X_star__te.T, diag__t[:, na] * X_star__te )
        # add d2LPrior to the diagonal
        d2LP[ range(D_star), range(D_star) ] -= l_star_inv__e
        return d2LP
    
    """
    ================
    Solution for `k`
    ================
    """

    @cached
    def negLP_objective( LP_training ):
        return -LP_training

    @cached
    def negLP_jacobian( dLP_training ):
        return -dLP_training

    @cached
    def negLP_hessian( d2LP_training ):
        return -d2LP_training

    def negLP_callback( self, **kw ):
        self.announce( 'LP step:  %.3f' % self.LP_training, n_steps=4, **kw )

    """
    =============================
    Posterior specific attributes
    =============================

    A Laplace approximation is taken for the posterior. This is initially
    expressed in reduced-coordinate space (*-space) as 
        `Normal( k_star__e, Lambda_star__ee )`,
    and can be expanded to 
        `Normal( k__d, Lambda__dd )`. 
    Here, `Lambda__dd` is the posterior covariance.

    Note that these can only be calculated for a posterior, as they depend
    on knowing the posterior mode (`k_star__e`), and the curvature around
    the mode.

    """
    
    @cached( cskip = ('is_posterior', False, 'negLP_hessian') )
    def Lambdainv_star__ee( is_posterior, negLP_hessian ):
        """ Posterior precision, in reduced space. """
        if not is_posterior:
            raise TypeError('attribute only available for posterior.')
        return negLP_hessian

    @cached
    def Lambda_star__ee( Lambdainv_star__ee ):
        """ Posterior covariance, in reduced space. """
        return inv( Lambdainv_star__ee )

    @cached( cskip = [
        ('R_is_identity', True, ['R__de', 'D', 'dims']),
        ('C_is_diagonal', True, 'R__de') ])
    def Lambda__dd( Lambda_star__ee, R_is_identity, C_is_diagonal, dims, 
            R__de, D ):
        """ Posterior covariance, in full space. """
        # if there is no dimensionality reduction
        if R_is_identity:
            return Lambda_star__ee
        # if the dimensionality reduction is just a projection
        elif C_is_diagonal:
            Lambda__dd = zeros( (D, D) )
            dim_idxs = np.nonzero( dims )[0]
            for e, d in enumerate( dim_idxs ):
                Lambda__dd[ d, dims ] = Lambda_star__ee[ e, : ]
            return Lambda__dd
        # if the dimensionality reduction is rotation + projection
        else:
            return mdot( R__de, Lambda_star__ee, R__de.T )

    @cached
    def logdet_Lambdainv_star( Lambdainv_star__ee ):
        """ log |Lambda ^ -1|. For evidence calculation. """
        return logdet( Lambdainv_star__ee )

    @cached
    def evidence_components( LL_training, logdet_C_star, 
            logdet_Lambdainv_star, k_star__e, l_star_inv__e ):
        """ Terms of the log evidence, assuming Laplace posterior. """
        return A([ 
            LL_training, 
            -0.5 * ( logdet_C_star + logdet_Lambdainv_star ),
            -0.5 * dot( k_star__e, l_star_inv__e * k_star__e ) ])
    
    @cached
    def evidence( evidence_components ):
        """ Log evidence, assuming Laplace posterior. """
        return np.sum( evidence_components )

    @cached #( cskip = ('is_posterior', False, ['X_star__te', 'mu__t']) )
    def E_star__ee( is_posterior, d2LP_training, D_star, l_star_inv__e ):
        #X_star__te, mu__t ):
        """ Hessian of neg log likelihood, for local evidence approx """
        if not is_posterior:
            raise TypeError('attribute only available for posterior.')
        # start with -d2LP
        E = -d2LP_training
        # remove -d2LPrior from the diagonal
        E[ range(D_star), range(D_star) ] -= l_star_inv__e
        return E
        #return dot( X_star__te.T, mu__t[:, na] * X_star__te )

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for the hyperparameter vector, `theta`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    """
    ===============
    Local evidence
    ===============

    The (log) evidence is expensive to compute -- it depends on calculating
    the posterior. The approximation is therefore to calculate the posterior
    at a fixed value of `theta`, denoted `theta_n`, and then consider values
    of `theta` nearby. These candidate solutions are calculated as `theta_c`. 
    There is an function which can then be used to estimate the evidence 
    at `theta_c` (which can then be maximised as a function of `theta_c`).

    To implement this, we calculate a posterior at `theta_n`, and carry this
    forward as the `posterior` argument to the attributes below. 

    Note that `star` here refers to the dim reduction induced 
    by `theta_n`, not `theta_c`. This means that a second dimensionality
    reduction is sometimes necessary. For convention, *-space means the 
    space induced by `theta_n`, and +-space means the space induced by
    both `theta_n` and `theta_c`.

    """

    @cached
    def C_c_star__ee( posterior, C__dd ):
        """ Covariance at theta_c, in *-space induced by theta_n. """
        # if theta_n has no dim reduction
        if posterior.R_is_identity:
            return C__dd
        # otherwise project
        else:
            return mdot( posterior.R__de, C, posterior.R__de.T )
        # TO DO: if posterior.C_is_diagonal

    @cached
    def C_c_eig( C_c_star__ee ):
        """ Eigendecomposition of covariance at `theta_c` """
        try:
            l, Q = eigh( C_c_star__ee )
        except np.linalg.LinAlgError:
            l, Q = eigh( C_c_star__ee * 1e4 )
            l /= 1e4
        return {'l_c_star__e':l, 'Q_c_star__ee':Q}

    @cached( cskip=[('C_is_diagonal', True, 'C_c_eig')] )
    def l_c_star__e( C_is_diagonal, posterior, l__d, C_c_eig ):
        """ Diagonalised covariance at theta_c, in *-space """
        if C_is_diagonal:
            return l__d[ posterior.dims ]
        else:
            return C_c_eig['l_c_star__e']

    @cached
    def l_c_star_is_zero( l_c_star__e ):
        """ If the prior covariance is zero (can happen, safeguard). """
        return ( maxabs( l_c_star__e) == 0 )

    @cached
    def dims_c_star( l_c_star__e, cutoff_lambda ):
        """ Bool array of dimensions to keep in *-space. """
        cutoff = cutoff_lambda * maxabs(l_c_star__e)
        return ( l_c_star__e >= cutoff )

    @cached
    def l_c_star_is_singular( dims_c_star ):
        """ Is a second dim reduction necessary. """
        return not dims_c_star.all()

    @cached
    def second_dr_k( C_is_diagonal, l_c_star_is_singular ):
        """ Type of second dim reduction to apply. """
        if C_is_diagonal:
            if l_c_star_is_singular:
                return 'project'
            else:
                return 'none'
        else:
            if l_c_star_is_singular:
                return 'full'
            else:
                return 'rotate'

    @cached
    def R_c_star__ef( C_c_eig, dims_c_star ):
        """ Projection+rotation for 2nd dim reduction: +-space to *-space """
        return C_c_eig['Q_c_star'][:, dims_c_star].T

    @cached( cskip=[
        ('second_dr_k', 'none', 'C_c_eig'), 
        ('second_dr_k', 'rotate', 'C_c_eig'), 
        ('second_dr_k', 'project', 'C_c_eig') ])
    def l_c_plus__f( second_dr_k, C_c_eig, l_c_star__e, dims_c_star ):
        """ Diagonal covariance matrix for `theta_c`, in +-space. """
        if second_dr_k in ['none', 'rotate']:
            return l_c_star__e
        elif second_dr_k == 'project':
            return l_c_star__e[ dims_c_star ]
        else:
            return C_c_eig['l_c_star__e'][dims_c_star]

    @cached
    def C_c_plus__ff( l_c_plus__f ):
        """ Covariance matrix for `theta_c`, in +-space. """
        return diag( l_c_plus__f )

    @cached
    def Cinv_c_plus__ff( l_c_plus__f ):
        """ Precision matrix for `theta_c`, in +-space. """
        return diag( 1. / l_c_plus__f )

    @cached
    def D_plus( dims_c_star ):
        """ Number of dimensions in +-space. """
        return np.sum( dims_c_star )

    @cached( cskip = [
        ('second_dr_k', 'none', 'R_c_star__ef'),
        ('second_dr_k', 'project', 'R_c_star__ef') ])
    def E_n_plus__ff( second_dr_k, posterior, dims_c_star, R_c_star__ef ):
        """ Hessian of -LL at `theta_n` posterior mode, in +-space. """
        # retrieve
        E__ee = posterior.E_star__ee
        # dimensionality reduction from *-space to +-space
        if second_dr_k == 'none':
            return E__ee
        elif second_dr_k == 'project':
            idx = dims_c_star
            return E__ee[idx, :][:, idx]
        else: 
            return mdot( R_c_star__ef.T, E__ee, R_c_star__ef )

    @cached( cskip = [
        ('second_dr_k', 'none', 'R_c_star__ef'),
        ('second_dr_k', 'project', 'R_c_star__ef') ])
    def Lambdainv_n_plus__ff(second_dr_k, posterior, dims_c_star, R_c_star__ef):
        """ Posterior precision at `theta_n`, in +-space. """
        # retrieve
        Laminv__ee = posterior.Lambdainv_star__ee
        # dimensionality reduction from *-space to +-space
        if second_dr_k == 'none':
            return Laminv__ee
        elif second_dr_k == 'project':
            idx = dims_c_star
            return Laminv__ee[idx, :][:, idx]
        else: 
            return mdot( R_c_star__ef.T, Laminv__ee, R_c_star__ef )

    @cached( cskip = [
        ('second_dr_k', 'none', 'R_c_star__ef'),
        ('second_dr_k', 'project', 'R_c_star__ef') ])
    def k_n_plus__f( second_dr_k, posterior, dims_c_star, R_c_star__ef ):
        """ Posterior mode at `theta_n`, in +-space. """
        # retrieve
        k_star__e = posterior.k_star__e
        # dimensionality reduction from *-space to +-space
        if second_dr_k == 'none':
            return k_star__e
        elif second_dr_k == 'project':
            return k_star__e[ dims_c_star ]
        else: 
            return dot( R_c_star__ef.T, k_star__e )

    @cached( cskip = [
        ('second_dr_k', 'none', 'R_c_star__ef'),
        ('second_dr_k', 'project', 'R_c_star__ef') ])
    def X_plus__tf( second_dr_k, posterior, dims_c_star, R_c_star__ef ):
        """ Regressor matrix in +-space. """
        # retrieve
        X_star__te = posterior.X_star__te
        # dimensionality reduction from *-space to +-space
        if second_dr_k == 'none':
            return X_star__te
        elif second_dr_k == 'project':
            return X_star__te[ :, dims_c_star ]
        else: 
            return dot( X_star__te, R_c_star__ef )

    """ Approximate posterior at candidate theta """

    @cached
    def Lambdainv_c_plus__ff( E_n_plus__ff, Cinv_c_plus__ff ):
        """ Approximate precision matrix at `theta_c`, in +-space. """
        return E_n_plus__ff + Cinv_c_plus__ff

    @cached
    def Lambda_c_plus__ff( Lambdainv_c_plus__ff ):
        """ Approximate covariance matrix at `theta_c`, in +-space. """
        return inv( Lambdainv_c_plus__ff )

    @cached
    def k_c_plus__f( Lambdainv_c_plus__ff, Lambdainv_n_plus__ff, k_n_plus__f ):
        """ Approximate posterior mode at `theta_c`, in +-space. """
        return ldiv( 
                Lambdainv_c_plus__ff, 
                dot(Lambdainv_n_plus__ff, k_n_plus__f) )

    @cached( cskip = [
        ('second_dr_k', 'none', 'R_c_star__ef'),
        ('second_dr_k', 'project', 'R_c_star__ef') ])
    def k_c__d( k_c_plus__f, C_is_diagonal, second_dr_k, R_c_star__ef, 
            dims_c_star, D, posterior ):
        """ Approximate posterior mode at `theta_c`, in full space. """
        # to *-space
        if second_dr_k == 'none':
            k_c_star__e = k_c_plus__f
        elif second_dr_k == 'project':
            k_c_star__e = np.zeros( posterior.D_star )
            k_c_star__e[ dims_c_star ] = k_c_plus__f
        else:
            k_c_star__e = dot( R_c_star__ef.T, k_c_plus__f )
        # to full space
        if posterior.R_is_identity:
            return k_c_star__e
        elif C_is_diagonal:
            k__d = np.zeros( D )
            k__d[ posterior.dims ] = k_c_star__e
            return k__d
        else:
            return dot( posterior.R__de, k_c_star__e )

    @cached
    def z_c__t( X_plus__tf, k_c_plus__f ):
        """ Argument to nonlinearity for `theta_c` approx posterior mode. """
        return dot( X_plus__tf, k_c_plus__f )

    @cached
    def ez_c__t( z_c__t ):
        """ Intermediate computation """
        return exp( z_c__t )

    @cached
    def mu_c__t( ez_c__t, nonlinearity ):
        """ Expected firing rate for `theta_c` approx posterior mode. """
        if nonlinearity == 'exp':
            return ez_c__t
        elif nonlinearity == 'soft':
            return log( 1 + ez_c__t ) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached( cskip = ('nonlinearity', 'exp', 'mu_c__t') )
    def log_mu_c__t( nonlinearity, z_c__t, mu_c__t ):
        """ Log of expected firing rate for `theta_c` approx posterior mode """
        if nonlinearity == 'exp':
            return z_c__t
        else:
            return log( mu_c__t )

    """ Local evidence at candidate theta """

    @cached
    def local_evidence_components( mu_c__t, log_mu_c__t, slice_by_training,
            y_training__t, l_c_plus__f, E_n_plus__ff, D_plus, k_c_plus__f ):
        """ Components of the approx. evidence at `theta_c`. """
        # log likelihood
        mu__t = slice_by_training( mu_c__t )
        log_mu__t = slice_by_training( log_mu_c__t )
        LL = -np.sum( mu__t ) + dot( y_training__t, log_mu__t )
        if ~np.isfinite( LL ):
            return -np.inf
        if len( k_c_plus__f ) == 0:
            return LL
        # psi(theta)
        psi = A([ 
            LL,
            -0.5 * logdet( l_c_plus__f[:, na] * E_n_plus__ff + eye(D_plus) ),
            -0.5 * dot( k_c_plus__f, k_c_plus__f / l_c_plus__f ) ])
        return psi

    @cached
    def local_evidence( local_evidence_components ):
        """ The approx. evidence at `theta_c`. """
        return np.sum( local_evidence_components )

    @cached
    def LE_objective( local_evidence ):
        """ Local evidence objective to minimise. """
        return -local_evidence

    @cached( cskip=[ 
        ('C_is_diagonal', True, ['R_c_star__ef', 'dC_dtheta__idd']),
        ('C_is_diagonal', False, ['dl_dtheta__id']) ])
    def LE_jacobian( C_is_diagonal, second_dr_k, 
            posterior, dC_dtheta__idd, dl_dtheta__id, R_c_star__ef, data, 
            mu_c__t, X_plus__tf, slice_by_training, C_c_plus__ff, E_n_plus__ff,
            Lambda_c_plus__ff, Cinv_c_plus__ff, k_c_plus__f, dims_c_star,
            l_c_plus__f, N_theta, Lambdainv_c_plus__ff, ez_c__t,
            _raise_bad_nonlinearity, nonlinearity ):
        """ Jacobian of local evidence objective. """
        # diagonal case
        if C_is_diagonal:
            # project to *-space
            if posterior.R_is_identity:
                dl_dtheta_star__ie = dl_dtheta__id
            else:
                dl_dtheta_star__ie = [ 
                        dl[ posterior.dims ] for dl in dl_dtheta__id ]
            # project to +-space
            if second_dr_k == 'none':
                dl_dtheta_plus__if = dl_dtheta_star__ie
            else:
                dl_dtheta_plus__if = [
                        dl[ dims_c_star ] for dl in dl_dtheta_star__ie ]

            # residuals at candidate solution
            resid_c__t = slice_by_training( data.y__t - mu_c__t )
            # intermediate quantities
            X_plus__tf = slice_by_training( X_plus__tf )
            if nonlinearity == 'exp':
                dLL_dk_c__f = dot( X_plus__tf.T, resid_c__t )
            elif nonlinearity == 'soft':
                dF_c__t = ez_c__t / (1 + ez_c__t) / log(2)
                dLL_dk_c__f = dot( X_plus__tf.T, resid_c__t * dF_c__t/mu_c__t )
            else:
                _raise_bad_nonlinearity()
            C_En__ff = l_c_plus__f[:, na] * E_n_plus__ff
            Lam_Cinv__ff = Lambda_c_plus__ff / l_c_plus__f[na, :]
            kT_En_minus_Cinv__f = dot( 
                    k_c_plus__f.T, 
                    E_n_plus__ff - diag( 1./l_c_plus__f ) )
            # calculate for each variable in theta
            dpsi__i = np.empty( N_theta )
            # derivatives wrt `theta_h` variables
            for j in range( N_theta ):
                dl__f = dl_dtheta_plus__if[ j ]
                B__ff = Lam_Cinv__ff * ( dl__f / l_c_plus__f )[na, :]
                Bk__f = ldiv( 
                        Lambdainv_c_plus__ff, 
                        k_c_plus__f * dl__f / (l_c_plus__f ** 2) )
                dpsi__i[ j ] = sum([
                    dot( dLL_dk_c__f, Bk__f ),
                    -0.5 * np.trace( dot(B__ff, C_En__ff) ),
                    0.5 * dot( kT_En_minus_Cinv__f, Bk__f ) ])
            # make negative
            return -dpsi__i

        # non-diagonal case
        else:
            # project to *-space
            if posterior.R_is_identity:
                dC_dtheta_star__iee = dC_dtheta__idd
            else:
                dC_dtheta_star__iee = [ 
                        mdot( posterior.R__de.T, dC__dd, posterior.R__de ) 
                        for dC__dd in dC_dtheta__idd ]
            # project to +-space
            dC_dtheta_plus__iff = [ 
                    mdot( R_c_star__ef.T, dC__ee, R_c_star__ef ) 
                    for dC__ee in dC_dtheta_star__iee ]

            # residuals at candidate solution
            resid_c__t = slice_by_training( data.y__t - mu_c__t )
            # intermediate quantities( all in +-space )
            X_plus__tf = slice_by_training( X_plus__tf )
            if nonlinearity == 'exp':
                dLL_dk_c__f = dot( X_plus__tf.T, resid_c__t )
            elif nonlinearity == 'soft':
                dF_c__t = ez_c__t / (1 + ez_c__t) / log(2)
                dLL_dk_c__f = dot( X_plus__tf.T, resid_c__t * dF_c__t/mu_c__t )
            else:
                _raise_bad_nonlinearity()
            C_En__ff = dot( C_c_plus__ff, E_n_plus__ff )
            Lam_Cinv__ff = dot( Lambda_c_plus__ff, Cinv_c_plus__ff )
            kT_En_minus_Cinv__f = dot( 
                    k_c_plus__f.T, E_n_plus__ff - Cinv_c_plus__ff )
            # calculate for each variable in theta
            dpsi__i = np.empty( N_theta )
            # derivatives wrt `theta_h` variables
            for j in range( N_theta ):
                dC__ff = dC_dtheta_plus__ff[ j ]
                B__ff = mdot( Lam_Cinv__ff, dC, Cinv_c_plus__ff )
                Bk__f = ldiv( 
                        Lambdainv_c_plus__ff, 
                        mdot( Cinv_c_plus__ff, dC__ff, Cinv_c_plus__ff, 
                            k_c_plus__f )
                        )
                dpsi__i[ j ] = sum([
                    dot( dLL_dk_c__f, Bk__f ),
                    -0.5 * np.trace( dot(B__ff, C_En__ff) ),
                    0.5 * dot( kT_En_minus_Cinv__f, Bk__f ) ])
            # make negative
            return -dpsi__i






"""
====================
Basic priors on `k`
====================
"""

class Prior( Solver ):

    """ Superclass for priors on `k`. """
    
    C_is_diagonal = False


class Diagonal_Prior( Prior ):

    """ Sets non-diagonal methods to return None """

    C_is_diagonal = True

    @cached
    def C__dd():
        raise TypeError('cannot compute `C`: diagonal covariance matrix')

    @cached
    def C_eig():
        raise TypeError('cannot compute `C_eig`: diagonal covariance matrix')

    @cached
    def R__de():
        raise TypeError('cannot compute `R__de`: diagonal covariance matrix')


class ML( Diagonal_Prior ):

    """ No prior on `k`, i.e. maximum likelihood. """

    # default variables
    hyperparameter_names = []
    bounds_theta = []
    default_theta0 = []

    R_is_identity = True
        
    """ Prior on `k` """

    @cached
    def l__d( D ):
        return np.ones( D ) * 1e12

    @cached
    def dl_dtheta__id():
        return []


class Ridge( Diagonal_Prior ):

    """ Ridge prior on `k`. 
    
    This prior is defined in terms of a single hyperparameter, `rho`.

    """

    R_is_identity = True

    # default variables
    hyperparameter_names = [ 'rho' ]
    bounds_theta = [ (-15, 15) ]
    default_theta0 = [0.]
        
    grid_search_theta_parameters = { 
            'bounds':[ [ -14, 14 ] ], 
            'spacing': [ 'linear' ], 
            'initial': A([ 0. ]), 
            'strategy': '1D', 
            'grid_size': (10,) }

    """ Prior on `k` """

    @cached
    def l__d( theta, D ):
        """ Diagonal of prior covariance matrix for `k`. """
        if np.iterable(theta):
            rho = theta[0]
        else:
            rho = theta
        return exp( -rho ) * ones( D )

    @cached
    def dl_dtheta__id( l__d ):
        """ Derivative of the diagonal prior covariance matrix. """
        return [ -l__d ]

