from common import *



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

"""
=============
Data class
=============
"""

class Data( AutoReloader ):

    """ Data structure, for a conditionally-Poisson response model. """

    def __init__( self, X_td, y_t, 
            normalise=False, whiten=False, add_constant=True, 
            announcer=None, verbose=False ):
        """ Construct a Data object. 
        
        Provide the stimulus `X_td`, which is size ( T x D ).
        Provide the spike count response `y_t`, which is size ( T ).

        Optional keywords:

        - `normalise` : transform each of the D dimensions of `X_td` to have
            zero mean and unitary standard deviation

        - `whiten` : transform so that each of the D dimensions of `X_td` has
            zero mean, and X^T . X = I

        - `add_constant` : append an extra dimension to `X_td` which is 
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
        if normalise or whiten:
            self.announce('normalising', prefix='Data')
            Xm = np.mean( X_td, axis=0 )
            Xs = np.std( X_td, axis=0 )
            if np.min( Xs ) > 0:
                raise ValueError('`X` contains degenerate dimension')
            X_td = ( X_td - Xm[na, :] ) / Xs[na, :]
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
        self.X_td, self.y_t = X_td, y_t
        # sizes
        T, D = self.T, self.D = shape( self.X_td )
        if len( y_t ) != self.T:
            raise ValueError('# time bins in `X_td` and `y_t` do not match')

    def announce( self, *a, **kw ):
        return self._announcer.announce( *a, **kw )

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
        y = self.y
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
===============
Solver classes
===============
"""

class Solver( AutoCR ):

    """ Superclass for GLM solvers.

    This should be subclassed to define certain structural priors over
    the variables.
    
    """

    def __init__( self, data, initial_conditions=None, 
            testing_proportion=0, testing_block_size_smp=10,
            solve=False, announcer=None, verbose=True, verbose_cache=False, 
            empty_copy=False, nonlinearity='exp', **kw ):
        """ Create a Solver object, for given data object, `data`.
        
        Keywords:

        - `initial_conditions`: Solver object, providing initial conditions

        - `solve`: solve by default

        - `testing_proportion` : what fraction of the data should be set aside
            for cross-validation (between 0 and 1)

        - `testing_block_size_smp` : max size of the contiguous blocks in
            the testing data set. 

        - `nonlinearity` : the output nonlinearity. Either 'exp' or 'soft'.
            * 'exp'  :  mu_t = exp( z_t )
            * 'soft' :  mu_t = log_2( 1 + exp( z_t ) )

        Verbosity:

        - `announcer` : provide Announcer object (constructed if not provided)
        - `verbose` : whether there are verbose announcements

        """
        # create announcer
        if announcer is None:
            announcer = Announcer( verbose=verbose )
        if not verbose_cache and not empty_copy:
            announcer.suppress('cache')
        self._announcer = announcer
        # empty copy?
        if empty_copy:
            return
        # save data
        self._data = data
        # define training and testing regions of the data
        self.define_training_and_testing_regions( 
                testing_proportion=testing_proportion,
                testing_block_size_smp=testing_block_size_smp )
        # nonlinearity
        if nonlinearity in ['exp', 'soft']:
            self.nonlinearity = nonlinearity
        else:
            raise ValueError("`nonlinearity` must be 'exp' or 'soft'")
        # initial conditions
        self._parse_initial_conditions( initial_conditions )
        self._reset()
        # solve, if requested
        if solve:
            self.solve()

    """ Posterior objects """

    # by default, this is not a posterior, unless specified otherwise
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
        p.nonlinearity_h = self.nonlinearity_h 
        # copy the history
        try:
            p._posterior_history = self._posterior_history
        except AttributeError:
            pass
        # set any remaining attributes
        for k, v in kw.items():
            setattr(p, k, v)
        # copy the current cache
        return p

    @property
    def data( self ):
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

    """ Methods """

    def announce(self, *a, **kw):
        """ Verbose print. Sends call to the `self._announcer` object. """
        return self._announcer.announce(*a, **kw)

    def _caching_callback( self, name=None, arguments=None ):
        """ Callback method when a cached variable is computed. """
        return self.announce( 'computing %s' % name, prefix='cache' )

    """
    =====================================
    Common methods for initial conditions
    =====================================
    """
    
    def _Prior_class( self ):
        """ Returns the superclass that defines the `L` or `C` method. """
        return [ c for c in self.__class__.mro() 
                if c.__dict__.has_key('L') or c.__dict__.has_key('C') ][0]

    def _recast_theta( self, ic ):
        """ Process results of previous solver to determine initial theta.

        Arguments:
        - `ic` : previously solved version of model
        
        Default: if initial conditions come from the same Prior class, then
        inherit the fitted theta.
        
        Subclass for additional behaviour.

        """
        c1 = ic._Prior_class
        c2 = self._Prior_class
        if c1 == c2:
            p = ic.posterior
            tv = ic.theta
            return tv
        else:
            err_str = 'subclass how to recast theta from %s to %s '
            err_str = err_str % ( c1.__name__, c2.__name__ )
            raise TypeError( err_str )

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
            if len( v ) == self.required_v_length
                return
            else:
                delattr( self, v + '_vec' )
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
        if hasattr( self, 'v' )
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
    ===============================
    Common methods for optimisation
    ===============================
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
        f = self.cfunction( eq, '_negLP_objective', 'v' )
        df = self.cfunction( eq, '_negLP_jacobian', 'v' )
        d2f = self.cfunction( eq, '_negLP_hessian', 'v' )
        # initialise starting value of v
        self.initialise_v()
        # reporting during each step
        def callback( v, assess_convergence=True ):
            # set the current value of `v` (if required)
            self.csetattr( 'v', v )
            # run the callback function
            self._negLP_callback( prefix='calc_posterior' )
            # have we converged
            last_negLP = self._last_negLP
            this_negLP = self._negLP_objective
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
        f = self.cfunction( np.array_equal, '_LE_objective', 'theta' )
        df = self.cfunction( np.array_equal, '_LE_jacobian', 'theta' )
        # a copy is necessary as fmin_l_bfgs_b makes changes in place
        g = lambda x: f( x.copy() )
        dg = lambda x: df( x.copy() )
        # reporting during each step
        def callback( theta, assess_convergence=True ):
            # set the current value of `theta` (if required)
            self.csetattr( 'theta', theta )
            # have we converged
            last_LE = self._last_LE
            this_LE = getattr( self, '_LE_objective' )
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
        g(theta_v)

    def solve_theta( self, grid_search=False, ftol_per_obs=None, 
            max_iterations=10, verbose=1, **kw ):
        """ Optimise for `theta` by maximum marginal likelihood. 
        
        Convergence is measured by `ftol_per_obs`. This is multiplied by
        `self.N_observations` to give a step improvement criterion for 
        continuing the optimisation. This defaults to `self.ftol_per_obs`
        if not provided.
        
        """
        # how many hyperparameters are we optimising over
        N_theta = self.N_theta
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # do we actually have a theta_v to solve
        if N_theta_v == 0:
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
                announce_str = announce_str % tuple(tv)
                max_len = 30 + 2*len(tv)
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

    def solve(self):
        """ Solve for parameters and hyperparameters. Define in subclass. """
        raise NotImplementedError()

    """ Data munging """

    @cached
    def T( data ):
        """ Number of time bins. """
        return data.T

    @cached
    def D( data ):
        """ Dimensionality of the stimulus. """
        return data.D

    """
    ================
    Cross-validation
    ================
    """
    
    def define_training_and_testing_regions( 
            self, testing_proportion, testing_block_size_smp ):
        """ Partitions the data for cross-validation. 
        
        Arguments:

        - `testing_proportion` : how much of the data to put aside for
            testing purposes. Should be between 0 and 0.4 (recommended: 0.2)

        - `testing_block_size_smp` : max size of the contiguous blocks in
            the testing data set. 
        
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
        block_start_idx = np.cumsum(np.hstack([[0], block_lengths]))[:-1]
        block_end_idx = np.cumsum(block_lengths)
        # create slices
        block_slices = A([ slice(idx_i, idx_j, None) 
                for idx_i, idx_j in zip(block_start_idx, block_end_idx) ])
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
        return len( testing_slices ) > 0

    @cached
    def T_training( training_slices ):
        return int(sum([ s.stop - s.start for s in training_slices ]))

    @cached
    def T_testing( testing_slices ):
        return int(sum([ s.stop - s.start for s in testing_slices ]))

    def slice_by_training( self, sig ):
        """ Retrieves the training slices of a signal. """
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
        """ Retrieves the testing slices of a signal. """
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
        """ Sets the signal components during testing regions to be zero. """
        if not self.has_testing_regions:
            return sig
        sig = sig.copy()
        for s in self.testing_slices:
            sig[ s, ... ] = 0
        return sig

    """
    ===================
    Derivative checking
    ===================
    """
    
    def check_LP_derivatives( self, eps=1e-6, debug=False, 
            error_if_fail=False, error_threshold=0.05 ):
        """ Check derivatives of the log posterior.

        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective attributes `_negLP_jacobian` and 
        `_negLP_hessian`, and compares with empirical values, obtained 
        from finite differences method on `_negLP_objective` and 
        `_negLP_jacobian` respectively.

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
        LP = self._negLP_objective
        dLP = self._negLP_jacobian
        # evaluate empirical jacobian
        edLP = np.zeros( N_v )
        for i in progress.dots( range(N_v), 'empirical jacobian' ):
            self.v = v0 + dv(i)
            edLP[i] = ( self._negLP_objective - LP ) / eps
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
        d2LP = self._negLP_hessian
        # evaluate empirical
        def empirical_d2LP(i):
            self.v = v0 + dv(i)
            return ( self._negLP_jacobian - dLP ) / eps
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
            raise ValueError('Hessian of LP failed at %.6f' % err1 )
        # debug
        if debug:
            tracer()

    def check_LE_derivatives( self, eps=1e-6, debug=False, 
            error_if_fail=False, error_threshold=0.05, **kw ):
        """ Check derivatives of the local evidence wrt `theta`.
        
        For the Jacobian, evaluates the analytic derivatives provided by
        the attribute `_LE_jacobian` and compares these with the empirical
        values, obtained from finite differences method on `_LE_objective`.

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
        LE = self._LE_objective
        a = dLE = self._LE_jacobian
        # evaluate empirical
        theta_v0 = np.array( self.theta ).astype(float)
        e = edLE = np.zeros( N_theta )
        for i in range( N_theta ):
            theta_v = theta_v0.copy()
            theta_v[i] += eps
            self.theta_v = theta_v
            edLE[i] = ( self._LE_objective - LE) / eps
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



class UnitSolver( Solver ):

    """ Superclass for single unit MoP model solvers. 
    
    Assumptions:
    =============

    (1) `h` is a circular Gaussian Process. 

    This allows a dramatic simplification of the problem over `h`: the
    covariance matrix is diagonalised by the (fixed) DFT matrix. For
    large datasets, this avoids the problem of huge eigendecompositions, 
    and we can jump straight to the dimensionality reduction.

    The prior over `h` in the freq. domain needs to be defined in a
    subclass.


    (2) `h` and `k` are estimated separately.

    The prior over `h` in the freq. domain, and the prior over `k`, need to be 
    defined in a subclass.


    (3) GLM-like model for stimulus-response relationship (`k`). 
    
    This assumes that the stimulus-response relationship is captured by a
    Linear-Nonlinear relationship, with an exponential link function, i.e.
    F[X, k] = exp( X.k ). Further, this also assumes that there is a 
    multivariate Gaussian prior on `k`, dependent on the set of 
    hyperparameters `theta_k`. It is assumed that `theta_k` can be fitted 
    via maximum marginal likelihood (MML).
   
    """

    # for dimensionality reduction
    _cutoff_lambda_k = 1e-12
    _cutoff_lambda_h = 1e-9

    _variable_names = ['k', 'h']

    """ Useful properties """

    @property
    def N_theta_k( self ):
        """ Number of hyperparameters for `k`. """
        return len( self._hyperparameter_names_k )

    @property
    def N_theta_h( self ):
        """ Number of hyperparameters for `h`. """
        return len( self._hyperparameter_names_h )

    @property
    def N_observations( self ):
        return self.T_training

    """
    ==================
    Initial conditions
    ==================
    """

    @property
    def _grid_search_theta_h_available( self ):
        return hasattr( self, '_grid_search_theta_h_parameters' )
    
    @property
    def _grid_search_theta_k_available( self ):
        return hasattr( self, '_grid_search_theta_k_parameters' )

    def _parse_initial_conditions( self, initial_conditions ):
        """ Sets up the initial conditions for the model. """
        # global defaults
        self.initial_conditions = ics = Bunch()
        ics['k'] = zeros( self.D )
        ics['h_vec'] = A([0.])
        ics['dims_h'] = zeros( self.T, dtype=bool )
        ics['dims_h'][0] = True
        ics['theta_k'] = self._default_theta_k0
        ics['theta_h'] = self._default_theta_h0
        # parse initial conditions: None
        if initial_conditions == None:
            pass
        # parse initial_conditions: dict
        elif isinstance( initial_conditions, dict ):
            for k in ics.keys():
                ics[k] = initial_conditions.get( k, ics[k] )
        # parse initial_conditions: Solver object
        else: 
            # copy posterior on `k`
            if hasattr( initial_conditions, 'posterior_k' ):
                ics['k'] = initial_conditions.posterior_k.k
            # copy posterior on `h`
            if hasattr( initial_conditions, 'posterior_h' ):
                try:
                   ics['h_vec'] = initial_conditions.posterior_h.h_vec
                   ics['dims_h'] = initial_conditions.posterior_h.dims_h
                except TypeError:
                    ics['h_vec'] = A([0.])
                    ics['dims_h'] = zeros( self.T, dtype=bool )
                    ics['dims_h'][0] = True
            # recast values of `theta_k` and `theta_h`
            try:
                ics['theta_k'] = self._recast_theta_k( initial_conditions )
            except TypeError:
                pass
            try:
                ics['theta_h'] = self._recast_theta_h( initial_conditions )
            except TypeError:
                pass
        # replace any invalid values
        for i in range( self.N_theta_k ):
            if (ics['theta_k'][i] is None) or (ics['theta_k'][i] == np.nan):
                ics['theta_k'] = self._default_theta_k0[i]
        for i in range( self.N_theta_h ):
            if (ics['theta_h'][i] is None) or (ics['theta_h'][i] == np.nan):
                ics['theta_h'] = self._default_theta_h0[i]

    @property
    def _Prior_k_class( self ):
        """ Returns the superclass that defines the `Lk` or `Ck` method. """
        return self._Prior_v_class( 'k' )

    @property
    def _Prior_h_class( self ):
        """ Returns the superclass that defines the `Lh` or `Ch` method. """
        return self._Prior_v_class( 'h' )

    def _recast_theta_k( self, ic ):
        """ Process results of previous solver to determine initial theta_k. """
        return self._recast_theta_v( 'k', ic )

    def _recast_theta_h( self, ic ):
        """ Process results of previous solver to determine initial theta_h. """
        return self._recast_theta_v( 'h', ic )

    def _initialise_theta_k( self ):
        """ If `theta_k` is not set, initialise it. """
        return self._initialise_theta_v( 'k' )

    def _initialise_theta_h( self ):
        """ If `theta_h` is not set, initialise it. """
        return self._initialise_theta_v( 'h' )

    def _initialise_k_vec( self ):
        """ If `k_vec` is not set (validly), initialise it. """
        return self._initialise_v_vec( 'k' )

    def _initialise_h_vec( self ):
        """ If `h_vec` is not set (validly), initialise it. """
        return self._initialise_v_vec( 'h' )

    def _reset_theta_k( self ):
        """ Force reset of `theta_k`. """
        self._reset_theta_v( 'k' )

    def _reset_theta_h( self ):
        """ Force reset of `theta_h`. """
        self._reset_theta_v( 'h' )

    def _reset_k_vec( self ):
        """ Force reset of `k_vec`. """
        self._reset_v_vec( 'k' )

    def _reset_h_vec( self ):
        """ Force reset of `h_vec`. """
        self._reset_v_vec( 'h' )

    
    """
    ========
    Plotting
    ========
    """
    
    def plot_h( self, posterior_h=None, *a, **kw ):
        """ Plot modulator. See `Data.plot_h` for docstring. 
        
        By default, this plots `self.posterior`, unless an alternative
        posterior is provided.
        
        """
        if posterior_h is None:
            posterior_h = self.posterior_h

        return self.data.plot_h( 
                posterior_h, 
                testing_slices=self.testing_slices,
                training_slices=self.training_slices,
                *a, **kw )

    def plot_y( self, posterior=None, **kw ):
        """ Plot spike counts. See `Data.plot_k` for docstring. 
        
        By default, this plots `self.posterior`, unless an alternative
        posterior is provided.
        
        """
        if posterior is None:
            posterior = self.posterior

        return self.data.plot_y( posterior, **kw )

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for `h`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    @cached
    def h_star( h_vec ):
        return h_vec

    """
    ====================
    Dim reduction on `h`
    ====================
    """

    def collapse_h( self, h, dims, padding='zero', reduce=True, **kw ):
        """ Reduce dimensionality of `h`. Returns `h_star`.

        Arguments:

        - `h`: (T) or (T x J) float array
        - `dims`: (2T) boolean array, of which fft coefficients to keep.

        Returns:

        - `h_star`: (T*) or (T* x J) float array, where T* is the number
            of `True` elements of `dims`.

        Dimensionality reduction takes place in three steps:
        
        (1) The signal `h` is padded by appending T rows. The value of this 
        padding depends on the keyword `padding`, and is for the purposes of 
        handling the circular boundary condition that holds between a[0] 
        and a[-1]:

        - 'linear': a[T:] is a linear interpolation between a[0] and a[-1]
        - 'cos': a[T:] is a cosine interpolation between a[0] and a[-1]
        - 'zero': a[T:] is zero (used for all derivatives)
        - 'flip': combination of flipped continuity and linear

        (2) The special fft (i.e. real-valued, real-signal, orthogonal) is
        taken on the length-2T padded signal.

        (3) Only the dimensions of the result for which `dims` is True are
        kept; the rest are excised.

        """
        # calculate shape of padded region
        ps = self.pad_size
        sh = A( shape(h) )
        if len(sh) > 2:
            raise NotImplementedError()
        sh_padded = sh.copy()
        sh_padded[0] += ps*2
        # create empty array for padded version of `h`
        h_padded = zeros( sh_padded )
        # fill in known
        h_padded[ :-2*ps, ... ] = h
        # fill in unknown
        if padding in ['linear', 'cos']:
            if padding == 'linear':
                ramp = np.linspace(0, 1., ps*2)
            elif padding == 'cos':
                ramp = 1 - 0.5*(1 + np.cos(np.linspace( 0, np.pi, ps*2 )))
            if len(sh) == 1:
                ramp *= ( h[0] - h[-1] )
                ramp += h[-1]
                h_padded[ -2*ps: ] = ramp
            elif len(sh) == 2:
                ramp = ramp[:, na] * (h[0] - h[-1])[na, :]
                ramp += h[-1][na, :]
                h_padded[ -2*ps:, : ] = ramp
        elif padding == 'flip':
            if not (h.shape[0] == ps * 2):
                raise NotImplementedError()
            ramp = np.linspace( 0, 1., ps*2 )
            ramp = ramp[:, na] * 2 * (h[0, :] - h[-1, :])[na, :]
            ramp += 2 * h[-1, :][na, :] - h[::-1, :]
            h_padded[ -2*ps:, : ] = ramp
        elif padding == 'zero':
            pass
        else:
            raise ValueError('unknown type of padding: %s' % padding)
        # perform fft, and keep only specified dimensions
        if reduce:
            h_star = rosfft( h_padded.T, dims ).T
        else:
            h_star = osfft( h_padded.T ).T
        return h_star

    def expand_h( self, h_star, dims ):
        """ Increase dimensionality of `h_star`. Returns `h`.

        Arguments:

        - `h_star`: (T*) or (T* x J) float array
        - `dims`: (2T) boolean array, of which fft coefficients are
            being reported in `h_star`.

        Dimensionality restoration takes place in three steps:

        (1) `h_star` is filled out to be of length 2T, where the missing
        dimensions are imputed to be zero value.

        (2) The inverse special fft is taken.

        (3) The padding is removed, returning `h`.

        """
        d1, d2 = h_star.shape[0], np.sum(dims)
        if not d1 == d2:
            err_str = '`h_star` is length %d, but should be length %d'
            raise ValueError(err_str % (d1, d2))
        # set missing dimensions to zero, then ifft
        h_padded = irosfft( h_star.T, dims ).T
        # trim
        ps = self.pad_size
        h = h_padded[ :-2*ps, ... ]
        # return
        return h

    @cached
    def abs_freqs_h( T, pad_size ):
        return calc_abs_freqs( T, pad_size )

    """
    ================
    Log Prior on `h`
    ================
    """
    
    @cached
    def Lh( theta_h ):
        """ Diagonal of prior covariance for `h` in the freq domain. """
        raise NotImplementedError('must subclass')
       
    @cached
    def dLh_dtheta( theta_h ):
        """ Jacbian of diag of prior covariance for `h` in the freq domain. """
        raise NotImplementedError('must subclass')

    @cached
    def dims_h( Lh, _cutoff_lambda_h ):
        return ( Lh/maxabs(Lh) > _cutoff_lambda_h )

    @cached
    def T_star( dims_h ):
        return np.sum( dims_h )

    @cached
    def required_h_vec_length( T_star ):
        return T_star

    @cached
    def Lh_star( Lh, dims_h ):
        return Lh[ dims_h ]

    @cached
    def Lhinv_star( Lh_star ):
        return 1. / Lh_star

    @cached
    def logdet_Ch_star( Lh_star ):
        return np.sum( log( Lh_star ) )

    @cached
    def h( expand_h, h_star, dims_h ):
        return expand_h( h_star, dims_h )

    def _reproject_to_h_vec( self, h_vec=None, dims_h=None, posterior=None ):
        """ Unprojects and reprojects `h_vec` (to the new dim).

        Can either provide `h_vec` and `dims_h`, *or* a posterior which
        contains these as attributes.

        It is assumed that `h_vec` has dimensionality described by `dims_h`.
        It is reprojected according to the current value of `self.dims_h`. 

        Note that this does not change the object's state.

        If P1 is the projection operator for dims_h_source, and P2 is the 
        projection operator for dims_h_target, then this is the equivalent
        of applying P2 . P1^-1 to the vector `h_vec`.

        """
        if [ h_vec, dims_h, posterior ].count(None) == 0:
            raise ValueError('either provide `h_vec/dims_h` or `posterior`')
        elif posterior is not None:
            return self._reproject_to_h_vec( 
                    h_vec=posterior.h_vec, dims_h=posterior.dims_h )
        else:
            new_h_vec = zeros( self.T * 2 )
            new_h_vec[ dims_h ] = h_vec
            new_h_vec = new_h_vec[ self.dims_h ]
            return new_h_vec

    """
    ===================
    Nonlinearity on `h`
    ===================
    """

    def _raise_bad_nonlinearity( self ):
        raise NotImplementedError(
                "`nonlinearity_h` should only be 'exp' or 'soft'")

    @cached
    def exph( h ):
        return exp(h)
    
    @cached
    def Fh( exph, nonlinearity_h ):
        if nonlinearity_h == 'exp':
            return exph
        elif nonlinearity_h == 'soft':
            return log( 1 + exph ) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached
    def dFh( exph, nonlinearity_h ):
        if nonlinearity_h == 'exp':
            raise AssertionError('should not reach this point')
            return exph
        elif nonlinearity_h == 'soft':
            return exph / (1 + exph) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached
    def d2Fh( exph, nonlinearity_h ):
        if nonlinearity_h == 'exp':
            raise AssertionError('should not reach this point')
            return exph
        elif nonlinearity_h == 'soft':
            return exph / ( (1 + exph)**2 ) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached( cskip = [ ('nonlinearity_h', 'exp', ['Fh', 'dFh']) ] )
    def dFh_on_Fh( nonlinearity_h, Fh, dFh ):
        if nonlinearity_h == 'exp':
            return 1
        else:
            return dFh / Fh

    @cached( cskip = [ ('nonlinearity_h', 'exp', ['Fh', 'd2Fh']) ] )
    def d2Fh_on_Fh( nonlinearity_h, Fh, d2Fh ):
        if nonlinearity_h == 'exp':
            return 1
        else:
            return d2Fh / Fh

    """
    ====================
    Log posterior on `h`
    ====================
    """
    
    @cached
    def g( Fh ):
        return Fh

    @cached
    def log_g( nonlinearity_h, h, Fh ):
        if nonlinearity_h == 'exp':
            return h
        elif nonlinearity_h == 'soft':
            return log( Fh )
        else:
            self._raise_bad_nonlinearity()

    @cached
    def mu__h( Fh, posterior_k ):
        return Fh * posterior_k.expected_FXk

    @cached
    def log_mu__h( log_g, posterior_k ):
        return log_g + posterior_k.expected_log_FXk

    @cached
    def y_training( data, slice_by_training ):
        return slice_by_training( data.y )

    @cached
    def LL_training__h( y_training, slice_by_training, mu__h, log_mu__h ):
        s = slice_by_training
        mu, log_mu = s(mu__h), s(log_mu__h)
        return -np.sum( mu ) + dot( y_training, log_mu )

    @cached
    def LPrior__h( logdet_Ch_star, h_star, Lhinv_star ):
        return sum([
            -0.5 * logdet_Ch_star, 
            -0.5 * dot( h_star.T, Lhinv_star * h_star ) ])

    @cached
    def LP_training__h( LL_training__h, LPrior__h ):
        return LL_training__h + LPrior__h

    """ Jacobian """

    @cached
    def resid__h( mu__h, data, zero_during_testing ):
        return zero_during_testing( data.y - mu__h )

    @cached
    def dLL_training__h( resid__h, dFh_on_Fh, dims_h, collapse_h ):
        """ Jacobian of training log likelihood wrt `h_vec`. """
        return collapse_h( resid__h * dFh_on_Fh, dims_h, padding='zero' )

    @cached
    def dLPrior__h( Lhinv_star, h_star ):
        """ Jacobian of log prior wrt `h_vec`. """
        return -Lhinv_star * h_star

    @cached
    def dLP_training__h( dLL_training__h, dLPrior__h ):
        """ Jacobian of training log prob wrt `h_vec`. """
        return dLL_training__h + dLPrior__h

    """ Hessian """

    @cached
    def d2LP_training__h( mu__h, resid__h, dims_h, T_star, Lhinv_star, 
            zero_during_testing, data, dFh_on_Fh, d2Fh_on_Fh, nonlinearity_h ):
        """ Hessian of training log prob wrt `h_vec`. """
        import sandwich
        if nonlinearity_h == 'exp':
            d2LL = -mu__h
        else:
            d2LL = resid__h * d2Fh_on_Fh - data.y * ( dFh_on_Fh ** 2 )
        d2LL = zero_during_testing( d2LL )
        d2LL_zero_padded = np.hstack([ d2LL, np.zeros_like(d2LL) ])
        yy = np.fft.fft( d2LL_zero_padded )
        F = ( np.sum(dims_h) - 1 ) / 2
        d2LP = sandwich.calzino( yy.real, yy.imag, F ) / len(yy)
        d2LP[ 0, 1: ] /= sqrt(2)
        d2LP[ 1:, 0 ] *= sqrt(2)
        # add d2LPrior to the diagonal
        d2LP[ range(T_star), range(T_star) ] -= Lhinv_star
        return d2LP

    """
    ===============
    Solution for `h`
    ===============
    """
   
    @cached
    def _negLP_objective__h( LP_training__h ):
        return -LP_training__h

    @cached
    def _negLP_jacobian__h( dLP_training__h ):
        return -dLP_training__h

    @cached
    def _negLP_hessian__h( d2LP_training__h ):
        return -d2LP_training__h

    def _negLP_callback__h( self, **kw ):
        try:
            offset = self.posterior_k.LPrior__k
        except AttributeError:
            offset = 0
        LP_total = self.LP_training__h + offset
        self.announce( 'LP step:  %.3f' % LP_total, n_steps=4, **kw )

    def _preoptimise_h0_vec_for_LP_objective(self, h_vec_0, ftol_per_obs=None):
        """ Trims the high freqs from `h_vec_0` as a pre-optimisation.

        Trimming continues so long as the improvement in the objective is
        greater than `ftol_per_obs * self.N_observations`. The parameter
        `ftol_per_obs`, if not provided, defaults to `self.ftol_per_obs`.

        """
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # functions
        eq = np.array_equal
        f = self.function( '_negLP_objective__h', 'h_vec' )
        # initial value
        best_h_vec_0 = h_vec_0
        best_objective = f( h_vec_0 )
        N_coeffs_removed = 0
        # start removing coefficients
        h_vec_1 = h_vec_0.copy()
        ndims = (len(h_vec_1) - 1)/2
        offsets = np.arange(ndims-1, -1, -1)
        for i, o in enumerate(offsets):
            h_vec_1[o + ndims + 1] = 0
            h_vec_1[o + 1] = 0
            # if we have decreased the objective, note this solution
            delta_obj = f(h_vec_1) - best_objective
            if delta_obj < 0:
                best_h_vec_0 = h_vec_1.copy()
                best_objective = f(best_h_vec_0)
                N_coeffs_removed = 2 * (i + 1)
            # if no improvement or too small, end here
            if delta_obj > -ftol_per_obs * self.N_observations:
                break
        # try the zero solution
        h_vec_1 *= 0
        if f(h_vec_1) < best_objective:
            best_h_vec_0 = h_vec_1.copy()
            best_objective = f(best_h_vec_0)
            N_coeffs_removed = len(h_vec_0)
        # announce
        if N_coeffs_removed > 0:
            self.announce( 'LP step: pre-zeroed %d coefficients of `h`' 
                    % N_coeffs_removed, prefix='calc_posterior_h' )
        return best_h_vec_0

    """
    ===============================
    Posterior_h specific attributes
    ===============================

    These should only be computed (and used) for posterior objects.

    """

    @cached
    def Lambdainv_h_vec( _negLP_hessian__h, is_posterior ):
        if not is_posterior:
            raise TypeError('attribute only available for `h` posterior.')
        return _negLP_hessian__h

    @cached
    def Lambdainv_h_star( Lambdainv_h_vec ):
        return Lambdainv_h_vec

    @cached
    def Lambda_h_star( Lambdainv_h_star ):
        return inv( Lambdainv_h_star )

    @cached
    def RhT( T_star, dims_h, expand_h ):
        return expand_h( eye(T_star), dims_h )

    @cached
    def diag_Lambda_h( Lambda_h_star, dims_h, RhT, expand_h ):
        return np.sum( expand_h( Lambda_h_star, dims_h ) * RhT, axis = 1 )

    @cached
    def expected_log_g( log_g ):
        return log_g

    @cached
    def MAP_log_g( log_g ):
        return log_g

    is_point_estimate = False
    # NB: This is ignored for 'soft' nonlinearity_h, as can only compute
    # MAP values here at present

    @cached( cskip = [
        ('nonlinearity_h', 'soft', ['diag_Lambda_h', 'h']),
        ('is_point_estimate', True, ['diag_Lambda_h', 'h']) ] )
    def expected_g( nonlinearity_h, is_point_estimate, g, h, diag_Lambda_h ):
        if is_point_estimate or (nonlinearity_h == 'soft'):
            return g
        else:
            return exp( h + 0.5 * diag_Lambda_h )

    @cached
    def MAP_g( g ):
        return g

    @cached
    def logdet_Lambdainv_h_star( Lambdainv_h_star ):
        return logdet( Lambdainv_h_star )

    @cached
    def evidence_components__h( LL_training__h, logdet_Ch_star, 
            logdet_Lambdainv_h_star, h_star, Lhinv_star ):
        return A([ 
            LL_training__h, 
            -0.5 * ( logdet_Ch_star + logdet_Lambdainv_h_star ),
            -0.5 * dot( h_star, Lhinv_star * h_star ) ])
    
    @cached
    def evidence__h( evidence_components__h ):
        return np.sum( evidence_components__h )

    @cached
    def E_star__h( mu__h, resid__h, dFh_on_Fh, d2Fh_on_Fh, dims_h, 
            posterior_k, is_posterior, nonlinearity_h, data,
            zero_during_testing ):
        """ Hessian of neg log likelihood, for local evidence approx """
        import sandwich
        if not is_posterior:
            raise TypeError('attribute only available for `h` posterior.')
        # construct hessian of neg log likelihood
        if nonlinearity_h == 'exp':
            neg_d2LL_n = mu__h
        else:
            neg_d2LL_n = -resid__h * d2Fh_on_Fh + data.y * ( dFh_on_Fh ** 2 )
        neg_d2LL_n = zero_during_testing( neg_d2LL_n )
        neg_d2LL_n_zero_padded = np.hstack([ neg_d2LL_n, np.zeros_like(neg_d2LL_n) ])
        yy = np.fft.fft( neg_d2LL_n_zero_padded )
        F = ( np.sum( dims_h ) - 1 ) / 2
        E_star__h = sandwich.calzino( yy.real, yy.imag, F ) / len(yy)
        E_star__h[ 0, 1: ] /= sqrt(2)
        E_star__h[ 1:, 0 ] *= sqrt(2)
        return E_star__h

    """
    ===========================
    Local evidence calculations
    ===========================

    At the current `theta_h` (which is a candidate solution, and so is denoted
    `theta_h_c`). Note that `star` here refers to the dim reduction induced 
    by `theta_h_n`, not `theta_h_c`, so dim reduction is not handled by the 
    same attributes as the standard `theta_h`.

    """

    @cached
    def Lh_c_star( Lh, posterior_h ):
        """ Freq-domain diagonal cov matrix at candidate theta_h_c. """
        return Lh[ posterior_h.dims_h ]

    @cached
    def Lh_c_star_is_singular( Lh_c_star, _cutoff_lambda_h ):
        """ Is candidate diagonal cov matrix near-singular. """
        cutoff = _cutoff_lambda_h * maxabs(Lh_c_star)
        return ( Lh_c_star < cutoff ).any()

    @cached
    def dims_h_c_star( Lh_c_star, _cutoff_lambda_h, Lh_c_star_is_singular ):
        """ How to dim reduce from star- to plus-space. """
        if Lh_c_star_is_singular:
            cutoff = _cutoff_lambda_h * maxabs(Lh_c_star)
            return ( Lh_c_star >= cutoff )
        else:
            return np.ones( len(Lh_c_star), dtype=bool )

    @cached
    def dims_h_c( Lh_c_star_is_singular, dims_h_c_star, posterior_h ):
        """ How to dim reduce from full- to plus-space. """
        if Lh_c_star_is_singular:
            dims_h_c = posterior_h.dims_h.copy()
            dims_h_c[ dims_h_c ] = dims_h_c_star
            return dims_h_c
        else:
            return posterior_h.dims_h

    @cached
    def Lh_c_plus( Lh_c_star, dims_h_c_star, Lh_c_star_is_singular ):
        if Lh_c_star_is_singular:
            return Lh_c_star[ dims_h_c_star ]
        else:
            return Lh_c_star

    @cached
    def T_plus( dims_h_c ):
        return np.sum( dims_h_c )

    @cached
    def E_n_plus__h( posterior_h, dims_h_c_star, Lh_c_star_is_singular ):
        if Lh_c_star_is_singular:
            idx = dims_h_c_star
            return posterior_h.E_star__h[idx, :][:, idx]
        else:
            return posterior_h.E_star__h

    @cached
    def Lambdainv_n_plus__h(posterior_h, dims_h_c_star, Lh_c_star_is_singular):
        if Lh_c_star_is_singular:
            idx = dims_h_c_star
            return posterior_h.Lambdainv_h_star[idx, :][:, idx]
        else:
            return posterior_h.Lambdainv_h_star

    @cached
    def h_n_plus( posterior_h, dims_h_c_star, Lh_c_star_is_singular ):
        if Lh_c_star_is_singular:
            return posterior_h.h_star[ dims_h_c_star ]
        else:
            return posterior_h.h_star

    """ Approximate posterior at candidate theta """

    @cached
    def Lambdainv_c_plus__h( E_n_plus__h, Lh_c_plus ):
        return E_n_plus__h + diag( 1. / Lh_c_plus )

    @cached
    def Lambda_c_plus__h( Lambdainv_c_plus__h ):
        return inv( Lambdainv_c_plus__h )

    @cached
    def h_c_plus( Lambdainv_c_plus__h, Lambdainv_n_plus__h, h_n_plus ):
        return ldiv( 
                Lambdainv_c_plus__h, 
                dot(Lambdainv_n_plus__h, h_n_plus) )

    @cached
    def h_c( h_c_plus, dims_h_c, expand_h ):
        return expand_h( h_c_plus, dims_h_c )

    @cached
    def exph_c( h_c ):
        return exp(h_c)
    
    @cached
    def Fh_c( exph_c, nonlinearity_h ):
        if nonlinearity_h == 'exp':
            return exph_c
        elif nonlinearity_h == 'soft':
            return log( 1 + exph_c ) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached
    def dFh_c( exph_c, nonlinearity_h ):
        if nonlinearity_h == 'exp':
            raise AssertionError('should not reach this point')
            return exph
        elif nonlinearity_h == 'soft':
            return exph_c / (1 + exph_c) / log(2)
        else:
            self._raise_bad_nonlinearity()

    @cached
    def mu_c__h( Fh_c, posterior_k ):
        return Fh_c * posterior_k.expected_FXk

    @cached
    def log_g_c( nonlinearity_h, h_c, Fh_c ):
        if nonlinearity_h == 'exp':
            return h_c
        else:
            return log( Fh_c )


    """ Local evidence at candidate theta """

    @cached
    def local_evidence_components__h( Lh_c_plus, E_n_plus__h, T_plus, h_c_plus, 
            log_g_c, mu_c__h, y_training, h_c, slice_by_training, posterior_k ):
        # log likelihood
        mu = slice_by_training( mu_c__h )
        log_mu = slice_by_training( log_g_c + posterior_k.expected_log_FXk )
        LL = -np.sum( mu ) + dot( y_training, log_mu )
        if len( h_c_plus ) == 0:
            return LL
        # psi(theta)
        psi = A([ 
            LL,
            -0.5 * logdet( Lh_c_plus[:, na] * E_n_plus__h + eye(T_plus) ),
            -0.5 * dot( h_c_plus, h_c_plus / Lh_c_plus ) ])
        return psi

    @cached
    def local_evidence__h( local_evidence_components__h ):
        return np.sum( local_evidence_components__h )

    @cached
    def _LE_objective__h( local_evidence__h ):
        return -local_evidence__h

    @cached( cskip = [
        ('nonlinearity_h', 'exp', ['Fh_c', 'dFh_c']) ] )
    def _LE_jacobian__h( dLh_dtheta, N_theta_h, dims_h_c, data, mu_c__h,
            zero_during_testing, collapse_h, Lh_c_plus, E_n_plus__h,
            Lambda_c_plus__h, Lambdainv_c_plus__h, h_c_plus,
            nonlinearity_h, dFh_c, Fh_c ):
        # project to +-space
        dLh_dtheta_plus = [ dLhi[dims_h_c] for dLhi in dLh_dtheta ]
        # residuals at candidate solution
        resid = data.y - mu_c__h
        resid = zero_during_testing( resid ) # only training data
        # intermediate quantities ( all in +-space )
        if nonlinearity_h == 'exp':
            dLL_dh_c = collapse_h( resid, dims_h_c, padding='zero' )
        else:
            dLL_dh_c = collapse_h( resid * dFh_c / Fh_c, dims_h_c, padding='zero' )
        C_En = Lh_c_plus[:, na] * E_n_plus__h
        Linv_c = 1. / Lh_c_plus
        Cinv_c = diag( Linv_c )
        Lam_Cinv = Lambda_c_plus__h * Linv_c[na, :]
        hT_En_minus_Cinv = dot( h_c_plus.T, E_n_plus__h - Cinv_c )
        Cinv2_h = Linv_c * Linv_c * h_c_plus
        # calculate for each variable in theta
        dpsi = np.empty( N_theta_h )
        # derivatives wrt `theta_h` variables
        for j in range( N_theta_h ):
            dLh = dLh_dtheta_plus[j]
            B = Lam_Cinv * ( dLh * Linv_c )[na, :]
            Bh = ldiv( Lambdainv_c_plus__h, dLh * Cinv2_h )
            dpsi[j] = sum([
                dot( dLL_dh_c, Bh ),
                -0.5 * np.trace( dot( B, C_En ) ),
                0.5 * dot( hT_En_minus_Cinv, Bh ) ])
        # make negative
        return -dpsi

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for `k`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    @cached
    def k_star( k_vec ):
        return k_vec

    """
    ================
    Log Prior on `k`
    ================
    """

    @cached
    def Ck( Ck_is_diagonal ):
        if Ck_is_diagonal:
            return None
        else:
            raise TypeError('must subclass to define `Ck` behaviour')
        
    @cached
    def dCk_dtheta():
        raise TypeError('must subclass to define `dCk_dtheta` behaviour')

    @cached
    def Ck_eig( Ck, Ck_is_diagonal ):
        if Ck_is_diagonal:
            return None
        try:
            Lk, Qk = eigh( Ck )
        except np.linalg.LinAlgError:
            Lk, Qk = eigh( Ck * 1e4 )
            Lk /= 1e4
        return {'Lk':Lk, 'Qk':Qk}

    @cached( cskip = ('Ck_is_diagonal', True, 'Ck_eig') )
    def Lk( Ck_is_diagonal, Ck_eig ):
        if Ck_is_diagonal:
            raise TypeError('must subclass to define `Lk` behaviour')
        else:
            return Ck_eig['Lk']
       
    @cached
    def dLk_dtheta():
        raise TypeError('must subclass to define `dLk_dtheta` behaviour')

    @cached
    def dims_k( Lk, _cutoff_lambda_k ):
        return ( Lk/maxabs(Lk) > _cutoff_lambda_k )

    @cached
    def D_star( dims_k ):
        return np.sum( dims_k )

    @cached
    def required_k_vec_length( D_star ):
        return D_star

    @cached
    def Rk( Ck_eig, dims_k, Ck_is_diagonal ):
        return Ck_eig['Qk'][ :, dims_k ].T

    @cached( cskip=('Ck_is_diagonal', True, ['Ck_eig', 'Rk', 'D_star']) )
    def Rk_is_identity( Ck_is_diagonal, Ck_eig, dims_k, Rk, D_star ):
        if Ck_is_diagonal:
            return np.all(dims_k) 
        else:
            return ( np.all(dims_k) and np.array_equal(Rk, eye(D_star)) )

    @cached( cskip=[ 
        ('Rk_is_identity', True, ['dims_k', 'Rk']),
        ('Ck_is_diagonal', True, 'Rk') ] )
    def X_star( Rk_is_identity, Ck_is_diagonal, data, dims_k, Rk ):
        if Rk_is_identity:
            return data.X
        elif Ck_is_diagonal:
            return data.X[:, dims_k]
        else:
            return dot( data.X, Rk.T )

    @cached
    def Lk_star( Lk, dims_k ):
        return Lk[ dims_k ]

    @cached
    def Lkinv_star( Lk_star ):
        return 1. / Lk_star

    @cached
    def logdet_Ck_star( Lk_star ):
        return np.sum(log(Lk_star))

    @cached( cskip=[
        ('Rk_is_identity', True, ['dims_k', 'Rk', 'D']),
        ('Ck_is_diagonal', True, ['Rk']) ])
    def k( k_star, Rk_is_identity, Ck_is_diagonal, dims_k, Rk, D ):
        if Rk_is_identity:
            return k_star
        elif Ck_is_diagonal:
            k = zeros( D )
            k[ dims_k ] = k_star
            return k
        else:
            return dot( Rk.T, k_star )

    def _reproject_to_k_vec( self, k=None, posterior=None ):
        """ Calculates `k_vec` from `k`. Does not change object's state. 
        
        Can either provide `k` *or* a posterior which contains this attribute.
        
        """
        # check inputs
        if [k, posterior].count(None) != 1:
            raise ValueError('either provide `k` or posterior')
        # provide a posterior
        elif posterior is not None:
            return self._reproject_to_k_vec( k=posterior.k )
        # provide a `k` value
        elif k is not None:
            if self.Rk_is_identity:
                return k
            elif self.Ck_is_diagonal:
                return k[ self.dims_k ]
            else:
                return dot( self.Rk, k )

    """
    ====================
    Log posterior on `k`
    ====================
    """

    @cached
    def log_FXk( X_star, k_star ):
        return dot( X_star, k_star )

    @cached
    def FXk( log_FXk ):
        return exp( log_FXk )

    @cached
    def mu__k( posterior_h, FXk ):
        return posterior_h.expected_g * FXk

    @cached
    def log_mu__k( posterior_h, log_FXk ):
        return posterior_h.expected_log_g + log_FXk

    @cached
    def LL_training__k( y_training, slice_by_training, mu__k, log_mu__k ):
        s = slice_by_training
        mu, log_mu = s(mu__k), s(log_mu__k)
        return -np.sum( mu ) + dot( y_training, log_mu )

    @cached
    def LPrior__k( logdet_Ck_star, k_star, Lkinv_star ):
        return sum([
            -0.5 * logdet_Ck_star, 
            -0.5 * dot( k_star.T, Lkinv_star * k_star ) ])

    @cached
    def LP_training__k( LL_training__k, LPrior__k ):
        return LL_training__k + LPrior__k

    """ Jacobian """

    @cached
    def resid__k( mu__k, data, zero_during_testing ):
        return zero_during_testing( data.y - mu__k )

    @cached
    def dLL_training__k( resid__k, X_star ):
        """ Jacobian of training log likelihood wrt `k_vec`. """
        return dot( X_star.T, resid__k )

    @cached
    def dLPrior__k( Lkinv_star, k_star ):
        return -Lkinv_star * k_star

    @cached
    def dLP_training__k( dLL_training__k, dLPrior__k ):
        """ Jacobian of training log prob wrt `k_vec`. """
        return dLL_training__k + dLPrior__k

    """ Hessian """

    @cached
    def d2LP_training__k( mu__k, D_star, slice_by_training,
            y_training, X_star, Lkinv_star ):
        """ Hessian of training log prob wrt `k_vec`. """
        # training data only
        s = slice_by_training
        y = y_training
        mu, X_star = s(mu__k), s(X_star)
        # evaluate d2LL
        mu_X_star = mu[:, na] * X_star
        d2LP = -dot( X_star.T, mu[:, na] * X_star )
        # add d2LPrior to the diagonal
        d2LP[ range(D_star), range(D_star) ] -= Lkinv_star
        return d2LP
    
    """
    ================
    Solution for `k`
    ================
    """

    @cached
    def _negLP_objective__k( LP_training__k ):
        return -LP_training__k

    @cached
    def _negLP_jacobian__k( dLP_training__k ):
        return -dLP_training__k

    @cached
    def _negLP_hessian__k( d2LP_training__k ):
        return -d2LP_training__k

    def _negLP_callback__k( self, **kw ):
        try:
            offset = self.posterior_h.LPrior__h
        except AttributeError:
            offset = 0
        LP_total = self.LP_training__k + offset
        self.announce( 'LP step:  %.3f' % LP_total, n_steps=4, **kw )

    """
    ===============================
    Posterior_k specific attributes
    ===============================

    These have dependencies on `k_star` and `Lambdainv_k_star`

    """
    
    @cached
    def Lambdainv_k_vec( _negLP_hessian__k, is_posterior ):
        if not is_posterior:
            raise TypeError('attribute only available for `k` posterior.')
        return _negLP_hessian__k

    @cached
    def Lambdainv_k_star( Lambdainv_k_vec ):
        return Lambdainv_k_vec

    @cached
    def Lambda_k_star( Lambdainv_k_star ):
        return inv( Lambdainv_k_star )

    @cached( cskip = [
        ('Rk_is_identity', True, ['Rk', 'D', 'dims_k']),
        ('Ck_is_diagonal', True, 'Rk') ])
    def Lambda_k(Lambda_k_star, Rk_is_identity, Ck_is_diagonal, dims_k, Rk, D):
        if Rk_is_identity:
            return Lambda_k_star
        elif Ck_is_diagonal:
            Lambda_k = zeros( (D, D) )
            dim_idxs = np.nonzero( dims_k )[0]
            for ii, dd in enumerate( dim_idxs ):
                Lambda_k[ dd, dims_k ] = Lambda_k_star[ ii, : ]
            return Lambda_k
        else:
            return mdot( Rk.T, Lambda_k_star, Rk )

    @cached
    def MAP_log_FXk( X_star, k_star ):
        return dot( X_star, k_star )

    @cached
    def expected_log_FXk( MAP_log_FXk ):
        return MAP_log_FXk

    @cached
    def MAP_FXk( expected_log_FXk ):
        return exp( expected_log_FXk )

    @cached( cskip = [
        ('is_point_estimate', True, ['X_star', 'Lambda_k_star', 'MAP_log_FXk']),
        ('is_point_estimate', False, 'MAP_FXk') ])
    def expected_FXk( is_point_estimate, MAP_log_FXk, MAP_FXk, X_star, 
            Lambda_k_star ):
        if is_point_estimate:
            return MAP_FXk
        else:
            return exp( MAP_log_FXk 
                    + 0.5 * np.sum(X_star.dot(Lambda_k_star) * X_star, axis=1))

    @cached
    def logdet_Lambdainv_k_star( Lambdainv_k_star ):
        return logdet( Lambdainv_k_star )

    @cached
    def evidence_components__k( LL_training__k, logdet_Ck_star, 
            logdet_Lambdainv_k_star, k_star, Lkinv_star ):
        return A([ 
            LL_training__k, 
            -0.5 * ( logdet_Ck_star + logdet_Lambdainv_k_star ),
            -0.5 * dot( k_star, Lkinv_star * k_star ) ])
    
    @cached
    def evidence__k( evidence_components__k ):
        return np.sum( evidence_components__k )

    @cached
    def E_star__k( X_star, mu__k, is_posterior ):
        """ Hessian of neg log likelihood, for local evidence approx """
        if not is_posterior:
            raise TypeError('attribute only available for `k` posterior.')
        return dot( X_star.T, mu__k[:, na] * X_star )

    """
    ===========================
    Local evidence on `k`
    ===========================

    At the current `theta_k` (which is a candidate solution, and so is denoted
    `theta_k_c`). Note that `star` here refers to the dim reduction induced 
    by `theta_k_n`, not `theta_k_c`, so dim reduction is not handled by the 
    same attributes as the standard `theta_k`.

    """

    @cached
    def Ck_c_star( posterior_k, Ck ):
        """ Covariance at theta_k_c, in *-space induced by theta_k_n. """
        pk = posterior_k
        if pk.Rk_is_identity:
            return Ck
        else:
            return mdot( pk.Rk, Ck, pk.Rk.T )

    @cached
    def Ck_c_eig( Ck_c_star ):
        try:
            Lk, Qk = eigh( Ck_c_star )
        except np.linalg.LinAlgError:
            Lk, Qk = eigh( Ck_c_star * 1e4 )
            Lk /= 1e4
        return {'Lk_c_star':Lk, 'Qk_c_star':Qk}

    @cached( cskip=[('Ck_is_diagonal', True, 'Ck_c_eig')] )
    def Lk_c_star( Ck_is_diagonal, posterior_k, Lk, Ck_c_eig ):
        """ Diagonalised covariance at theta_k_c, in *-space """
        pk = posterior_k
        if Ck_is_diagonal:
            return Lk[ pk.dims_k ]
        else:
            return Ck_c_eig['Lk_c_star']

    @cached
    def Lk_c_star_is_zero( Lk_c_star ):
        return ( maxabs( Lk_c_star) == 0 )

    @cached
    def dims_k_c_star( Lk_c_star, _cutoff_lambda_k ):
        """ Bool array of dimensions to keep in *-space. """
        cutoff = _cutoff_lambda_k * maxabs(Lk_c_star)
        return ( Lk_c_star >= cutoff )

    @cached
    def Lk_c_star_is_singular( dims_k_c_star ):
        """ Is a second dim reduction necessary. """
        return not dims_k_c_star.all()

    @cached
    def second_dr_k( Ck_is_diagonal, Lk_c_star_is_singular ):
        """ Type of second dim reduction to apply. """
        if Ck_is_diagonal:
            if Lk_c_star_is_singular:
                return 'project'
            else:
                return 'none'
        else:
            if Lk_c_star_is_singular:
                return 'full'
            else:
                return 'rotate'

    @cached
    def Rk_c_star( Ck_c_eig, dims_k_c_star ):
        return Ck_c_eig['Qk_c_star'][:, dims_k_c_star].T

    @cached( cskip=[
        ('second_dr_k', 'none', 'Ck_c_eig'), 
        ('second_dr_k', 'rotate', 'Ck_c_eig'), 
        ('second_dr_k', 'project', 'Ck_c_eig') ])
    def Lk_c_plus( second_dr_k, Ck_c_eig, Lk_c_star, dims_k_c_star ):
        if second_dr_k in ['none', 'rotate']:
            return Lk_c_star
        elif second_dr_k == 'project':
            return Lk_c_star[ dims_k_c_star ]
        else:
            return Ck_c_eig['Lk_c_star'][dims_k_c_star]

    @cached
    def Ck_c_plus( Lk_c_plus ):
        return diag( Lk_c_plus )

    @cached
    def Ckinv_c_plus( Lk_c_plus ):
        return diag( 1. / Lk_c_plus )

    @cached
    def D_plus( dims_k_c_star ):
        return np.sum(dims_k_c_star)

    @cached( cskip = [
        ('second_dr_k', 'none', 'Rk_c_star'),
        ('second_dr_k', 'project', 'Rk_c_star') ])
    def E_n_plus__k( second_dr_k, posterior_k, dims_k_c_star, Rk_c_star ):
        Ek = posterior_k.E_star__k
        if second_dr_k == 'none':
            return Ek
        elif second_dr_k == 'project':
            idx = dims_k_c_star
            return Ek[idx, :][:, idx]
        else: 
            return mdot( Rk_c_star, Ek, Rk_c_star.T )

    @cached( cskip = [
        ('second_dr_k', 'none', 'Rk_c_star'),
        ('second_dr_k', 'project', 'Rk_c_star') ])
    def Lambdainv_n_plus__k(second_dr_k, posterior_k, dims_k_c_star, Rk_c_star):
        Laminv = posterior_k.Lambdainv_k_star
        if second_dr_k == 'none':
            return Laminv
        elif second_dr_k == 'project':
            idx = dims_k_c_star
            return Laminv[idx, :][:, idx]
        else: 
            return mdot( Rk_c_star, Laminv, Rk_c_star.T )

    @cached( cskip = [
        ('second_dr_k', 'none', 'Rk_c_star'),
        ('second_dr_k', 'project', 'Rk_c_star') ])
    def k_n_plus( second_dr_k, posterior_k, dims_k_c_star, Rk_c_star ):
        k_star = posterior_k.k_star
        if second_dr_k == 'none':
            return k_star
        elif second_dr_k == 'project':
            return k_star[ dims_k_c_star ]
        else: 
            return dot( Rk_c_star, k_star )

    @cached( cskip = [
        ('second_dr_k', 'none', 'Rk_c_star'),
        ('second_dr_k', 'project', 'Rk_c_star') ])
    def X_plus( second_dr_k, posterior_k, dims_k_c_star, Rk_c_star ):
        X_star = posterior_k.X_star
        if second_dr_k == 'none':
            return X_star
        elif second_dr_k == 'project':
            return X_star[ :, dims_k_c_star ]
        else: 
            return dot( X_star, Rk_c_star.T )

    """ Approximate posterior at candidate theta """

    @cached
    def Lambdainv_c_plus__k( E_n_plus__k, Ckinv_c_plus ):
        return E_n_plus__k + Ckinv_c_plus

    @cached
    def Lambda_c_plus__k( Lambdainv_c_plus__k ):
        return inv( Lambdainv_c_plus__k )

    @cached
    def k_c_plus( Lambdainv_c_plus__k, Lambdainv_n_plus__k, k_n_plus ):
        return ldiv( 
                Lambdainv_c_plus__k, 
                dot(Lambdainv_n_plus__k, k_n_plus) )

    @cached( cskip = [
        ('second_dr_k', 'none', 'Rk_c_star'),
        ('second_dr_k', 'project', 'Rk_c_star') ])
    def k_c( k_c_plus, Ck_is_diagonal, second_dr_k, Rk_c_star, dims_k_c_star,
            D, posterior_k ):
        pk = posterior_k
        # to *-space
        if second_dr_k == 'none':
            k_c_star = k_c_plus
        elif second_dr_k == 'project':
            k_c_star = np.zeros( pk.D_star )
            k_c_star[ dims_k_c_star ] = k_c_plus
        else:
            k_c_star = dot( Rk_c_star.T, k_c_plus )
        # to full space
        if pk.Rk_is_identity:
            return k_c_star
        elif Ck_is_diagonal:
            k = np.zeros( D )
            k[ pk.dims_k ] = k_c_star
            return k
        else:
            return dot( pk.Rk.T, k_c_star )

    @cached
    def log_FXk_c( k_c_plus, X_plus ):
        return dot( X_plus, k_c_plus )

    @cached
    def FXk_c( log_FXk_c ):
        return exp( log_FXk_c )

    @cached
    def mu_c__k( FXk_c, posterior_h ):
        return FXk_c * posterior_h.expected_g

    """ Local evidence at candidate theta """

    @cached
    def local_evidence_components__k( mu_c__k, slice_by_training, log_FXk_c,
            y_training, Lk_c_plus, E_n_plus__k, D_plus, k_c_plus, posterior_h ):
        # log likelihood
        mu = slice_by_training( mu_c__k )
        log_mu = slice_by_training( log_FXk_c + posterior_h.expected_log_g )
        LL = -np.sum( mu ) + dot( y_training, log_mu )
        if ~np.isfinite( LL ):
            return -np.inf
        if len( k_c_plus ) == 0:
            return LL
        # psi(theta)
        psi = A([ 
            LL,
            -0.5 * logdet( Lk_c_plus[:, na] * E_n_plus__k + eye(D_plus) ),
            -0.5 * dot( k_c_plus, k_c_plus / Lk_c_plus ) ])
        return psi

    @cached
    def local_evidence__k( local_evidence_components__k ):
        return np.sum( local_evidence_components__k )

    @cached
    def _LE_objective__k( local_evidence__k ):
        return -local_evidence__k

    @cached( cskip=[ 
        ('Ck_is_diagonal', True, ['Rk_c_star', 'dCk_dtheta']),
        ('Ck_is_diagonal', False, ['dLk_dtheta']) ])
    def _LE_jacobian__k( Ck_is_diagonal, second_dr_k, 
            posterior_k, dCk_dtheta, dLk_dtheta, Rk_c_star, data, mu_c__k,
            X_plus, slice_by_training, Ck_c_plus, E_n_plus__k,
            Lambda_c_plus__k, Ckinv_c_plus, k_c_plus, dims_k_c_star,
            Lk_c_plus, N_theta_k, Lambdainv_c_plus__k ):
        # convenience
        pk = posterior_k
        # diagonal case
        if Ck_is_diagonal:
            # project to *-space
            if pk.Rk_is_identity:
                dLk_dtheta_star = dLk_dtheta
            else:
                dLk_dtheta_star = [ dLk[ pk.dims_k ] for dLk in dLk_dtheta ]
            # project to +-space
            if second_dr_k == 'none':
                dLk_dtheta_plus = dLk_dtheta_star
            else:
                dLk_dtheta_plus = [dLk[ dims_k_c_star ] for dLk in dLk_dtheta]

            # residuals at candidate solution
            resid = slice_by_training( data.y - mu_c__k )
            # intermediate quantities
            X_plus = slice_by_training( X_plus )
            dLL_dk_c = dot( X_plus.T, resid )
            C_En = Lk_c_plus[:, na] * E_n_plus__k
            Lam_Cinv = Lambda_c_plus__k / Lk_c_plus[na, :]
            kT_En_minus_Cinv = dot( 
                    k_c_plus.T, 
                    E_n_plus__k - diag( 1./Lk_c_plus ) )
            # calculate for each variable in theta
            dpsi = np.empty( N_theta_k )
            # derivatives wrt `theta_h` variables
            for j in range( N_theta_k ):
                dLk = dLk_dtheta_plus[j]
                B = Lam_Cinv * ( dLk / Lk_c_plus )[na, :]
                Bk = ldiv( 
                        Lambdainv_c_plus__k, 
                        k_c_plus * dLk / (Lk_c_plus ** 2) )
                dpsi[j] = sum([
                    dot( dLL_dk_c, Bk ),
                    -0.5 * np.trace( dot(B, C_En) ),
                    0.5 * dot( kT_En_minus_Cinv, Bk ) ])
            # make negative
            return -dpsi

        # non-diagonal case
        else:
            # project to *-space
            if pk.Rk_is_identity:
                dCk_dtheta_star = dCk_dtheta
            else:
                dCk_dtheta_star = [ mdot( pk.Rk, dCk, pk.Rk.T ) 
                        for dCk in dCk_dtheta ]
            # project to +-space
            dCk_dtheta_plus = [ mdot( Rk_c_star, dCk, Rk_c_star.T ) 
                    for dCk in dCk_dtheta_star ]

            # residuals at candidate solution
            resid = slice_by_training( data.y - mu_c__k )
            # intermediate quantities( all in +-space )
            X_plus = slice_by_training( X_plus )
            dLL_dk_c = dot( X_plus.T, resid )
            C_En = dot( Ck_c_plus, E_n_plus__k )
            Lam_Cinv = dot( Lambda_c_plus__k, Ckinv_c_plus )
            kT_En_minus_Cinv = dot( k_c_plus.T, E_n_plus__k - Ckinv_c_plus )
            # calculate for each variable in theta
            dpsi = np.empty( N_theta_k )
            # derivatives wrt `theta_h` variables
            for j in range( N_theta_k ):
                dCk = dCk_dtheta_plus[j]
                B = mdot( Lam_Cinv, dCk, Ckinv_c_plus )
                Bk = ldiv( 
                        Lambdainv_c_plus__k, 
                        mdot( Ckinv_c_plus, dCk, Ckinv_c_plus, k_c_plus ) )
                dpsi[j] = sum([
                    dot( dLL_dk_c, Bk ),
                    -0.5 * np.trace( dot(B, C_En) ),
                    0.5 * dot( kT_En_minus_Cinv, Bk ) ])
            # make negative
            return -dpsi

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    ftol_per_obs = 1e-5

    """ Solving: individual """

    def calc_posterior_h( self, **kw ):
        return self._calc_posterior_v( 'h', **kw )

    def calc_posterior_k( self, **kw ):
        return self._calc_posterior_v( 'k', **kw )

    def get_next_theta_h_n( self, **kw ):
        return self.get_next_theta_v_n( 'h', **kw )

    def get_next_theta_k_n( self, **kw ):
        return self.get_next_theta_v_n( 'k', **kw )

    def solve_theta_k( self, **kw ):
        return self._solve_theta_v( 'k', **kw )

    def solve_theta_h( self, **kw ):
        return self._solve_theta_v( 'h', **kw )

    """ Solving: combined """

    def calc_posterior( self, min_iterations=2, max_iterations=20, 
            verbose=1, ftol_per_obs=None, **kw ):
        """ Calculate posterior on `h` and `k`.

        Uses coordinate gradient descent, and iteratively solves for `h` then
        `k`, using the `calc_posterior_h` and `calc_posterior_k` methods.
        After every joint iteration, checks to see whether the overall
        log posterior has improved and terminates on convergence, or after the 
        number of iterations reaches the max.

        Keywords on fitting:

        - `min_iterations` / `max_iterations` 

        - `ftol_per_obs` : required improvement in log posterior (per time bin) 
            to assess convergence. Note that `xtol` still applies on the
            coordinate steps. If this is not provided, defaults to 
            `self.ftol_per_obs`.

        Verbosity levels:

        - 0 : print nothing
        - 1 : print LP step after each h/k combined iteration
        - 2 : print LP steps within each h/k step
        - None : change nothing

        """
        # verbosity
        self._announcer.thresh_allow( verbose, 1, 'calc_posterior' )
        self._announcer.thresh_allow( 
                verbose, 2, 'calc_posterior_h', 'calc_posterior_k' )
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # initialise
        LP_prev = -np.inf
        # loop
        for i in range(max_iterations):
            # solve for `h`
            self.calc_posterior_h( verbose=None, **kw )
            # solve for `k`
            self.calc_posterior_k( verbose=None, **kw )
            # current LP
            LP = ( self.posterior_k.LL_training__k + 
                    self.posterior_k.LPrior__k +
                    self.posterior_h.LPrior__h )
            if np.isfinite( LP ):                
                self.announce( 'LP iteration %d: %d' % (i, LP) )
            else:
                self.announce( 'LP iteration %d: -inf' % i )
            # time to stop
            if i >= min_iterations:
                dLP = (LP - LP_prev)
                if dLP < (ftol_per_obs * self.N_observations):
                    break
            # update for the next round
            LP_prev = LP
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 'calc_posterior', 
                'calc_posterior_h', 'calc_posterior_k' )

    def solve( self, max_iterations=3, grid_search_theta_h=True, 
            grid_search_theta_k=True, verbose=1, **kw ):
        """ Solve for all hyperparameters and parameters. 

        First iteratively solves for the hyperparameters `theta_k` and 
        `theta_h`. Then solves for the posterior on `k` and `h` given the max 
        marginal likelihood values of `theta_k` and `theta_h`.

        On the first iteration of the solver, a grid search is run on `theta_h`
        if `grid_search_theta_h` is True. If this is done, it will override the
        current value of `theta_h`. 
        
        Likewise for `grid_search_theta_k`.

        Keywords:

        - `max_iterations` : number of successive attempts to solve
            for `theta_k` and `theta_h`

        Verbosity levels:

        - 0 : print nothing
        - 1 : print evidence steps only
        - 2 : print evidence and posterior steps
        - None : change nothing

        """
        # verbosity
        self._announcer.thresh_allow( verbose, 1, 'solve', 'solve_theta_h', 
                'solve_theta_k', 'calc_posterior' )
        self._announcer.thresh_allow( 
                verbose, 2, 'calc_posterior_h', 'calc_posterior_k' )
        # get initial posterior 
        self.announce('Initial posterior')
        self.calc_posterior( verbose=None )
        # extract starting points for first cycle
        self._reset_k_vec()
        self._reset_h_vec()
        # cycle
        for i in range( max_iterations ):
            # solve for theta_k
            self.announce( 'iteration %d: solving for theta_k:' % i )
            self.solve_theta_k( verbose=None, grid_search=grid_search_theta_k,
                    **kw )
            # solve for theta_h
            self.announce( 'iteration %d: solving for theta_h:' % i )
            self.solve_theta_h( verbose=None, grid_search=grid_search_theta_h, 
                    **kw )
            # extract starting points for next cycle
            self._reset_k_vec()
            self._reset_h_vec()
            # ensure no more grid searches are run
            grid_search_theta_h = False
            grid_search_theta_k = False
        # restore verbosity
        self._announcer.allow( 
                'solve', 'solve_theta_h', 'solve_theta_k', 
                'calc_posterior', 'calc_posterior_h', 'calc_posterior_k' )

    """ Derivative checking """

    def _check_LP_derivatives__h( self, **kw ):
        return self._check_LP_derivatives__v( 'h', **kw )

    def _check_LE_derivatives__h( self, **kw ):
        return self._check_LE_derivatives__v( 'h', **kw )

    def _check_LP_derivatives__k( self, **kw ):
        return self._check_LP_derivatives__v( 'k', **kw )

    def _check_LE_derivatives__k( self, **kw ):
        return self._check_LE_derivatives__v( 'k', **kw )

    def _check_Lk_derivatives( self, eps=1e-6, debug=False,
            error_if_fail=False, error_threshold=0.05, hessian=False ):
        """ Check derivatives of diagonal of prior covariance wrt `theta_k`. 
        
        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective methods `_dLk_dtheta` and `_d2Lk_dtheta2`,
        and compares with empirical values, obtained from finite differences
        method on `_Lk` and `_dLk_dtheta` respectively.
        
        Typically, this prints out the size of the error between analytic
        and empirical estimates, as a norm. For example, for the Jacobian 
        (analytic `aJ` and empirical `eJ` respectively), this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        This tests the derivatives at the current value of `theta_k`.

        Keywords:

        - `theta_k` : value of hyperparameter at which to test derivative.
            (defaults to `self.theta_k0`)

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.

        - `hessian` : boolean. whether to check hessian
        
        """
        # can only run this if Ck is diagonal
        if not self.Ck_is_diagonal:
            raise TypeError('`Ck` is not diagonal: check `Lk` derivs instead.')
        # initial condition
        theta_k0 = self.theta_k
        N_theta_k = self.N_theta_k
        D = self.D
        # calculate L
        L = self.Lk
        # JACOBIAN
        # analytic Jacobian
        dL = A( self.dLk_dtheta )
        # helper function
        def dth(i):
            z = zeros( N_theta_k )
            z[i] = eps
            return z
        # empirical Jacobian
        edL = np.zeros( (N_theta_k, D) )
        for i in range(N_theta_k):
            self.theta_k = theta_k0 + dth(i)
            edL[i] = ( self.Lk - L ) / eps
        # print error
        err1 = norm( dL - edL ) / norm( dL )
        print ' '
        print 'dL norm deviation: %.6f' % err1
        print ' '
        # raise error?
        if error_if_fail and (err1 > error_threshold):
            raise ValueError('Jacobian of Ck failed at %.6f' % err1 )
        # HESSIAN
        if hessian:
            raise NotImplementedError()
        # debug
        if debug:
            tracer()

    def _check_Ck_derivatives( self, eps=1e-6, debug=False,
            error_if_fail=False, error_threshold=0.05, hessian=False ):
        """ Check derivatives of the prior covariance matrix wrt `theta_k`. 
        
        For both the Jacobian and Hessian, evaluates the analytic derivatives
        provided by the respective methods `_dCk_dtheta` and `_d2Ck_dtheta2`,
        and compares with empirical values, obtained from finite differences
        method on `_Ck` and `_dCk_dtheta` respectively.
        
        Typically, this prints out the size of the error between analytic
        and empirical estimates, as a norm. For example, for the Jacobian 
        (analytic `aJ` and empirical `eJ` respectively), this computes 
        norm(aJ - eJ) / norm(aJ). If required, a ValueError can be thrown
        if the deviation is too large.

        This tests the derivatives at the current value of `theta_k`.

        Keywords:

        - `theta_k` : value of hyperparameter at which to test derivative.
            (defaults to `self.theta_k0`)

        - `eps` : absolute size of step in finite difference method

        - `debug` : drop into IPython debugger after evaluation

        - `error_if_fail` : raise ValueError if the relative deviation
            is too large

        - `error_threshold` : if `error_if_fail`, only raise ValueError
            if the relative deviation is greater than this value.

        - `hessian` : whether to test the same for the hessian
        
        """
        # can only run this if Ck is not diagonal
        if self.Ck_is_diagonal:
            raise TypeError('`Ck` is diagonal: check `Lk` derivatives instead')
        # initial condition
        theta_k0 = self.theta_k
        N_theta_k = self.N_theta_k
        D = self.D
        # calculate C
        C = self.Ck
        # JACOBIAN
        # analytic Jacobian
        dC = A( self.dCk_dtheta )
        # helper function
        def dth(i):
            z = zeros( N_theta_k )
            z[i] = eps
            return z
        # empirical Jacobian
        edC = np.zeros( (N_theta_k, D, D) )
        for i in range(N_theta_k):
            self.theta_k = theta_k0 + dth(i)
            edC[i] = ( self.Ck - C ) / eps
        # print error
        err1 = norm( dC - edC ) / norm( dC )
        print ' '
        print 'dC norm deviation: %.6f' % err1
        print ' '
        # raise error?
        if error_if_fail and (err1 > error_threshold):
            raise ValueError('Jacobian of Ck failed at %.6f' % err1 )
        # HESSIAN
        if hessian:
            raise NotImplementedError()
        # debug
        if debug:
            tracer()

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Cross-validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    """
    =======
    For `h`
    =======
    """
    
    @cached
    def y_testing( data, slice_by_testing ):
        return slice_by_testing( data.y )

    @cached
    def LL_testing__h( slice_by_testing, mu__h, log_mu__h, y_testing ):
        s = slice_by_testing
        mu, log_mu = s(mu__h), s(log_mu__h)
        return -np.sum( mu ) + dot( y_testing, log_mu )

    @cached
    def LL_training_per_observation__h( LL_training__h, T_training ):
        return LL_training__h / T_training

    @cached
    def LL_testing_per_observation__h( LL_testing__h, T_testing ):
        return LL_testing__h / T_testing

    """
    =======
    For `k`
    =======
    """
    
    @cached
    def LL_testing__k( slice_by_testing, mu__k, log_mu__k, y_testing ):
        s = slice_by_testing
        mu, log_mu = s(mu__k), s(log_mu__k)
        return -np.sum( mu ) + dot( y_testing, log_mu )

    @cached
    def LL_training_per_observation__k( LL_training__k, T_training ):
        return LL_training__k / T_training

    @cached
    def LL_testing_per_observation__k( LL_testing__k, T_testing ):
        return LL_testing__k / T_testing

    """
    ===========================
    Linear interpolated version
    ===========================
    """

    @cached
    def _interp_h( h, testing_slices, T ):
        h = h.copy()
        for s in testing_slices:
            if s.start == 0:
                h[s] = h[s.stop]
            elif s.stop == T:
                h[s] = h[s.start]
            else:
                h[s] = np.linspace(h[s.start], h[s.stop - 1], s.stop - s.start)
        return h

    @cached
    def _interp_g( nonlinearity_h, _interp_h ):
        if nonlinearity_h == 'exp':
            return exp( _interp_h )
        elif nonlinearity_h == 'soft':
            return log( 1 + exp(_interp_h) )
        else:
            self._raise_bad_nonlinearity()

    @cached( cskip = [ ('nonlinearity_h', 'exp', ['_interp_g']) ] )
    def _interp_log_g( nonlinearity_h, _interp_h, _interp_g ):
        if nonlinearity_h == 'exp':
            return _interp_h
        else:
            return log( _interp_g )

    @cached
    def _interp_expected_log_g( _interp_log_g ):
        return _interp_log_g

    @cached
    def _interp_MAP_log_g( _interp_log_g ):
        return _interp_log_g

    @cached( cskip = [
        ('nonlinearity_h', 'soft', ['diag_Lambda_h', '_interp_h']),
        ('is_point_estimate', True, ['diag_Lambda_h', '_interp_h']),
        ('is_point_estimate', False, ['_interp_g']) ] )
    def _interp_expected_g( 
            nonlinearity_h, is_point_estimate, _interp_g, _interp_h, diag_Lambda_h ):
        if is_point_estimate or (nonlinearity_h == 'soft'):
            return _interp_g
        else:
            return exp( _interp_h + 0.5 * diag_Lambda_h )

    @cached
    def _interp_MAP_g( _interp_g ):
        return _interp_g

    @cached
    def _interp_mu__h( _interp_g, posterior_k ):
        return _interp_g * posterior_k.expected_FXk

    @cached
    def _interp_log_mu__h( _interp_log_g, posterior_k ):
        return _interp_log_g + posterior_k.expected_log_FXk

    @cached
    def _interp_LL_testing__h( 
            slice_by_testing, _interp_mu__h, _interp_log_mu__h, y_testing ):
        s = slice_by_testing
        mu, log_mu = s(_interp_mu__h), s(_interp_log_mu__h)
        return -np.sum( mu ) + dot( y_testing, log_mu )

    @cached
    def _interp_LL_testing_per_observation__h( 
            _interp_LL_testing__h, T_testing ):
        return _interp_LL_testing__h / T_testing

    @cached
    def _interp_mu__k( posterior_h, FXk ):
        return posterior_h._interp_expected_g * FXk

    @cached
    def _interp_log_mu__k( posterior_h, log_FXk ):
        return posterior_h._interp_expected_log_g + log_FXk

    @cached
    def _interp_LL_testing__k( slice_by_testing, _interp_mu__k, 
            _interp_log_mu__k, y_testing ):
        s = slice_by_testing
        mu, log_mu = s(_interp_mu__k), s(_interp_log_mu__k)
        return -np.sum( mu ) + dot( y_testing, log_mu )

    @cached
    def _interp_LL_testing_per_observation__k( 
            _interp_LL_testing__k, T_testing ):
        return _interp_LL_testing__k / T_testing


"""
==============
Priors on `h`
==============
"""

class Prior_h( AutoReloader ):

    """ Superclass for priors on `h`. """

    pass


class ALDf_h( Prior_h ):

    """ ALDf prior on `h`. 
    
    The prior is defined by a Gaussian power spectrum. The mean (`fmean_h`) 
    and standard deviation (`fstd_h`) of this Gaussian are hyperparameters,
    as well as an additional ridge-like term `rho_h`.

    Note that in the frequency domain, the prior covariance matrix is diagonal.
    For this reason, methods and variables are defined in terms of `Lh` which 
    is the diagonal of `Ch`.

    For convenience purposes, frequencies are relative to the sampling 
    frequency, i.e. 0 is the DC term, 0.5 is the Nyquist frequency.
    
    """

    # variable names
    _hyperparameter_names_h = [ 'fmean_h', 'fstd_h', 'rho_h' ]
    _bounds_h = [ (-2, 2), (1e-6, 1e-2), (-20, 20) ]
    _default_theta_h0 = [ 0, 0.01, 0 ]

    """ Prior on `h_star`. """
    
    @cached
    def Lh( theta_h, abs_freqs_h ):
        """ Diagonal of the prior covariance matrix for `h`. """
        # parse input
        fm, fs, r = theta_h
        # absolute frequencies
        f = abs_freqs_h
        # calculate
        return exp( -0.5 / (fs ** 2) * (f - fm)**2 - r )

    @cached
    def dLh_dtheta( theta_h, abs_freqs_h, Lh ):
        """ Jacobian of the diagonal of the prior cov matrix for `h`. """
        # parse input
        fm, fs, r = theta_h
        # frequencies
        f = abs_freqs_h
        # calculate
        dLh_dfm = (fs ** -2) * (f - fm) * Lh
        dLh_dfs = (fs ** -3) * (f - fm)**2 * Lh
        dLh_dr = -Lh 
        return [ dLh_dfm, dLh_dfs, dLh_dr ]


class Lowpass_Gaussian_h( Prior_h ):

    """ Lowpass prior on `h`. 
    
    As with the ALDf prior, this is defined by a Gaussian power spectrum. 
    Here, the mean is assumed to be zero. The two hyperparameters are the
    standard deviation (`fstd_h`) of the Gaussian, and an additional 
    ridge-like term `rho_h`.

    Note that in the frequency domain, the prior covariance matrix is diagonal.
    For this reason, methods and variables are defined in terms of `Lh` which 
    is the diagonal of `Ch`.

    For convenience purposes, frequencies are relative to the sampling 
    frequency, i.e. 0 is the DC term, 0.5 is the Nyquist frequency.
    
    """

    # variable names
    _hyperparameter_names_h = [ 'log_fstd_h', 'rho_h' ]
    _max_len_h_star = 1000

    @property
    def _default_theta_h0( self ):
        return [ -np.log( self.T ) , -10. ]
   
    def _calc_bounds_h( self ):
        from scipy.optimize import fmin
        def f(s):
            self.theta_h = [ s, 0 ]
            return ( self.required_h_vec_length - self._max_len_h_star ) ** 2
        smax = fmin( f, -np.log(self.T), disp=0 )
        def f(s):
            self.theta_h = [ s, 0 ]
            return ( self.required_h_vec_length - 1 ) ** 2
        smin = fmin( f, -np.log(self.T), disp=0 )
        return [ (smin[0], smax[0]), (-40, 20) ]

    @cached( settable=True )
    def _bounds_h( _calc_bounds_h ):
        return _calc_bounds_h()

    @property
    def _grid_search_theta_h_parameters( self ):
        return {
            'bounds':[ list( self._bounds_h[0] ), [ -20, 10 ] ], 
            'spacing': [ 'linear', 'linear'], 
            'initial': self._default_theta_h0,
            'strategy': '1D two pass', 
            'grid_size': (10, 6) }

    """ Prior on `h_star`. """

    @cached
    def Lh( theta_h, abs_freqs_h ):
        """ Diagonal of the prior covariance matrix for `h`. """
        # parse input
        log_fs, r = theta_h
        fs = exp( log_fs )
        # calculate
        return exp( -0.5 / (fs ** 2) * (abs_freqs_h)**2 - r )

    @cached
    def dLh_dtheta( theta_h, abs_freqs_h, Lh ):
        """ Jacobian of the diagonal of the prior cov matrix for `h`. """
        # parse input
        log_fs, r = theta_h
        fs = exp( log_fs )
        # calculate
        dLh_dlogfs = (fs ** -2) * (abs_freqs_h)**2 * Lh
        dLh_dr = -Lh 
        return [ dLh_dlogfs, dLh_dr ]


class Lowpass_BlackmanHarris_h( Prior_h ):

    # variable names
    _hyperparameter_names_h = [ 'logFc_h', 'rho_h' ]

    @property
    def _default_theta_h0(self):
        return [ np.log(self.abs_freqs_h[50]), -10. ]

    _max_len_h_star = 1000

    @property
    def _bounds_h( self ):
        freqs = self.abs_freqs_h
        F_min = freqs[1] * 0.99999
        if len(freqs) < self._max_len_h_star:
            F_max = self._max_len_h_star
        else:
            F_max = freqs[ self._max_len_h_star // 2 ]
        return [ (np.log(F_min), np.log(F_max)), (-40, 20) ]

    @property
    def _grid_search_theta_h_parameters( self ):
        return {
            'bounds':[ tuple(self._bounds_h[0]), [ -20, 10 ] ], 
            'spacing': [ 'linear', 'linear'], 
            'initial': self._default_theta_h0,
            'strategy': '1D two pass', 
            'grid_size': (10, 6) }

    @cached
    def Lh( theta_h, abs_freqs_h ):
        """ Diagonal of the prior covariance matrix for `h`. """
        # parse input
        logFc, r = theta_h
        Fc = np.exp(logFc)
        f = abs_freqs_h
        # calculate
        n = np.arange(1, 4)
        L = np.cos( np.pi * n[na, :] * (1 + f[:, na] / Fc) )
        L = L.dot( A([-0.48829, 0.14128, -0.01168]) ) + 0.35875
        L[ np.abs(f) > Fc ] = 0
        return exp(-r) * L

    @cached
    def dLh_dtheta( theta_h, abs_freqs_h, Lh ):
        """ Jacobian of the diagonal of the prior cov matrix for `h`. """
        # parse input
        logFc, r = theta_h
        Fc = np.exp(logFc)
        f = abs_freqs_h
        # calculate
        n = np.arange(1, 4)
        pi = np.pi
        dL_dFc = np.sin( pi * n[na, :] * (1 + f[:, na] / Fc) )
        dL_dFc *= pi * n[na, :] * f[:, na] / (Fc**2) * exp(-r)
        dL_dFc = dL_dFc.dot( A([-0.48829, 0.14128, -0.01168]) ) 
        dL_dFc[ np.abs(f) > Fc ] = 0
        dL_dlogFc = dL_dFc * Fc
        dL_dr = -Lh 
        return [ dL_dlogFc, dL_dr ]


Lowpass_h = Lowpass_BlackmanHarris_h
#Lowpass_h = Lowpass_Gaussian_h


class Poisson_h( Prior_h ):

    """ No gain signal. """

    # variable names
    _hyperparameter_names_h = []
    _bounds_h = []
    _default_theta_h0 = []

    def _reproject_to_h_vec( self, *a, **kw ):
        return A([0.])

    def calc_posterior_h( self, *a, **kw ):
        p_kws = {}
        p_kws[ 'theta_h' ] = []
        p_kws[ 'h_vec' ] = A([0.])
        for vi in [n for n in self._variable_names if n != 'h' ]:
            try:
                p_kws[ 'posterior_' + vi ] = getattr(self, 'posterior_' + vi)
            except AttributeError:
                pass
        p = self.create_posterior( **p_kws )
        # save posterior object
        setattr( self, 'posterior_h', p )
        # save the value of h_vec
        p.csetattr( 'h_vec', A([0.]) )

    @cached 
    def LPrior__h():
        return 0.

    @property
    def h( self ):
        return zeros( self.T )

    @property
    def g( self ):
        return ones( self.T )

    @property
    def expected_g( self ):
        return self.g

    @property
    def MAP_g( self ):
        return self.g

    @property
    def expected_log_g( self ):
        return self.h

    @property
    def MAP_log_g( self ):
        return self.h

    @cached
    def Lh( T ):
        raise TypeError('should not get to this point')

    @cached
    def dLh_dtheta():
        raise TypeError('should not get to this point')



"""
===============
Grid searching
===============
"""

class GridSearcher( AutoReloader ):

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


"""
====================
Basic priors on `k`
====================
"""

class Prior_k( AutoReloader ):

    """ Superclass for priors on `k`. """
    
    Ck_is_diagonal = False


class Diagonal_Prior_k( Prior_k ):

    """ Sets non-diagonal methods to return None """

    Ck_is_diagonal = True

    @cached
    def Ck():
        raise TypeError('cannot compute `Ck`: diagonal covariance matrix')

    @cached
    def Ck_eig():
        raise TypeError('cannot compute `Ck_eig`: diagonal covariance matrix')

    @cached
    def Rk():
        raise TypeError('cannot compute `Rk`: diagonal covariance matrix')


class ML_k( Diagonal_Prior_k ):

    """ No prior on `k`, i.e. maximum likelihood. """

    # default variables
    _hyperparameter_names_k = []
    _bounds_k = []
    _default_theta_k0 = []

    Rk_is_identity = True
        
    """ Prior on `k` """

    @cached
    def Lk( D ):
        return np.ones( D ) * 1e12

    @cached
    def dLk_dtheta():
        return []


class Ridge_k( Diagonal_Prior_k ):

    """ Ridge prior on `k`. 
    
    This prior is defined in terms of a single hyperparameter, `rho_k`.

    """

    Rk_is_identity = True

    # default variables
    _hyperparameter_names_k = [ 'rho_k' ]
    _bounds_k = [ (-15, 15) ]
    _default_theta_k0 = [0.]
        
    _grid_search_theta_k_parameters = { 
            'bounds':[ [ -14, 14 ] ], 
            'spacing': [ 'linear' ], 
            'initial': A([ 0. ]), 
            'strategy': '1D', 
            'grid_size': (10,) }

    """ Prior on `k` """

    @cached
    def Lk( theta_k, D ):
        """ Diagonal of prior covariance matrix for `k`. """
        if np.iterable(theta_k):
            rho_k = theta_k[0]
        else:
            rho_k = theta_k
        return exp( -rho_k ) * ones( D )

    @cached
    def dLk_dtheta( Lk ):
        return [ -Lk ]


    

"""
========
Together
========
"""

class ML( ML_k, Lowpass_h, UnitSolver ):
    pass


class Ridge( Ridge_k, Lowpass_h, UnitSolver ):
    pass

