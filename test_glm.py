from common import *

import glm
reload(glm)
from glm import *

from simulation import SimulatedData

from nose.plugins.attrib import attr
from scipy.stats import linregress

allclose = np.testing.assert_allclose
equal = np.testing.assert_equal
almost_equal = np.testing.assert_almost_equal
array_almost_equal = np.testing.assert_array_almost_equal


"""
===================
Helper functions
===================
"""

def test_data_normalise():
    """ Test that the normalisation and whitening of data works correctly."""
    X = normal( size=(200, 50) )
    y = poisson( size=(200) )
    data = Data( X, y, add_constant=False, normalise=True, whiten=False )
    # check that the mean is 0 and standard deviation is 1
    array_almost_equal( np.mean( data.X, axis=0 ), 0 )
    array_almost_equal( np.std( data.X, axis=0 ), 1 )
    # whiten
    data = Data( X, y, add_constant=False, whiten=True )
    array_almost_equal( dot( data.X.T, data.X ), eye(50) )


def test_training_testing_uniform_coverage():
    """ Test that cross-val regions have uniform coverage probability. """
    # set up test
    X = normal( size=(1000, 10) )
    X[:, 0] = np.arange(1000)
    y = np.arange(1000)
    data = Data( X, y, add_constant=False )
    s = Ridge( data )
    s.define_training_and_testing_regions( 0.2, 60 )
    # lengths of training and testing are correct
    assert s.T == 1000
    assert s.T_training == 800
    assert s.T_testing == 200
    # training and testing are disjoint sets
    count = zeros( 1000, dtype=int )
    for sl in s.testing_slices + s.training_slices:
        count[sl] += 1
    assert np.all( count == 1 )
    # slicing by training / testing works
    y_training = s.slice_by_training( y )
    y_testing = s.slice_by_testing( y )
    y_both = np.concatenate([ y_training, y_testing ])
    assert ( np.sort(y_both) == y ).all()
    X_training = s.slice_by_training( X )[:, 0]
    X_testing = s.slice_by_testing( X )[:, 0 ]
    X_both = np.concatenate([ X_training, X_testing ])
    assert ( np.sort(X_both) == X[:, 0] ).all()
    # uniform probability of coverage
    count = zeros( 1000, dtype=float )
    N_reps = 1000
    for _ in range(N_reps):
        s.define_training_and_testing_regions( 0.2, 60 )
        for sl in s.testing_slices:
            count[sl] += 1
    count /= N_reps
    assert np.std(count) < 0.05
    assert np.min(count) > 0.1
    assert np.max(count) < 0.3


"""
=======
Fitting
=======
"""

def test_simple_k_fitting():
    """ With constant gain, the fitted kernel must be close. Little data. """
    # create demo data
    d = SimulatedData( N_sec=10 )
    # fit ridge
    r = Ridge( d )
    r.calc_posterior()
    # check correlation between true and estimated `k`
    k_true = d.k_true
    k_hat = r.posterior.k__d
    c = np.corrcoef( k_true, k_hat )[0, 1]
    print 'correlation: %.2f' % c 
    assert c > 0.8
    return r

@attr('slow')
def test_simple_k_fitting_long():
    """ With constant gain, the fitted kernel must be close. Lots of data. """
    # create demo data
    d = SimulatedData( N_sec=40.96 )
    # fit ridge
    r = Ridge( d )
    r.solve()
    # check correlation between true and estimated `k`
    k_true = d.k_true
    k_hat = r.posterior.k__d
    c = np.corrcoef( k_true, k_hat )[0, 1]
    print 'correlation: %.2f' % c 
    assert c > 0.9

@attr('fast')
def test_ridge_vs_ml_k():
    d = SimulatedData( N_sec=30 )
    # fit ridge
    r = Ridge( d, testing_proportion=0.2, initial_conditions={'theta':[5]} )
    r.calc_posterior()
    # fit ml
    m = ML( d, initial_conditions={'theta':[]} )
    m.training_slices = r.training_slices
    m.testing_slices = r.testing_slices
    m.calc_posterior()
    # training LLs should be better on ML version (since overfit)
    training_dLL = (
            r.posterior.LL_training_per_observation - 
            m.posterior.LL_training_per_observation )
    assert training_dLL < 0
    # testing LLs should be better on Ridge version
    testing_dLL = (
            r.posterior.LL_testing_per_observation - 
            m.posterior.LL_testing_per_observation )
    #assert testing_dLL > 0
    print 'training/testing dLL :  %.1e / %.1e' % (training_dLL, testing_dLL)
    return m, r


"""
============================
Local evidence approximation
============================
"""

@attr('fast')
def test_evidence_and_LE_match():
    """ When `theta`=`theta_h`, the approximation of evidence is exact. """
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=5.12 )
    r = Ridge( d )
    r.theta = [5]
    r.calc_posterior()
    # there should be no second dim reduction
    assert r.second_dr_k == 'none'
    # local evidence: at same candidate theta, should get same solution
    r.theta = r.posterior.theta
    r.v = r.posterior.v
    array_almost_equal( r.k__d, r.k_c__d )
    array_almost_equal( r.mu__t, r.mu_c__t )
    array_almost_equal( 
            r.posterior.evidence_components, r.local_evidence_components )

"""
===========
Derivatives
===========
"""

@attr('fast')
def test_LP_derivatives():
    """ Matching analytic and empirical derivatives of log posterior. """
    kw = { 'error_if_fail':True, 'error_threshold':0.02, 'eps':1e-3 }
    for nl in progress.numbers( ['exp', 'soft'] ):
        # create demo data
        d = SimulatedData( N_sec=5 )
        r = Ridge( d, nonlinearity=nl, testing_proportion=0.2 )
        r.theta = [5]
        r.calc_posterior( verbose=0 )
        # check LP derivatives at random point
        r.v = normal( size=r.required_v_length )
        r.check_LP_derivatives( **kw )


@attr('slow')
def test_LE_derivatives():
    """ Matching analytic and empirical derivatives of log posterior. """
    kw = { 'error_if_fail':True, 'error_threshold':0.01 }
    for nl in progress.numbers( ['exp', 'soft'] ):
        # create demo data
        d = SimulatedData( N_sec=5 )
        r = Ridge( d, nonlinearity=nl )
        r.theta = [5]
        r.calc_posterior( verbose=0 )
        # check LE derivatives at theta_h_n = theta_h
        r.check_LE_derivatives( **kw )
        # check LE derivatives at another point
        r.theta_k = [1]
        r.check_LE_derivatives( **kw )

@attr('fast')
def test_ridge_derivatives():
    # create demo data
    d = SimulatedData( N_sec=5 )
    r = Ridge( d )
    r.theta = [5]
    r.check_l_derivatives( error_if_fail=True, error_threshold=0.01 )




if __name__ == '__main__':
    #r = test_h_fitting_mi()
    #r = test_simple_k_fitting_long()
    #m, r = test_ridge_vs_ml_k()
    #r = test_cross_validation()
    pass
