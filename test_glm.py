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
    d = SimulatedData( h_std=0, N_sec=30 )
    # fit ridge
    r = Ridge( d, testing_proportion=0.2, 
            initial_conditions={'theta_h':[-11, 4], 'theta_k':[5]} )
    r.calc_posterior()
    # fit ml
    m = ML( d, initial_conditions={'theta_h':[-11, 4], 'theta_k':[]} )
    m.training_slices = r.training_slices
    m.testing_slices = r.testing_slices
    m.calc_posterior()
    # calculate LLs
    m.LL_training = m.posterior_k.LL_training__k
    r.LL_training = r.posterior_k.LL_training__k
    # log likelihoods should be better on ML version (since overfit)
    assert m.LL_training > r.LL_training
    #m.LL_testing = m._LL( m.posterior, data='testing' )
    #r.LL_testing = r._LL( r.posterior, data='testing' )
    return m, r


"""
============================
Local evidence approximation
============================
"""

@attr('fast')
def test_evidence_and_LE_match_for_k():
    """ When `theta`=`theta_h`, the approximation of evidence is exact. """
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=5.12, h_mode='mi', h_std=1 )
    r = Ridge( d )
    r.theta_k = [5]
    r.theta_h = [-7, -10]
    r.calc_posterior()
    # there should be no second dim reduction
    assert r.second_dr_k == 'none'
    # local evidence: at same candidate theta_k, should get same solution
    r.theta_k = r.posterior_k.theta_k
    r.k_vec = r.posterior_k.k_vec
    array_almost_equal( r.k, r.k_c )
    array_almost_equal( r.mu__k, r.mu_c__k )
    array_almost_equal( 
            r.posterior_k.evidence_components__k, 
            r.local_evidence_components__k )

"""
===========
Derivatives
===========
"""

@attr('fast')
def test_LP_derivatives():
    """ Matching analytic and empirical derivatives of log posterior. """
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=5.12, h_mode='mi', h_std=1 )
    r = Ridge( d )
    r.theta_k = [5]
    r.theta_h = [-7, -10]
    r.calc_posterior()
    kw = { 'error_if_fail':True, 'error_threshold':0.01 }
    # check LP derivatives at random point
    r.h_vec = normal( size=r._required_h_vec_length )
    r.k_vec = normal( size=r._required_k_vec_length )
    r._check_LP_derivatives__h( **kw )
    r._check_LP_derivatives__k( **kw )


@attr('slow')
def test_LE_derivatives():
    """ Matching analytic and empirical derivatives of log posterior. """
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=5.12, h_mode='mi', h_std=1 )
    r = Ridge( d )
    r.theta_k = [5]
    r.theta_h = [-7, -10]
    r.calc_posterior()
    kw = { 'error_if_fail':True, 'error_threshold':0.01 }
    # check LE derivatives at theta_h_n = theta_h
    r._check_LE_derivatives__k( **kw )
    r._check_LE_derivatives__h( **kw )
    # check LE derivatives at another point
    r.theta_k = [1]
    r.theta_h = [-3, -11]
    r._check_LE_derivatives__k( **kw )
    r._check_LE_derivatives__h( **kw )

@attr('fast')
def test_ridge_k_derivatives():
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=5.12, h_mode='mi', h_std=1 )
    r = Ridge( d )
    r.theta_k = [5]
    r.theta_h = [-7, -10]
    r._check_Lk_derivatives( error_if_fail=True, error_threshold=0.01 )

"""
================
Cross-validation
================
"""

def test_cross_validation():
    # create demo data, with monotonically increasing `h`
    d = SimulatedData( N_sec=10.24, h_mode='mi', h_std=2 )
    # fit ridge, learning `theta_k` and `theta_h`
    r = Ridge( d, testing_proportion=0.2 )
    r.solve()
    # interpolated versions should be very similar in posterior_h
    near = np.testing.assert_array_almost_equal
    ph = r.posterior_h
    for k in ['h', 'g', 'log_mu__h', 'mu__h', 'LL_testing__h', 
            'LL_testing_per_observation__h']:
        near( getattr(ph, k), getattr(ph, '_interp_'+k), 1 )
    # interpolated versions should be very similar in posterior_k
    pk = r.posterior_k
    for k in ['log_mu__k', 'mu__k']:
        x1 = getattr(pk, k)[100:-100]
        x2 = getattr(pk, '_interp_'+k)[100:-100]
        dx = norm( x1 - x2 ) / norm(x1)
        assert dx < 1e-3
    # when theta_h is faster, we should do better on training, but worse
    # on prediction:
    r.theta_h = r.theta_h + A([2, 0])
    r.calc_posterior_h()
    assert r.LL_training__h > ph.LL_training__h
    #assert r.LL_testing__h < ph.LL_testing__h
    return r



if __name__ == '__main__':
    #r = test_h_fitting_mi()
    #r = test_simple_k_fitting_long()
    #m, r = test_ridge_vs_ml_k()
    #r = test_cross_validation()
    pass
