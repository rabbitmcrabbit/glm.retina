GLM toolbox for modelling retinal data
======================================

Design considerations
---------------------

- Object-oriented
- Unit-tested throughout
- Modular
- Inheritance patterns to allow for nested extensions
- Design to an interface (think: other people, including future-you, need to be able to use and develop this)
- Simulation: sample, then fit model
 

Useful patterns
---------------

- Cached computations: avoid repeated computations
- Runtime rebasing of objects on reload
- Hyperparameters should be explicit
- Cross-validation should be built-in (not separate code)
- Switchable components (e.g. nonlinearities, choice of optimisation function) should be modularised
- Methods for empirically checking derivatives


Model elements
--------------

- stimulus dimensionality is typically D=3 (x/y/t)
- spike history
- coupling
- slow modulator
- parameterised priors
- low-rank option


Conventions
-----------

- The dimensionality of arrays is given through variable names, e.g. `X_td` is a (time x dimension) matrix; `y_t` is a (time) vector
- `X` denotes stimulus to regress against
- `y` denotes spike counts
- `k` denotes the linear kernel
- `F` denotes the output nonlinearity
- `v` denotes the set of all parameters to be learned, given the current setting of the hyperparameters (`theta`). Note that since the current hyperparameters might induce a dimensionality reduction, the cardinality of `v` may be smaller than `k`.
- `theta` denotes hyperparameters. These might be fixed in advance, or learned through cross-validation or evidence optimisation.
