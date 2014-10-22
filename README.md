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

- The dimensionality of arrays is given through variable names: `X__td` is a (time x dimension) matrix; `y__t` is a (time) vector
- `X` denotes stimulus to regress against
- `y` denotes spike counts
- `k` denotes the linear kernel
- `F` denotes the output nonlinearity
- `v` denotes the set of all parameters to be learned, given the current setting of the hyperparameters (`theta`). Note that since the current hyperparameters might induce a dimensionality reduction, the cardinality of `v` may be smaller than `k`, and these may live in a different space from `k` (but we should have that `k` is a linear function of `v`).
- `theta` denotes hyperparameters. These might be fixed in advance, or learned through cross-validation or evidence optimisation.


Dimensionality reduction
------------------------

- Stimulus dimensionality is very high
- When priors are introduced, there are directions in kernel space that can be highly suppressed
- Rotating and projecting out these dimensions is very useful -- it brings substantial speed improvements, and stops the objective function from being ill-conditioned.
- Here, we use a matrix `R__de` to make this combined rotation/projection
- The dimension-reduced space we denote '*-space', and we use the suffix `__e` rather than `__d`
- A second dimensionality reduction is sometimes needed; we called this '+-space', and we use the suffix `__f`
