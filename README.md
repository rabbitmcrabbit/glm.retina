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
