# SSplines
## Simplex Splines on the Powell-Sabin 12-split

[![Build Status](https://travis-ci.org/qTipTip/SSplines2.svg?branch=master)](https://travis-ci.org/qTipTip/SSplines2)
[![Coverage Status](https://coveralls.io/repos/github/qTipTip/SSplines2/badge.svg?branch=master)](https://coveralls.io/github/qTipTip/SSplines2?branch=master)

`SSplines` is a small Python library for the evaluation of simplex splines over
the Powell-Sabin 12-split of a triangle. The evaluation makes use of the
convenient matrix recurrence relation for the S-spline basis functions for
constant, linear and quadratic simplex splines as developed in [this
paper](http://www.ams.org/journals/mcom/2013-82-283/S0025-5718-2013-02664-6/S0025-5718-2013-02664-6.pdf) by
Cohen, Lyche and Reisenfeld.

## Functionality

At the moment, the SSpline library features:

1. `SplineFunction` objects representing a callable spline function over a
   single triangle, and the `SplineSpace` object facilitating instantiation of
   several functions in the same spline space.
2. Evaluation and differentiation of constant, linear and quadratic simplex
   splines with convenient short cuts for gradient, divergence and laplacian
   operators.
3. Conversion between quadratic S-spline basis and the quadratic Hermite nodal
   basis often employed in finite element methods.
4. A Method for sampling of triangles for ease of evaluation and visualization.
5. Some basic subdomain integration methods over the Powell--Sabin 12-split for
   use in finite element computations.
6. Methods for returning the polynomial restrictions of a spline to each of the
   twelve sub-triangles of the split.

## Installation

Installation through `pip` is not yet available, but will soon be supported.
The package can be installed locally by cloning the repository:

```bash
git clone https://github.com/qTipTip/SSplines2
```

The directory contains a setup-script, which can be run using
```python
python setup.py install
```
