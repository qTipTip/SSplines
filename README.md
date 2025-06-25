# SSplines

**Simplex Splines on the Powell-Sabin 12-split**

[![Build Status](https://travis-ci.org/qTipTip/SSplines.svg?branch=master)](https://travis-ci.org/qTipTip/SSplines)
[![Coverage Status](https://coveralls.io/repos/github/qTipTip/SSplines/badge.svg?branch=master)](https://coveralls.io/github/qTipTip/SSplines?branch=master)
[![Downloads](http://pepy.tech/badge/ssplines)](http://pepy.tech/project/ssplines)
[![DOI](https://zenodo.org/badge/121780435.svg)](https://doi.org/10.5281/zenodo.15742326)

SSplines is a Python library for the evaluation of simplex splines over the Powell-Sabin 12-split of a triangle. The library provides efficient evaluation using the matrix recurrence relation for S-spline basis functions for constant, linear, and quadratic simplex splines as developed by [Cohen, Lyche and Riesenfeld](http://www.ams.org/journals/mcom/2013-82-283/S0025-5718-2013-02664-6/S0025-5718-2013-02664-6.pdf).

## Features

- **SplineFunction objects**: Callable spline functions over a single triangle
- **SplineSpace objects**: Facilitate instantiation of multiple functions in the same spline space
- **Evaluation and differentiation**: Support for constant, linear, and quadratic simplex splines with convenient shortcuts for gradient, divergence, and Laplacian operators
- **Hermite basis conversion**: Conversion between quadratic S-spline basis and quadratic Hermite nodal basis for finite element methods
- **Triangle sampling**: Methods for sampling triangles for evaluation and visualization
- **Numerical integration**: Basic subdomain integration methods over the Powell-Sabin 12-split for finite element computations
- **Polynomial pieces**: Methods for returning polynomial restrictions of splines to each of the twelve sub-triangles
- **Symbolic computation**: Integration with SymPy for exact symbolic calculations

## Installation

### Using pip

```bash
pip install SSplines
```

### Using uv (recommended for development)

```bash
# Install uv first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install SSplines
uv pip install SSplines

# For development with testing and visualization
uv pip install "SSplines[dev]"
```

### Local installation

```bash
git clone https://github.com/qTipTip/SSplines
cd SSplines
uv pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import numpy as np
from SSplines import SplineSpace, SplineFunction

# Define a triangle
triangle = np.array([
    [0, 0],
    [1, 0], 
    [0, 1]
])

# Create a quadratic spline space
space = SplineSpace(triangle, degree=2)

# Get the dimension of the space
print(f"Spline space dimension: {space.dimension}")  # 12 for quadratic

# Create a spline function with random coefficients
coefficients = np.random.rand(space.dimension)
spline_func = space.function(coefficients)

# Evaluate the spline at points
points = np.array([
    [0.3, 0.3],
    [0.1, 0.2],
    [0.5, 0.1]
])
values = spline_func(points)
print(f"Spline values: {values}")
```

### Working with Basis Functions

```python
# Get all basis functions
basis_functions = space.basis()

# Evaluate the first basis function
first_basis = basis_functions[0]
value_at_point = first_basis([0.3, 0.3])

# Compute derivatives
gradient = first_basis.grad([0.3, 0.3])
laplacian = first_basis.lapl([0.3, 0.3])
```

### Hermite Basis

```python
# For quadratic splines, you can use Hermite basis
hermite_basis = space.hermite_basis()

# Each Hermite basis function corresponds to nodal values or derivatives
h0 = hermite_basis[0]  # Value at first vertex
h1 = hermite_basis[1]  # x-derivative at first vertex
h2 = hermite_basis[2]  # y-derivative at first vertex
```

### Symbolic Computation

```python
from SSplines.symbolic import polynomial_basis_quadratic

# Get symbolic polynomial representations
symbolic_basis = polynomial_basis_quadratic(triangle)

# Each entry contains the polynomial pieces over the 12 sub-triangles
print(f"Number of polynomial pieces: {len(symbolic_basis[0])}")  # 12
```

## Dependencies

- **Core dependencies**:
  - `numpy` ≥ 1.24.0: Numerical computations
  - `sympy` ≥ 1.10: Symbolic mathematics

- **Development dependencies**:
  - `pytest` ≥ 7.0.1: Testing framework
  - `pytest-cov`: Test coverage
  - `matplotlib` ≥ 3.5.1: Plotting and visualization

## Development

### Setting up a development environment

```bash
# Clone the repository
git clone https://github.com/qTipTip/SSplines
cd SSplines

# Install in development mode with all dependencies
uv pip install -e ".[dev]"
```

### Running tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov SSplines/

# Run specific test file
uv run pytest tests/test_spline_function.py -v
```

### Code structure

```
SSplines/
├── __init__.py              # Main module exports
├── constants.py             # Mathematical constants and lookup tables
├── helper_functions.py      # Core computational functions
├── simplex_spline.py        # SimplexSpline class
├── spline_function.py       # SplineFunction class  
├── spline_space.py          # SplineSpace class
├── symbolic.py              # Symbolic computation utilities
└── dicts.py                 # Lookup dictionaries for sub-triangles
```

## Mathematical Background

The library implements S-splines on the Powell-Sabin 12-split, which divides a triangle into 12 sub-triangles by:

1. Adding a point at the centroid
2. Adding midpoints on each edge  
3. Connecting these points to create 12 sub-triangles

The S-splines are defined using a matrix recurrence relation that allows efficient evaluation without explicit polynomial representation. This approach is particularly useful for:

- Finite element methods
- Computer-aided geometric design
- Approximation theory
- Scientific computing applications requiring smooth basis functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SSplines in your research, please cite the original theoretical work and the software implementation:

**Original S-splines theory:**
```bibtex
@article{cohen2013splines,
  title={S-splines},
  author={Cohen, Elaine and Lyche, Tom and Riesenfeld, Richard},
  journal={Mathematics of Computation},
  volume={82},
  number={283},
  pages={1577--1596},
  year={2013},
  doi={10.1090/S0025-5718-2013-02664-6}
}
```

**SSplines software and implementation:**
```bibtex
@mastersthesis{stangeby2018ssplines,
  title={Simplex Splines on the Powell-Sabin 12-Split},
  author={Stangeby, Ivar Haugal{\o}kken},
  school={University of Oslo},
  year={2018},
  url={http://hdl.handle.net/10852/64070},
  urn={URN:NBN:no-66606}
}
```

**Software repository:**
```bibtex
@software{stangeby2024ssplines,
  title={SSplines: Simplex Splines on the Powell-Sabin 12-split},
  author={Stangeby, Ivar},
  year={2024},
  doi={10.5281/zenodo.15742326},
  url={https://github.com/qTipTip/SSplines}
}
```


## Author

**Ivar Stangeby** - [istangeby@gmail.com](mailto:istangeby@gmail.com)

## Links

- [GitHub Repository](https://github.com/qTipTip/SSplines)
- [Documentation](https://github.com/qTipTip/SSplines)
- [PyPI Package](https://pypi.org/project/SSplines/)
- [Original Paper](http://www.ams.org/journals/mcom/2013-82-283/S0025-5718-2013-02664-6/S0025-5718-2013-02664-6.pdf)