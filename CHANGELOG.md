# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - 2025-06-26

### Breaking Changes
- **Python 3.13+ required**: Dropped support for Python < 3.13
- Modernized build system to use pyproject.toml instead of setup.py

### Changed
- Updated numpy to modern versions (>= 1.24.0)
- Migrated from setup.py to pyproject.toml
- Added automated CI/CD pipeline with GitHub Actions

### Dependencies
- numpy >= 1.24.0
- sympy >= 1.10

## [2.0.1] - 2018-09-03

### Fixed
- Various bug fixes and improvements
- Updated documentation

### Added
- Initial implementation of S-splines on Powell-Sabin 12-split
- SplineFunction and SplineSpace objects
- Support for constant, linear, and quadratic splines
- Hermite basis conversion
- Triangle sampling methods
- Basic integration methods