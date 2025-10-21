# Adaptive Fractionation for Overlapping Organs

[![Test Suite](https://github.com/YoelPH/Overlap_adaptfx/actions/workflows/tests.yml/badge.svg)](https://github.com/YoelPH/Overlap_adaptfx/actions/workflows/tests.yml)
[![Quick Tests](https://github.com/YoelPH/Overlap_adaptfx/actions/workflows/quick-tests.yml/badge.svg)](https://github.com/YoelPH/Overlap_adaptfx/actions/workflows/quick-tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Python package for adaptive radiotherapy fractionation that optimizes dose delivery based on real-time organ-at-risk overlap measurements during treatment planning and delivery.

## Overview

This package implements adaptive fractionation algorithms that dynamically adjust radiation dose delivery based on:
- Planning scan overlap measurements  
- Prescription dose constraints
- Penalty optimization for tumor coverage vs organ sparing

The adaptive approach improves treatment outcomes by maximizing dose when overlap is small and minimizing dose when overlap is large, compared to standard uniform fractionation.

## Installation

### From Source (Development)
```bash
git clone https://github.com/YoelPH/Overlap_adaptfx.git
cd Overlap_adaptfx
pip install -e .
```

## Quick Start

Check the `evaluation/Evaluation.ipynb` notebook for a comprehensive example of using the package with clinical data.

## Algorithm Details

The adaptive fractionation algorithm:

1. **Models overlap uncertainty** using probability distributions
2. **Optimizes dose selection** via dynamic programming with future value estimation
3. **Ensures constraint satisfaction** for clinical safety requirements

### Penalty Function
The algorithm minimizes a penalty function that increases with:
- Higher doses when overlap is large (organ sparing)
- Lower doses when overlap is small (tumor coverage)
- Deviations from prescription dose targets

## Project Structure

```
adaptive_fractionation_overlap/
├── __init__.py                 # Package initialization
├── constants.py                # Default parameters and constants  
├── core_adaptfx.py            # Main adaptive fractionation algorithms
└── helper_functions.py        # Mathematical utilities and penalties

tests/
├── conftest.py                # Test fixtures and shared data
├── test_constants.py          # Constants validation tests
├── test_helper_functions.py   # Mathematical function tests  
├── test_core_adaptfx.py       # Core algorithm tests
├── test_fixtures.py           # Test infrastructure validation
└── test_error_handling.py     # Edge cases and error conditions

evaluation/
├── Evaluation.ipynb           # Analysis notebooks
└── ACTION_patients_overlap_only.xlsx  # Clinical dataset

.github/workflows/
├── tests.yml                  # Comprehensive CI/CD pipeline
└── quick-tests.yml           # Fast feedback for development
```

## Testing

The package includes comprehensive test coverage:

```bash
# Run all tests
pytest tests/

# Run specific test categories  
pytest tests/test_core_adaptfx.py          # Core algorithms
pytest tests/test_helper_functions.py      # Mathematical functions
pytest tests/test_error_handling.py        # Edge cases

# Run with coverage reporting
pytest tests/ --cov=adaptive_fractionation_overlap --cov-report=html
```

## Citation

If you use this package in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Yoel Perez-Haas
- **Institution**: University Hospital Zurich
- **Email**: yoel.perezhaas@usz.ch
- **Repository**: https://github.com/YoelPH/Overlap_adaptfx
