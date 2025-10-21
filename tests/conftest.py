"""
Pytest configuration and shared fixtures

This file defines test fixtures that can be used across all test files.
Fixtures are reusable test data and setup code.
"""

import pytest
import numpy as np
from adaptive_fractionation_overlap.constants import (
    SLOPE, 
    INTERCEPT, 
    DEFAULT_MIN_DOSE, 
    DEFAULT_MAX_DOSE, 
    DEFAULT_MEAN_DOSE,
    DEFAULT_DOSE_STEPS, 
    DEFAULT_NUMBER_OF_FRACTIONS,
    DEFAULT_ALPHA,
    DEFAULT_BETA
)


@pytest.fixture
def default_parameters():
    """
    Fixture providing default parameters for testing.
    
    This creates a dictionary with all the standard parameters
    used throughout the adaptive fractionation algorithm.
    """
    return {
        'min_dose': DEFAULT_MIN_DOSE,
        'max_dose': DEFAULT_MAX_DOSE,
        'mean_dose': DEFAULT_MEAN_DOSE,
        'number_of_fractions': DEFAULT_NUMBER_OF_FRACTIONS,
        'steepness_penalty': -SLOPE,  # Convert to positive (functions expect positive)
        'steepness_benefit': -INTERCEPT,  # Use INTERCEPT or default
        'dose_steps': DEFAULT_DOSE_STEPS,
        'alpha': DEFAULT_ALPHA,
        'beta': DEFAULT_BETA
    }


@pytest.fixture
def sample_volumes():
    """
    Fixture providing realistic sample volume data.
    
    This represents a typical 5-fraction treatment with varying
    PTV-OAR overlap volumes.
    """
    return np.array([0.41, 2.37, 0.68, 2.67, 1.62, 1.27])


@pytest.fixture
def sample_volumes_list():
    """
    Fixture providing sample volume data as a list (for adaptfx_full).
    
    Returns the same data as sample_volumes but as a list, which is
    the expected input format for adaptfx_full function.
    """
    return [0.41, 2.37, 0.68, 2.67, 1.62, 1.27]


@pytest.fixture
def evaluation_patient_data():
    """
    Fixture providing realistic patient data similar to evaluation.ipynb.
    
    Returns:
        dict: Contains patient overlaps, prescriptions, and derived values
    """
    # Sample patient overlap data (6 values: planning scan + 5 treatment fractions)
    patient_overlaps = [
        [0.0, 2.1, 3.5, 2.8, 3.1, 2.9],  # Patient 1
        [0.0, 4.2, 3.8, 4.1, 3.9, 4.3],  # Patient 2  
        [0.0, 1.5, 2.1, 1.8, 2.3, 1.9],  # Patient 3
    ]
    
    # Corresponding prescription doses (total dose for all 5 fractions)
    prescription_doses = [40.0, 42.5, 37.5]
    
    return {
        'overlaps': patient_overlaps,
        'prescriptions': prescription_doses,
        'mean_doses': [dose/5 for dose in prescription_doses]  # Mean dose per fraction
    }


@pytest.fixture
def penalty_calc_test_data():
    """
    Fixture providing test data for penalty calculation functions.
    
    Returns:
        dict: Standard parameters for penalty calculation testing
    """
    return {
        'physical_dose': 8.0,
        'min_dose': 6.0,
        'actual_volume': 3.0,
        'intercept': INTERCEPT,
        'slope': SLOPE
    }


@pytest.fixture
def high_overlap_volumes():
    """Fixture for high overlap scenario (challenging case)."""
    return np.array([9.08, 19.79, 6.02, 9.45, 19.59, 12.62])


@pytest.fixture
def low_overlap_volumes():
    """Fixture for low overlap scenario (favorable case).""" 
    return np.array([1.74, 2.17, 1.18, 1.43, 3.08, 2.26])

@pytest.fixture
def decreasing_overlap_volumes():
    """Fixture for improving scenario (overlap decreases over time)."""
    return np.array([6, 5, 4, 3, 2, 1])


@pytest.fixture
def increasing_overlap_volumes():
    """Fixture for worsening scenario (overlap increases over time)."""
    return np.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def dose_range():
    """Fixture providing a range of test doses."""
    return np.arange(6.0, 10.5, 0.5)  # [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]


@pytest.fixture
def volume_range():
    """Fixture providing a range of overlap volumes."""
    return np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]


@pytest.fixture
def steepness_range():
    """Fixture providing a range of steepness parameters."""
    return [0.1, 0.25, 0.5, 0.75, 1.0]


# Configuration hooks
def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    This function runs when pytest starts and sets up
    custom test markers for organizing tests.
    """
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, multiple components)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (performance or stress tests)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    This function runs after pytest collects all tests and can
    automatically add markers based on test names or locations.
    """
    # Automatically mark slow tests
    for item in items:
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
            
        # Mark integration tests
        if "integration" in item.name or "full" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
            
        # Mark unit tests (default for most tests)
        elif not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Utility functions for tests
def assert_dose_bounds(doses, min_dose=DEFAULT_MIN_DOSE, max_dose=DEFAULT_MAX_DOSE):
    """
    Utility function to check dose bounds.
    
    This is a helper function that tests can use to verify
    that doses are within acceptable ranges.
    """
    doses = np.asarray(doses)
    assert np.all(doses >= min_dose), f"Some doses below minimum {min_dose}: {doses[doses < min_dose]}"
    assert np.all(doses <= max_dose), f"Some doses above maximum {max_dose}: {doses[doses > max_dose]}"


def assert_probability_distribution(probabilities, axis=None, tolerance=1e-10):
    """
    Utility function to check that arrays represent valid probability distributions.
    
    Probabilities should sum to 1 and be non-negative.
    """
    probabilities = np.asarray(probabilities)
    
    # Check non-negative
    assert np.all(probabilities >= 0), "Probabilities must be non-negative"
    
    # Check sum to 1
    if axis is None:
        prob_sum = np.sum(probabilities)
        assert abs(prob_sum - 1.0) < tolerance, f"Probabilities should sum to 1, got {prob_sum}"
    else:
        prob_sums = np.sum(probabilities, axis=axis)
        assert np.allclose(prob_sums, 1.0, atol=tolerance), f"Probability rows should sum to 1, got {prob_sums}"


def assert_increasing(array, strict=False):
    """
    Utility function to check that an array is increasing.
    
    Args:
        array: Array to check
        strict: If True, requires strictly increasing (no equal values)
    """
    array = np.asarray(array)
    if strict:
        assert np.all(array[1:] > array[:-1]), "Array should be strictly increasing"
    else:
        assert np.all(array[1:] >= array[:-1]), "Array should be non-decreasing"


# Custom assertions for adaptive fractionation
def assert_valid_dose_plan(physical_doses, accumulated_doses, target_dose=None, tolerance=2.0):
    """
    Comprehensive validation for a dose plan.
    
    This checks all the standard requirements for a valid
    adaptive fractionation dose plan.
    """
    physical_doses = np.asarray(physical_doses)
    accumulated_doses = np.asarray(accumulated_doses)
    
    # Check array lengths match
    assert len(physical_doses) == len(accumulated_doses), \
        "Physical and accumulated dose arrays should have same length"
    
    # Check dose bounds
    assert_dose_bounds(physical_doses)
    
    # Check accumulated doses are increasing
    assert_increasing(accumulated_doses)
    
    # Check accumulated doses are cumulative sums
    expected_accumulated = np.cumsum(physical_doses)
    assert np.allclose(accumulated_doses, expected_accumulated, atol=1e-10), \
        "Accumulated doses should be cumulative sum of physical doses"
    
    # Check final dose is reasonable
    if target_dose is not None:
        final_dose = accumulated_doses[-1]
        assert abs(final_dose - target_dose) < tolerance, \
            f"Final dose {final_dose:.1f} should be within {tolerance} Gy of target {target_dose}"


def assert_algorithm_output(result, expected_length=9):
    """
    Validate the output format of adaptive_fractionation_core.
    
    The core algorithm returns a specific tuple of 9 elements.
    """
    assert isinstance(result, (list, tuple)), "Result should be list or tuple"
    assert len(result) == expected_length, f"Result should have {expected_length} elements"
    
    # Unpack and check types
    [policies, policies_overlap, volume_space, physical_dose, 
     penalty_added, values, dose_space, probabilities, final_penalty] = result
    
    assert isinstance(policies, np.ndarray), "Policies should be numpy array"
    assert isinstance(policies_overlap, np.ndarray), "Policies overlap should be numpy array" 
    assert isinstance(volume_space, np.ndarray), "Volume space should be numpy array"
    assert isinstance(physical_dose, (float, np.floating)), "Physical dose should be scalar"
    assert isinstance(penalty_added, (float, np.floating)), "Penalty should be scalar"
    assert isinstance(values, np.ndarray), "Values should be numpy array"
    assert isinstance(dose_space, np.ndarray), "Dose space should be numpy array"
    assert isinstance(probabilities, np.ndarray), "Probabilities should be numpy array"
    assert isinstance(final_penalty, (float, np.floating)), "Final penalty should be scalar"


# Test data generators
def generate_random_volumes(n_fractions=5, min_vol=0.0, max_vol=10.0, seed=42):
    """Generate random but reproducible volume data for testing."""
    np.random.seed(seed)
    return np.random.uniform(min_vol, max_vol, n_fractions + 1)