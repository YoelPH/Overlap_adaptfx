"""
Simple test to verify our fixtures work correctly
"""

import pytest
import numpy as np


def test_default_parameters_fixture(default_parameters):
    """Test that the default_parameters fixture works correctly."""
    # The fixture should provide a dictionary with all required parameters
    assert isinstance(default_parameters, dict), "Should return a dictionary"
    
    # Check that all expected keys are present
    expected_keys = [
        'min_dose', 'max_dose', 'mean_dose', 'number_of_fractions',
        'steepness_penalty', 'steepness_benefit', 'dose_steps', 'alpha', 'beta'
    ]
    
    for key in expected_keys:
        assert key in default_parameters, f"Missing key: {key}"
    
    # Check some basic values
    assert default_parameters['min_dose'] == 6.0
    assert default_parameters['max_dose'] == 10.0
    assert default_parameters['mean_dose'] == 8.0
    assert default_parameters['number_of_fractions'] == 5
    
    print("✅ default_parameters fixture works correctly!")
    print("Parameters:", default_parameters)


def test_sample_volumes_fixture(sample_volumes):
    """Test that the sample_volumes fixture works correctly."""
    # Should be a numpy array with 6 volumes (planning + 5 fractions)
    assert isinstance(sample_volumes, np.ndarray), "Should be numpy array"
    assert len(sample_volumes) == 6, "Should have 6 volume values (planning + 5 fractions)"
    
    # All volumes should be larger than or equal to 0
    assert np.all(sample_volumes >= 0), "All volumes should be non-negative"

    
    print("✅ sample_volumes fixture works correctly!")
    print("Volumes:", sample_volumes)


def test_multiple_fixtures(default_parameters, sample_volumes):
    """Test using multiple fixtures together."""
    # This tests that pytest can inject multiple fixtures
    assert len(sample_volumes) == default_parameters['number_of_fractions'] + 1
    print("✅ Multiple fixtures work together!")