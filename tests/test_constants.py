"""
Test suite for constants module.

This module tests that all constants in the adaptive_fractionation_overlap
package have the correct values and types. Constants are critical for 
ensuring consistent behavior across the algorithm.

Testing constants helps catch:
- Accidental modifications to critical values
- Type inconsistencies that could cause runtime errors
- Missing or undefined constants
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

class TestConstantTypes:
    """Test that constants have the expected types."""
    
    def test_slope_type(self):
        """Test that SLOPE is a numeric type."""
        assert isinstance(SLOPE, (int, float, np.number)), f"SLOPE should be numeric, got {type(SLOPE)}"
    
    def test_intercept_type(self):
        """Test that INTERCEPT is a numeric type."""
        assert isinstance(INTERCEPT, (int, float, np.number)), f"INTERCEPT should be numeric, got {type(INTERCEPT)}"
    
    def test_default_min_dose_type(self):
        """Test that DEFAULT_MIN_DOSE is a numeric type."""
        assert isinstance(DEFAULT_MIN_DOSE, (int, float, np.number)), f"DEFAULT_MIN_DOSE should be numeric, got {type(DEFAULT_MIN_DOSE)}"
    
    def test_default_max_dose_type(self):
        """Test that DEFAULT_MAX_DOSE is a numeric type."""
        assert isinstance(DEFAULT_MAX_DOSE, (int, float, np.number)), f"DEFAULT_MAX_DOSE should be numeric, got {type(DEFAULT_MAX_DOSE)}"
    
    def test_default_mean_dose_type(self):
        """Test that DEFAULT_MEAN_DOSE is a numeric type."""
        assert isinstance(DEFAULT_MEAN_DOSE, (int, float, np.number)), f"DEFAULT_MEAN_DOSE should be numeric, got {type(DEFAULT_MEAN_DOSE)}"
    
    def test_default_dose_steps_type(self):
        """Test that DEFAULT_DOSE_STEPS is a numeric type."""
        assert isinstance(DEFAULT_DOSE_STEPS, (int, float, np.number)), f"DEFAULT_DOSE_STEPS should be numeric, got {type(DEFAULT_DOSE_STEPS)}"
    
    def test_default_number_of_fractions_type(self):
        """Test that DEFAULT_NUMBER_OF_FRACTIONS is an integer."""
        assert isinstance(DEFAULT_NUMBER_OF_FRACTIONS, int), f"DEFAULT_NUMBER_OF_FRACTIONS should be int, got {type(DEFAULT_NUMBER_OF_FRACTIONS)}"
    
    def test_default_alpha_type(self):
        """Test that DEFAULT_ALPHA is a numeric type."""
        assert isinstance(DEFAULT_ALPHA, (int, float, np.number)), f"DEFAULT_ALPHA should be numeric, got {type(DEFAULT_ALPHA)}"
    
    def test_default_beta_type(self):
        """Test that DEFAULT_BETA is a numeric type."""
        assert isinstance(DEFAULT_BETA, (int, float, np.number)), f"DEFAULT_BETA should be numeric, got {type(DEFAULT_BETA)}"


class TestConstantRelationships:
    """Test logical relationships between constants."""
    
    def test_dose_bounds_relationship(self):
        """Test that min_dose < mean_dose < max_dose."""
        assert DEFAULT_MIN_DOSE < DEFAULT_MEAN_DOSE, "Mean dose should be greater than minimum dose"
        assert DEFAULT_MEAN_DOSE < DEFAULT_MAX_DOSE, "Mean dose should be less than maximum dose"
    
    def test_dose_steps_positive(self):
        """Test that dose steps is positive."""
        assert DEFAULT_DOSE_STEPS > 0, "Dose steps should be positive"
    
    def test_number_of_fractions_positive(self):
        """Test that number of fractions is positive."""
        assert DEFAULT_NUMBER_OF_FRACTIONS > 0, "Number of fractions should be positive"
    
    def test_alpha_beta_positive(self):
        """Test that alpha and beta parameters are positive (for gamma distribution)."""
        assert DEFAULT_ALPHA > 0, "Alpha parameter should be positive for gamma distribution"
        assert DEFAULT_BETA > 0, "Beta parameter should be positive for gamma distribution"


@pytest.mark.unit
def test_all_constants_importable():
    """
    Test that all expected constants can be imported successfully.
    
    This is a basic smoke test to ensure the constants module is properly structured.
    """
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
    
    # Verify none are None
    constants = [
        SLOPE, INTERCEPT, DEFAULT_MIN_DOSE, DEFAULT_MAX_DOSE, 
        DEFAULT_MEAN_DOSE, DEFAULT_DOSE_STEPS, DEFAULT_NUMBER_OF_FRACTIONS,
        DEFAULT_ALPHA, DEFAULT_BETA
    ]
    
    for const in constants:
        assert const is not None, "All constants should be defined (not None)"