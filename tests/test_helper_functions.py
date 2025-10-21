"""
Test suite for helper_functions module.

This module tests all the helper functions used in the adaptive fractionation
algorithm. These functions handle core computations like penalty calculations,
state space generation, and probability distributions.

The tests verify:
- Mathematical correctness of penalty calculations
- Proper handling of input/output types and shapes
- Edge cases and boundary conditions
- Consistency with evaluation.ipynb usage patterns
"""

import pytest
import numpy as np
from scipy.stats import norm
from adaptive_fractionation_overlap.helper_functions import (
    data_fit,
    hyperparam_fit,
    std_calc,
    get_state_space,
    probdist,
    penalty_calc_single,
    penalty_calc_matrix,
    max_action,
    actual_policy_plotter,
    analytic_plotting
)
from adaptive_fractionation_overlap.constants import (
    SLOPE, INTERCEPT, DEFAULT_MIN_DOSE, DEFAULT_ALPHA, DEFAULT_BETA
)


class TestPenaltyCalcSingle:
    """Test the penalty_calc_single function."""
    
    def test_penalty_calc_single_basic(self, penalty_calc_test_data):
        """Test basic penalty calculation with known values."""
        result = penalty_calc_single(
            physical_dose=penalty_calc_test_data['physical_dose'],
            min_dose=penalty_calc_test_data['min_dose'],
            actual_volume=penalty_calc_test_data['actual_volume'],
            intercept=penalty_calc_test_data['intercept'],
            slope=penalty_calc_test_data['slope']
        )
        
        # Result should be a scalar number
        assert isinstance(result, (int, float, np.number)), f"Result should be numeric, got {type(result)}"
        
        # Based on the evaluation.ipynb, penalty formula is:
        # penalty = (physical_dose - min_dose) * actual_volume + 
        #           (physical_dose - min_dose)^2 * abs(intercept + slope * actual_volume) / 2
        expected_dose_excess = penalty_calc_test_data['physical_dose'] - penalty_calc_test_data['min_dose']
        expected_steepness = abs(penalty_calc_test_data['intercept'] + penalty_calc_test_data['slope'] * penalty_calc_test_data['actual_volume'])
        expected_penalty = (expected_dose_excess * penalty_calc_test_data['actual_volume'] + 
                          expected_dose_excess**2 * expected_steepness / 2)
        
        assert np.isclose(result, expected_penalty, rtol=1e-10), \
            f"Expected {expected_penalty}, got {result}"
    
    def test_penalty_calc_single_min_dose(self):
        """Test penalty when physical dose equals minimum dose."""
        result = penalty_calc_single(
            physical_dose=6.0,
            min_dose=6.0,
            actual_volume=3.0,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # When physical_dose == min_dose, penalty should be zero
        assert np.isclose(result, 0.0, atol=1e-10), f"Penalty should be 0 when dose equals min_dose, got {result}"
    
    def test_penalty_calc_single_array_volume(self):
        """Test penalty calculation with array of volumes (as used in evaluation.ipynb)."""
        volumes = np.array([2.1, 3.5, 2.8, 3.1, 2.9])
        result = penalty_calc_single(
            physical_dose=8.0,
            min_dose=6.0,
            actual_volume=volumes,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # Result should be an array of same length
        assert isinstance(result, np.ndarray), "Result should be numpy array for array input"
        assert len(result) == len(volumes), f"Result length {len(result)} should match input length {len(volumes)}"
        
        # All penalties should be non-negative (for positive dose excess)
        assert np.all(result >= 0), "All penalties should be non-negative for dose above minimum"
    
    def test_penalty_calc_single_zero_volume(self):
        """Test penalty calculation with zero volume."""
        result = penalty_calc_single(
            physical_dose=8.0,
            min_dose=6.0,
            actual_volume=0.0,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # With zero volume, only intercept term should contribute
        dose_excess = 8.0 - 6.0
        steepness = abs(INTERCEPT + SLOPE * 0.0)
        expected = dose_excess**2 * steepness / 2
        
        assert np.isclose(result, expected, rtol=1e-10), \
            f"Expected {expected}, got {result}"
    
    def test_penalty_calc_single_large_volume(self):
        """Test penalty calculation with large volume."""
        large_volume = 20.0
        result = penalty_calc_single(
            physical_dose=8.0,
            min_dose=6.0,
            actual_volume=large_volume,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # Should return a finite number, not inf or nan
        assert np.isfinite(result), f"Result should be finite for large volume, got {result}"
        assert result > 0, "Penalty should be positive for large volumes"


class TestPenaltyCalcMatrix:
    """Test the penalty_calc_matrix function."""
    
    def test_penalty_calc_matrix_basic(self):
        """Test basic matrix penalty calculation."""
        doses = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        volumes = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = penalty_calc_matrix(
            delivered_doses=doses,
            volume_space=volumes,
            min_dose=6.0,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # Result should be a 2D array
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert result.ndim == 2, "Result should be 2D matrix"
        # Note: actual shape is (volumes, doses) not (doses, volumes)
        assert result.shape == (len(volumes), len(doses)), \
            f"Shape should be ({len(volumes)}, {len(doses)}), got {result.shape}"
    
    def test_penalty_calc_matrix_properties(self):
        """Test mathematical properties of penalty matrix."""
        doses = np.linspace(6.0, 10.0, 5)
        volumes = np.linspace(0.5, 5.0, 4)
        
        result = penalty_calc_matrix(doses, volumes, 6.0, INTERCEPT, SLOPE)
        
        # All penalties should be non-negative (for doses >= min_dose)
        assert np.all(result >= 0), "All penalties should be non-negative"
        
        # Penalties should generally increase with dose (for fixed volume)
        # Note: result shape is (volumes, doses), so we iterate over rows (volumes)
        for i in range(result.shape[0]):
            row = result[i, :]  # Penalties for volume i across all doses
            # Should be non-decreasing (allowing for floating point precision)
            assert np.all(np.diff(row) >= -1e-10), f"Penalties should increase with dose for volume {volumes[i]}"


class TestStdCalc:
    """Test the std_calc function."""
    
    def test_std_calc_basic(self, sample_volumes):
        """Test basic standard deviation calculation."""
        result = std_calc(sample_volumes, DEFAULT_ALPHA, DEFAULT_BETA)
        
        assert isinstance(result, (int, float, np.number)), f"Result should be numeric, got {type(result)}"
        assert result > 0, "Standard deviation should be positive"
        assert np.isfinite(result), "Standard deviation should be finite"
    
    def test_std_calc_constant_volumes(self):
        """Test standard deviation with constant volumes."""
        constant_volumes = np.array([3.0, 3.0, 3.0, 3.0])
        result = std_calc(constant_volumes, DEFAULT_ALPHA, DEFAULT_BETA)
        
        # Should return a small but positive value
        assert result > 0, "Standard deviation should be positive even for constant data"
        assert result < 1.0, "Standard deviation should be small for constant data"
    
    def test_std_calc_varying_volumes(self):
        """Test standard deviation with highly varying volumes."""
        varying_volumes = np.array([0.5, 2.0, 5.0, 1.0, 3.5])
        result = std_calc(varying_volumes, DEFAULT_ALPHA, DEFAULT_BETA)
        
        assert result > 0, "Standard deviation should be positive"
        # Should be larger than for constant volumes
        constant_result = std_calc(np.array([2.5, 2.5, 2.5, 2.5, 2.5]), DEFAULT_ALPHA, DEFAULT_BETA)
        assert result > constant_result, "Varying data should have larger standard deviation"


class TestProbdist:
    """Test the probdist function."""
    
    def test_probdist_basic(self):
        """Test basic probability distribution calculation."""
        X = norm(loc=3.0, scale=1.0)
        state_space = np.linspace(0, 6, 21)
        
        result = probdist(X, state_space)
        
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert len(result) == len(state_space), "Result length should match state space"
        
        # Should be a valid probability distribution
        assert np.all(result >= 0), "All probabilities should be non-negative"
        assert np.isclose(np.sum(result), 1.0, rtol=1e-2), "Probabilities should sum to approximately 1"
    
    def test_probdist_properties(self):
        """Test mathematical properties of probability distribution."""
        X = norm(loc=5.0, scale=1.5)
        state_space = np.linspace(0, 10, 51)
        
        result = probdist(X, state_space)
        
        # Maximum probability should be near the mean
        max_idx = np.argmax(result)
        max_state = state_space[max_idx]
        assert abs(max_state - 5.0) < 1.0, f"Maximum probability at {max_state} should be near mean 5.0"
        
        # Distribution should be unimodal (single peak)
        # Find the peak and check monotonicity on both sides
        left_side = result[:max_idx]
        right_side = result[max_idx:]
        
        # Allow some tolerance for numerical precision
        left_increasing = np.all(np.diff(left_side) >= -1e-10)
        right_decreasing = np.all(np.diff(right_side) <= 1e-10)
        
        assert left_increasing or len(left_side) <= 1, "Left side should be non-decreasing"
        assert right_decreasing or len(right_side) <= 1, "Right side should be non-increasing"


class TestDataFittingFunctions:
    """Test data fitting and hyperparameter functions."""
    
    def test_data_fit_basic(self, sample_volumes):
        """Test basic data fitting."""
        result = data_fit(sample_volumes)
        
        # data_fit returns a scipy distribution object, not parameters
        assert hasattr(result, 'pdf'), "Result should be a distribution with pdf method"
        assert hasattr(result, 'mean'), "Result should be a distribution with mean method"
        
        # Should be able to compute basic statistics
        try:
            mean_val = result.mean()
            assert np.isfinite(mean_val), "Mean should be finite"
        except Exception as e:
            pytest.fail(f"Failed to compute mean: {e}")
    
    def test_hyperparam_fit_basic(self, evaluation_patient_data):
        """Test hyperparameter fitting."""
        # hyperparam_fit expects 2D data (patients × measurements)
        patient_data = np.array(evaluation_patient_data['overlaps'])  # 3 patients × 6 measurements
        
        result = hyperparam_fit(patient_data)
        
        # Should return hyperparameters
        assert isinstance(result, (tuple, list, np.ndarray)), "Result should be a sequence"
        assert len(result) == 2, "Should return exactly 2 hyperparameters (alpha, beta)"
        
        # Hyperparameters should be positive for gamma distribution
        alpha, beta = result
        assert alpha > 0, f"Alpha {alpha} should be positive"
        assert beta > 0, f"Beta {beta} should be positive"


class TestPlottingFunctions:
    """Test plotting helper functions (basic validation only)."""
    
    def test_actual_policy_plotter_no_error(self):
        """Test that actual_policy_plotter doesn't raise errors."""
        policies_overlap = np.array([8.0, 7.5, 8.5, 7.0])
        volume_space = np.array([1.0, 2.0, 3.0, 4.0])
        probabilities = np.array([0.1, 0.3, 0.4, 0.2])
        
        # Should not raise an exception
        try:
            actual_policy_plotter(policies_overlap, volume_space, probabilities)
        except Exception as e:
            pytest.fail(f"actual_policy_plotter raised an exception: {e}")
    
    def test_analytic_plotting_no_error(self):
        """Test that analytic_plotting doesn't raise errors."""
        fraction = 3
        number_of_fractions = 5
        # Values should be a 3D array: (fractions, volume_space, dose_space)
        values = np.random.rand(number_of_fractions, 10, 15)
        volume_space = np.linspace(0, 5, 10)
        dose_space = np.linspace(6, 10, 15)
        
        # Should not raise an exception
        try:
            analytic_plotting(fraction, number_of_fractions, values, volume_space, dose_space)
        except Exception as e:
            pytest.fail(f"analytic_plotting raised an exception: {e}")


# Integration tests using evaluation-style data
@pytest.mark.integration
class TestHelperFunctionsIntegration:
    """Integration tests using patterns from evaluation.ipynb."""
    
    def test_evaluation_penalty_calculation(self, evaluation_patient_data):
        """Test penalty calculation as used in evaluation.ipynb."""
        for patient_overlaps, prescription in zip(
            evaluation_patient_data['overlaps'], 
            evaluation_patient_data['prescriptions']
        ):
            mean_dose = prescription / 5
            
            # Test standard penalty calculation (as in evaluation.ipynb)
            # Using volumes[1:] to skip planning scan
            standard_penalty = penalty_calc_single(
                physical_dose=mean_dose,
                min_dose=6.0,
                actual_volume=np.array(patient_overlaps[1:]),  # Skip planning scan
                intercept=INTERCEPT,
                slope=SLOPE
            )
            
            assert isinstance(standard_penalty, np.ndarray), "Should return array for array input"
            assert len(standard_penalty) == 5, "Should have 5 penalty values for 5 fractions"
            assert np.all(np.isfinite(standard_penalty)), "All penalties should be finite"
    
    def test_std_calc_with_patient_data(self, evaluation_patient_data):
        """Test std_calc with realistic patient data."""
        for patient_overlaps in evaluation_patient_data['overlaps']:
            volumes = np.array(patient_overlaps)
            std_result = std_calc(volumes, DEFAULT_ALPHA, DEFAULT_BETA)
            
            assert std_result > 0, "Standard deviation should be positive"
            assert np.isfinite(std_result), "Standard deviation should be finite"
            assert std_result < 10.0, "Standard deviation should be reasonable for medical data"
    
    def test_state_space_generation_realistic(self, evaluation_patient_data):
        """Test state space generation with realistic patient data."""
        for patient_overlaps in evaluation_patient_data['overlaps']:
            volumes = np.array(patient_overlaps)
            std_val = std_calc(volumes, DEFAULT_ALPHA, DEFAULT_BETA)
            distribution = norm(loc=volumes.mean(), scale=std_val)
            
            state_space = get_state_space(distribution)
            
            assert len(state_space) > 10, "State space should have reasonable size"
            # Note: state space can include negative volumes (this is realistic for uncertainty modeling)
            assert np.max(state_space) > volumes.max(), "State space should extend beyond observed data"
            assert np.max(state_space) - np.min(state_space) > 2.0, "State space should cover reasonable range"


# Edge case and error handling tests
@pytest.mark.unit
class TestHelperFunctionsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_penalty_calc_single_negative_dose(self):
        """Test penalty calculation with negative dose excess."""
        # When physical dose < min dose, penalty calculation behavior
        result = penalty_calc_single(
            physical_dose=5.0,  # Less than min_dose
            min_dose=6.0,
            actual_volume=3.0,
            intercept=INTERCEPT,
            slope=SLOPE
        )
        
        # Should handle negative dose excess gracefully
        assert np.isfinite(result), "Should return finite result for negative dose excess"
    
    def test_std_calc_single_value(self):
        """Test std_calc with single value array."""
        single_volume = np.array([3.0])
        result = std_calc(single_volume, DEFAULT_ALPHA, DEFAULT_BETA)
        
        assert result > 0, "Should return positive std even for single value"
        assert np.isfinite(result), "Should return finite result"
    
    def test_get_state_space_extreme_distribution(self):
        """Test state space with extreme distribution parameters."""
        # Very small scale
        small_scale_dist = norm(loc=2.0, scale=0.01)
        result_small = get_state_space(small_scale_dist)
        
        assert len(result_small) > 0, "Should generate state space for small scale"
        
        # Very large scale
        large_scale_dist = norm(loc=2.0, scale=10.0)
        result_large = get_state_space(large_scale_dist)
        
        assert len(result_large) > 0, "Should generate state space for large scale"
        assert np.max(result_large) - np.min(result_large) > 5.0, "Should cover wide range for large scale"