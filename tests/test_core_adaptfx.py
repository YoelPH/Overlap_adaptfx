"""
Test suite for core_adaptfx module.

This module tests the core adaptive fractionation functions including:
- adaptive_fractionation_core: The main algorithm for single fraction optimization
- adaptfx_full: Complete treatment planning for all fractions
- precompute_plan: Plan precomputation functionality

These tests verify the main algorithmic logic and ensure the functions work
correctly with realistic patient data as shown in evaluation.ipynb.
"""

import pytest
import numpy as np
from adaptive_fractionation_overlap.core_adaptfx import (
    adaptive_fractionation_core,
    adaptfx_full,
    precompute_plan
)
from adaptive_fractionation_overlap.constants import (
    DEFAULT_MIN_DOSE, 
    DEFAULT_MAX_DOSE, 
    DEFAULT_MEAN_DOSE,
    DEFAULT_DOSE_STEPS, 
    DEFAULT_NUMBER_OF_FRACTIONS,
    DEFAULT_ALPHA,
    DEFAULT_BETA
)


class TestAdaptiveFractionationCore:
    """Test the adaptive_fractionation_core function."""
    
    def test_adaptive_fractionation_core_basic(self, sample_volumes):
        """Test basic functionality of adaptive_fractionation_core."""
        # Test with first 2 volumes (planning + first fraction)
        volumes = sample_volumes[:2]
        fraction = 1
        accumulated_dose = 0.0
        
        result = adaptive_fractionation_core(
            fraction=fraction,
            volumes=volumes,
            accumulated_dose=accumulated_dose,
            number_of_fractions=DEFAULT_NUMBER_OF_FRACTIONS,
            min_dose=DEFAULT_MIN_DOSE,
            max_dose=DEFAULT_MAX_DOSE,
            mean_dose=DEFAULT_MEAN_DOSE
        )
        
        # Verify result structure (should return 9 elements)
        assert isinstance(result, (list, tuple)), "Result should be list or tuple"
        assert len(result) == 9, f"Result should have 9 elements, got {len(result)}"
        
        # Unpack and verify types
        [policies, policies_overlap, volume_space, physical_dose, 
         penalty_added, values, dose_space, probabilities, final_penalty] = result
        
        assert isinstance(policies, np.ndarray), "Policies should be numpy array"
        assert isinstance(volume_space, np.ndarray), "Volume space should be numpy array"
        assert isinstance(physical_dose, (int, float, np.number)), "Physical dose should be scalar"
        assert isinstance(values, np.ndarray), "Values should be numpy array"
        assert isinstance(dose_space, np.ndarray), "Dose space should be numpy array"
        assert isinstance(probabilities, np.ndarray), "Probabilities should be numpy array"
        
        # Verify dose is within bounds
        assert 6.0 <= physical_dose <= 10.0, f"Physical dose {physical_dose} should be within bounds [6.0, 10.0]"
    
    def test_adaptive_fractionation_core_fraction_progression(self, sample_volumes):
        """Test adaptive_fractionation_core for different fractions."""
        accumulated_dose = 0.0
        
        for fraction in range(1, 6):  # Test fractions 1-5
            volumes_up_to_fraction = sample_volumes[:fraction+1]  # Include planning + fractions up to current
            
            result = adaptive_fractionation_core(
                fraction=fraction,
                volumes=volumes_up_to_fraction,
                accumulated_dose=accumulated_dose,
                number_of_fractions=5,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=8.0
            )
            
            # Extract physical dose
            physical_dose = result[3]
            
            # Update accumulated dose for next iteration
            accumulated_dose += physical_dose
            
            # Verify dose bounds
            assert 6.0 <= physical_dose <= 10.0, \
                f"Fraction {fraction}: dose {physical_dose} should be within bounds"
            
            # Verify accumulated dose is reasonable
            assert accumulated_dose <= 40.0, \
                f"Fraction {fraction}: accumulated dose {accumulated_dose} should be reasonable"
    
    def test_adaptive_fractionation_core_last_fraction(self, sample_volumes):
        """Test adaptive_fractionation_core for the last fraction."""
        # Test last fraction (should use all volumes)
        accumulated_dose = 32.0  # Assuming 4 fractions @ 8 Gy each
        
        result = adaptive_fractionation_core(
            fraction=5,
            volumes=sample_volumes,  # All volumes
            accumulated_dose=accumulated_dose,
            number_of_fractions=5,
            min_dose=6,
            max_dose=10,
            mean_dose=8
        )
        
        physical_dose = result[3]
        
        # Should be within bounds
        assert 6.0 <= physical_dose <= 10.0, \
            f"Last fraction dose {physical_dose} should be within bounds"
        
        # Final accumulated dose should be reasonable
        final_dose = accumulated_dose + physical_dose
        assert final_dose == 40.0, \
            f"Final total dose {final_dose} should be reasonable"
    
    def test_adaptive_fractionation_core_parameters(self, sample_volumes):
        """Test adaptive_fractionation_core with different parameters."""
        volumes = sample_volumes[:2]
        
        # Test with different min/max doses
        result_tight = adaptive_fractionation_core(
            fraction=1,
            volumes=volumes,
            accumulated_dose=0.0,
            min_dose=7.0,
            max_dose=9.0,
            mean_dose=8.0
        )
        
        result_loose = adaptive_fractionation_core(
            fraction=1,
            volumes=volumes,
            accumulated_dose=0.0,
            min_dose=5.0,
            max_dose=11.0,
            mean_dose=8.0
        )
        
        dose_tight = result_tight[3]
        dose_loose = result_loose[3]
        
        assert 7.0 <= dose_tight <= 9.0, "Tight bounds should be respected"
        assert 5.0 <= dose_loose <= 11.0, "Loose bounds should be respected"
    
    def test_adaptive_fractionation_core_dose_space(self, sample_volumes):
        """Test that dose space is generated correctly."""
        volumes = sample_volumes[:2]
        
        result = adaptive_fractionation_core(
            fraction=1,
            volumes=volumes,
            accumulated_dose=0.0,
            dose_steps=0.5  # Different step size
        )
        
        dose_space = result[6]
        
        # Should be evenly spaced (allowing for floating point precision)
        assert len(dose_space) > 1, "Dose space should have multiple values"
        
        if len(dose_space) > 1:
            steps = np.diff(dose_space)
            # Check that most steps are close to expected step size (allowing for boundary effects)
            expected_step = 0.5
            most_steps = steps[:-1]  # Exclude last step which might be different due to boundary
            assert np.allclose(most_steps, expected_step, rtol=1e-6), \
                "Most dose space steps should match expected step size"
            assert np.isclose(np.mean(steps), expected_step, rtol=0.1), \
                "Average step size should be close to expected"


class TestAdaptfxFull:
    """Test the adaptfx_full function."""
    
    def test_adaptfx_full_basic(self, sample_volumes_list):
        """Test basic functionality of adaptfx_full."""
        # adaptfx_full expects a list input (as used in evaluation.ipynb)
        result = adaptfx_full(
            volumes=sample_volumes_list,
            number_of_fractions=5,
            min_dose=6.0,
            max_dose=10.0,
            mean_dose=8.0
        )
        
        # Should return 3 arrays: physical_doses, accumulated_doses, total_penalty
        assert isinstance(result, (tuple, list)), "Result should be tuple or list"
        assert len(result) == 3, f"Result should have 3 elements, got {len(result)}"
        
        physical_doses, accumulated_doses, total_penalty = result
        
        # Verify types and shapes
        assert isinstance(physical_doses, np.ndarray), "Physical doses should be numpy array"
        assert isinstance(accumulated_doses, np.ndarray), "Accumulated doses should be numpy array"
        assert isinstance(total_penalty, (int, float, np.number)), "Total penalty should be scalar"
        
        assert len(physical_doses) == 5, f"Should have 5 physical doses, got {len(physical_doses)}"
        assert len(accumulated_doses) == 5, f"Should have 5 accumulated doses, got {len(accumulated_doses)}"
        
        # Verify dose bounds
        assert np.all(physical_doses >= 6.0), "All physical doses should be >= min_dose"
        assert np.all(physical_doses <= 10.0), "All physical doses should be <= max_dose"
        
        # Verify accumulated doses are increasing (excluding first zero)
        assert np.all(np.diff(accumulated_doses[1:]) > 0), "Accumulated doses should be strictly increasing"
        
        # Verify accumulated doses structure: starts with 0, then cumulative sums
        # accumulated_doses[0] = 0, accumulated_doses[i+1] = accumulated_doses[i] + physical_doses[i]
        assert accumulated_doses[0] == 0.0, "First accumulated dose should be 0.0"
        expected_accumulated = np.concatenate([[0.0], np.cumsum(physical_doses[:-1])])
        assert np.allclose(accumulated_doses, expected_accumulated, atol=1e-10), \
            "Accumulated doses should follow algorithm pattern: [0, cumsum(doses[:-1])]"
    
    def test_adaptfx_full_evaluation_style(self, evaluation_patient_data):

        for i, (patient_overlaps, prescription) in enumerate(
            zip(evaluation_patient_data['overlaps'], evaluation_patient_data['prescriptions'])
        ):
            mean_dose = prescription / 5
            
            physical_doses, accumulated_doses, total_penalty = adaptfx_full(
                volumes=patient_overlaps,  # List of 6 values (planning + 5 fractions)
                number_of_fractions=5,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=mean_dose
            )
            
            # Verify results
            assert len(physical_doses) == 5, f"Patient {i}: Should have 5 doses"
            assert len(accumulated_doses) == 5, f"Patient {i}: Should have 5 accumulated doses"
            
            # Check final dose is close to prescription
            total_delivered_dose = accumulated_doses[-1] + physical_doses[-1]
            assert abs(total_delivered_dose - prescription) < 0.0001, \
                f"Patient {i}: Total delivered dose {total_delivered_dose} should be close to prescription {prescription}"
            
            # Verify penalty is finite
            assert np.isfinite(total_penalty), f"Patient {i}: Total penalty should be finite"
    
    def test_adaptfx_full_parameter_variations(self, sample_volumes_list):
        """Test adaptfx_full with different parameter sets."""
        base_params = {
            'volumes': sample_volumes_list,
            'number_of_fractions': 5,
            'min_dose': 6.0,
            'max_dose': 10.0,
            'mean_dose': 8.0
        }
        
        # Test with different dose steps
        result_coarse = adaptfx_full(**base_params, dose_steps=0.5)
        result_fine = adaptfx_full(**base_params, dose_steps=0.25)
        
        # Both should produce valid results
        for result in [result_coarse, result_fine]:
            physical_doses, accumulated_doses, total_penalty = result
            assert len(physical_doses) == 5, "Should have 5 doses"
            assert np.all(6.0 <= physical_doses) and np.all(physical_doses <= 10.0), "Doses should be in bounds"
        
        # Fine steps might give different (potentially better) results
        # But both should be reasonable
        assert np.isfinite(result_coarse[2]), "Coarse result penalty should be finite"
        assert np.isfinite(result_fine[2]), "Fine result penalty should be finite"
    
    def test_adaptfx_full_dose_constraints(self, sample_volumes_list):
        """Test that adaptfx_full respects dose constraints properly."""
        # Test with tight constraints
        tight_result = adaptfx_full(
            volumes=sample_volumes_list,
            min_dose=7.5,
            max_dose=8.5,
            mean_dose=8.0
        )
        
        physical_doses = tight_result[0]
        
        # All doses should respect tight bounds
        assert np.all(physical_doses >= 7.5), "All doses should be >= 7.5"
        assert np.all(physical_doses <= 8.5), "All doses should be <= 8.5"
        
        # Final dose should be close to target
        total_delivered_dose = tight_result[1][-1] + tight_result[0][-1]  # accumulated + last physical dose
        target_dose = 8.0 * 5  # 40 Gy total
        assert abs(total_delivered_dose - target_dose) < 0.0001, \
            f"Total delivered dose {total_delivered_dose} should be close to target {target_dose}"
    
    def test_adaptfx_full_different_fractions(self, sample_volumes_list):
        """Test adaptfx_full with different numbers of fractions."""
        # Adapt sample data for different fraction numbers
        base_volumes = sample_volumes_list[:4]  # Use first 4 values
        
        # Test 3 fractions
        result_3 = adaptfx_full(
            volumes=base_volumes,  # 4 values: planning + 3 fractions
            number_of_fractions=3,
            mean_dose=8.0
        )
        
        physical_doses_3 = result_3[0]
        assert len(physical_doses_3) == 3, "Should have 3 doses for 3 fractions"
        
        # Final dose should be around 24 Gy (3 * 8)
        total_delivered_dose_3 = result_3[1][-1] + result_3[0][-1]  # accumulated + last physical dose
        assert abs(total_delivered_dose_3 - 8*3) < 0.0001, f"3-fraction total {total_delivered_dose_3} should be close to traget"


class TestPrecomputePlan:
    """Test the precompute_plan function."""
    
    def test_precompute_plan_basic(self, sample_volumes):
        """Test basic functionality of precompute_plan."""
        volumes = sample_volumes[:3]  # First 3 volumes
        
        result = precompute_plan(
            fraction=2,
            volumes=volumes,
            accumulated_dose=6,
            number_of_fractions=5
        )
        
        # Should return DataFrame and two lists
        assert isinstance(result, (tuple, list)), "Result should be tuple or list"
        assert len(result) == 3, f"Result should have 3 elements, got {len(result)}"
        
        df, volume_list, dose_list = result
        
        # Verify DataFrame
        assert hasattr(df, 'columns'), "First element should be DataFrame-like"
        assert len(df) > 0, "DataFrame should not be empty"
        
        # Verify lists
        assert isinstance(volume_list, (list, np.ndarray)), "Volume list should be list or array"
        assert isinstance(dose_list, (list, np.ndarray)), "Dose list should be list or array"
        assert len(volume_list) == len(dose_list), "Volume and dose lists should have same length"
    
    def test_precompute_plan_different_fractions(self, sample_volumes):
        """Test precompute_plan for different fractions."""
        for fraction in [1, 2, 3, 4, 5]:
            volumes = sample_volumes[:fraction+1]
            accumulated_dose = (fraction - 1) * 8.0  # Assume 8 Gy per previous fraction
            
            result = precompute_plan(
                fraction=fraction,
                volumes=volumes,
                accumulated_dose=accumulated_dose
            )
            
            df, volume_list, dose_list = result
            
            # All results should be valid
            assert len(df) > 0, f"Fraction {fraction}: DataFrame should not be empty"
            assert len(volume_list) > 0, f"Fraction {fraction}: Volume list should not be empty"
            assert len(dose_list) > 0, f"Fraction {fraction}: Dose list should not be empty"
            
            # Doses should be within bounds
            if isinstance(dose_list, (list, np.ndarray)) and len(dose_list) > 0:
                doses = np.array(dose_list)
                assert np.all(doses >= 6.0), f"Fraction {fraction}: All doses should be >= 6.0"
                assert np.all(doses <= 10.0), f"Fraction {fraction}: All doses should be <= 10.0"


# Integration tests combining multiple functions
@pytest.mark.integration
class TestCoreAdaptfxIntegration:
    """Integration tests for core adaptive fractionation functions."""
    
    def test_adaptfx_full_vs_core_consistency(self, sample_volumes_list):
        """Test that adaptfx_full gives consistent results with adaptive_fractionation_core."""
        # Run adaptfx_full
        full_result = adaptfx_full(
            volumes=sample_volumes_list,
            number_of_fractions=5,
            min_dose=6.0,
            max_dose=10.0,
            mean_dose=8.0
        )
        
        physical_doses_full, accumulated_doses_full, total_penalty_full = full_result
        
        # Now manually run adaptive_fractionation_core for each fraction
        physical_doses_manual = []
        accumulated_dose = 0.0
        sample_volumes_array = np.array(sample_volumes_list)
        
        for fraction in range(1, 6):
            if fraction < 5:
                # For fractions 1-4, use volumes up to current fraction
                volumes_subset = sample_volumes_array[:-5+fraction]
            else:
                # For last fraction, use all volumes
                volumes_subset = sample_volumes_array
            
            core_result = adaptive_fractionation_core(
                fraction=fraction,
                volumes=volumes_subset,
                accumulated_dose=accumulated_dose,
                number_of_fractions=5,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=8.0
            )
            
            physical_dose = core_result[3]
            physical_doses_manual.append(physical_dose)
            accumulated_dose += physical_dose
        
        # Results should be very similar (allowing for numerical differences)
        physical_doses_manual = np.array(physical_doses_manual)
        
        assert np.allclose(physical_doses_full, physical_doses_manual, rtol=1e-6), \
            "adaptfx_full and manual core results should be consistent"
    
    def test_evaluation_workflow_reproduction(self, evaluation_patient_data):
        """Test reproducing the evaluation.ipynb workflow."""
        for i, (patient_overlaps, prescription) in enumerate(
            zip(evaluation_patient_data['overlaps'], evaluation_patient_data['prescriptions'])
        ):
            mean_dose = prescription / 5
            
            # Run adaptfx_full (as in evaluation.ipynb)
            physical_doses, accumulated_doses, total_penalty = adaptfx_full(
                volumes=patient_overlaps,
                number_of_fractions=5,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=mean_dose
            )
            
            # Verify results match expected properties
            assert len(physical_doses) == 5, f"Patient {i}: Should have 5 physical doses"
            
            # Check dose bounds
            assert np.all(6.0 <= physical_doses), f"Patient {i}: All doses should be >= 6.0"
            assert np.all(physical_doses <= 10.0), f"Patient {i}: All doses should be <= 10.0"
            
            # Check final dose is reasonable
            total_delivered_dose = accumulated_doses[-1] + physical_doses[-1]
            assert abs(total_delivered_dose - prescription) < 3.0, \
                f"Patient {i}: Total delivered dose {total_delivered_dose:.1f} should be close to prescription {prescription}"
            
            # Check penalty is finite
            assert np.isfinite(total_penalty), f"Patient {i}: Total penalty should be finite"
            
            # Check that doses adapt to volume changes
            # Higher volumes should tend to get lower doses (but not always due to constraints)
            volumes_treatment = patient_overlaps[1:]  # Skip planning scan
            correlation = np.corrcoef(volumes_treatment, physical_doses)[0, 1]
            
            # Correlation should be negative (higher volume -> lower dose) or close to zero
            # We're lenient here because constraints can override this pattern
            assert correlation <= 0.5, \
                f"Patient {i}: Doses should not be strongly positively correlated with volumes"


# Performance and edge case tests
@pytest.mark.unit
class TestCoreAdaptfxEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_adaptfx_full_minimum_volumes(self):
        """Test adaptfx_full with minimum required volumes."""
        # Test with exactly the required number of volumes (6 for 5 fractions)
        minimal_volumes = [0.0, 1.0, 2.0, 1.5, 2.5, 1.8]
        
        result = adaptfx_full(
            volumes=minimal_volumes,
            number_of_fractions=5
        )
        
        physical_doses, accumulated_doses, total_penalty = result
        
        assert len(physical_doses) == 5, "Should work with minimal volume data"
        assert np.all(np.isfinite(physical_doses)), "All doses should be finite"
    
    def test_adaptive_fractionation_core_high_accumulated_dose(self, sample_volumes):
        """Test adaptive_fractionation_core when accumulated dose is high."""
        volumes = sample_volumes[:2]
        high_accumulated_dose = 35.0  # Very high for early fraction
        
        result = adaptive_fractionation_core(
            fraction=2,
            volumes=volumes,
            min_dose = 6.0,
            accumulated_dose=high_accumulated_dose,
            number_of_fractions=5,
            mean_dose=8.0
        )
        
        physical_dose = result[3]
        
        # Should probably choose minimum dose when accumulated dose is high
        assert physical_dose == 6.0, "Dose should be minimal dose"

    def test_adaptive_fractionation_core_low_accumulated_dose(self, sample_volumes):
        """Test adaptive_fractionation_core when accumulated dose is low."""
        volumes = sample_volumes[:5]
        low_accumulated_dose = 6.0  # Very low for early fraction

        result = adaptive_fractionation_core(
            fraction=4,
            volumes=volumes,
            min_dose = 6.0,
            max_dose = 10.0,
            accumulated_dose=low_accumulated_dose,
            number_of_fractions=5,
            mean_dose=8.0
        )
        
        physical_dose = result[3]
        
        # Should probably choose minimum dose when accumulated dose is high
        assert physical_dose == 10.0, "Dose should be maximum dose" 


@pytest.mark.slow
class TestCoreAdaptfxPerformance:
    """Performance tests for core functions."""
    
    def test_adaptfx_full_multiple_patients(self, evaluation_patient_data):
        """Test adaptfx_full performance with multiple patients."""
        import time
        
        start_time = time.time()
        
        # Run adaptfx_full for all patients
        for patient_overlaps, prescription in zip(
            evaluation_patient_data['overlaps'], 
            evaluation_patient_data['prescriptions']
        ):
            mean_dose = prescription / 5
            
            physical_doses, accumulated_doses, total_penalty = adaptfx_full(
                volumes=patient_overlaps,
                number_of_fractions=5,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=mean_dose
            )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete within reasonable time
        assert elapsed < 10.0, f"Should complete 3 patients in <10 seconds, took {elapsed:.2f}s"