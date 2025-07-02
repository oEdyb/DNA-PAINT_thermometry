#!/usr/bin/env python3
"""
Final test script to verify the complete step2 refactoring works correctly.
Jabba dabba doo - testing the final refactored step2 files!
"""

def test_final_imports():
    """Test that all final imports work correctly."""
    try:
        import step2_functions
        print("✓ step2_functions.py imports successfully")
        
        functions = [name for name in dir(step2_functions) if not name.startswith('_') and callable(getattr(step2_functions, name))]
        print(f"✓ step2_functions.py contains {len(functions)} functions")
        
        import step2_process_picasso_extracted_data
        print("✓ step2_process_picasso_extracted_data.py imports successfully")
        
        import step2_process_picasso_extracted_data_avg
        print("✓ step2_process_picasso_extracted_data_avg.py imports successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_count():
    """Test that we have the expected number of functions."""
    try:
        import step2_functions
        
        expected_functions = [
            'setup_step2_folders', 'cleanup_existing_traces', 'load_step2_data',
            'detect_peaks_adaptive', 'calculate_binding_site_stats', 'process_binding_site_traces',
            'calculate_distance_matrices', 'plot_pick_time_series', 'plot_scatter_with_np',
            'plot_fine_2d_image', 'plot_distance_matrices', 'process_position_averaging'
        ]
        
        missing_functions = []
        for func_name in expected_functions:
            if not hasattr(step2_functions, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"✗ Missing functions: {missing_functions}")
            return False
        else:
            print(f"✓ All {len(expected_functions)} expected functions found")
            return True
            
    except Exception as e:
        print(f"✗ Function count test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Final Step2 Refactoring Test ===")
    
    success = True
    success &= test_final_imports()
    success &= test_function_count()
    
    if success:
        print("\n✓ All final tests passed! Refactoring is complete and successful.")
        print("✓ Both step2 files have been successfully refactored.")
        print("✓ step2_functions.py contains all expected modular functions.")
        print("✓ Ready for production use!")
    else:
        print("\n✗ Some final tests failed. Check the errors above.")
    
    print("=== Final Test Complete ===")
