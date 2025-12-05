"""
Convert Python NumPy results to MATLAB/OCTAVE format.

This script converts results_std_bnr.npz to results_std_bnr.mat
for use with OCTAVE visualization scripts.

Handles all data types robustly including scalars, arrays, and dictionaries.
"""

import numpy as np
import scipy.io as sio
import os
import sys


def convert_npz_to_mat(npz_file='results_std_bnr.npz', 
                       mat_file='results_std_bnr.mat'):
    """
    Convert NumPy NPZ file to MATLAB MAT format.
    
    Parameters
    ----------
    npz_file : str
        Input NPZ filename
    mat_file : str
        Output MAT filename
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if not os.path.exists(npz_file):
        print(f"ERROR: Input file not found: {npz_file}")
        print(f"\nPlease run monte_carlo_validation.py first to generate results.")
        return False
    
    print(f"Converting {npz_file} to {mat_file}...")
    print(f"Loading data...")
    
    try:
        # Load NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        # Convert to dictionary for MAT format
        mat_dict = {}
        skipped = []
        
        for key in data.files:
            value = data[key]
            
            try:
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        # Scalar array - extract the scalar value
                        scalar_val = value.item()
                        mat_dict[key] = scalar_val
                    elif value.dtype == object:
                        # Object array - might contain dict or other complex types
                        # Try to extract if it's a single element
                        if value.size == 1:
                            extracted = value.item()
                            if isinstance(extracted, dict):
                                # Skip dictionaries
                                skipped.append(f"{key} (dict in object array)")
                                continue
                            else:
                                mat_dict[key] = extracted
                        else:
                            # Array of objects - skip
                            skipped.append(f"{key} (object array)")
                            continue
                    else:
                        # Regular numeric array
                        if value.ndim == 1:
                            # Column vector for MATLAB
                            mat_dict[key] = value.reshape(-1, 1)
                        else:
                            # Keep multi-dimensional as is
                            mat_dict[key] = value
                
                # Handle Python numeric types
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    mat_dict[key] = float(value)
                
                # Handle lists and tuples
                elif isinstance(value, (list, tuple)):
                    arr = np.array(value)
                    if arr.dtype == object:
                        # Contains mixed types or complex objects
                        skipped.append(f"{key} (complex list)")
                        continue
                    mat_dict[key] = arr.reshape(-1, 1) if arr.ndim == 1 else arr
                
                # Skip dictionaries
                elif isinstance(value, dict):
                    skipped.append(f"{key} (dict)")
                    continue
                
                # Try to convert other types to string
                else:
                    str_val = str(value)
                    if len(str_val) < 1000:  # Only store reasonably short strings
                        mat_dict[key] = str_val
                    else:
                        skipped.append(f"{key} (too large: {type(value).__name__})")
                        continue
                        
            except Exception as e:
                skipped.append(f"{key} (error: {str(e)[:30]})")
                continue
        
        # Print summary
        print(f"\nData summary:")
        print("-" * 60)
        
        for key, value in sorted(mat_dict.items()):
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    print(f"  {key:20s}: scalar = {value.item()}")
                else:
                    print(f"  {key:20s}: {str(value.shape):15s} ({value.dtype})")
            elif isinstance(value, (int, float)):
                print(f"  {key:20s}: scalar = {value}")
            else:
                val_str = str(value)[:40]
                print(f"  {key:20s}: {type(value).__name__} = '{val_str}'")
        
        print(f"\nTotal fields saved: {len(mat_dict)}")
        
        if skipped:
            print(f"\nSkipped fields ({len(skipped)}):")
            for item in skipped:
                print(f"  - {item}")
        
        # Save as MAT file
        print(f"\nSaving to {mat_file}...")
        sio.savemat(mat_file, mat_dict, oned_as='column')
        
        # Verify the file was created
        if os.path.exists(mat_file):
            file_size = os.path.getsize(mat_file)
            print(f"File saved successfully!")
            print(f"Size: {file_size / 1024:.1f} KB")
        else:
            print("ERROR: File was not created!")
            return False
        
        print(f"\nConversion complete!")
        print(f"\nYou can now run the OCTAVE visualization script:")
        print(f"  octave analyze_results_octave.m")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function."""
    print("=" * 60)
    print("NPZ to MAT Converter for OCTAVE Visualization")
    print("=" * 60)
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
        mat_file = npz_file.replace('.npz', '.mat')
    else:
        npz_file = 'results_std_bnr.npz'
        mat_file = 'results_std_bnr.mat'
    
    # Convert
    success = convert_npz_to_mat(npz_file, mat_file)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
