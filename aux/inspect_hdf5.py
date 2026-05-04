#!/usr/bin/env python3
"""Inspect HDF5 files from testing_data and compare with current code structure."""
import h5py
import sys

def inspect_hdf5(filepath):
    """Print all groups, datasets, and attributes in an HDF5 file."""
    print(f"\n{'='*60}")
    print(f"File: {filepath.split('/')[-1]}")
    print('='*60)
    
    with h5py.File(filepath, 'r') as f:
        def print_items(name, obj):
            indent = '  ' * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"{indent}GROUP: /{name}")
                for key, val in obj.attrs.items():
                    print(f"{indent}  @{key} = {val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}DATASET: /{name} (shape={obj.shape}, dtype={obj.dtype})")
                for key, val in obj.attrs.items():
                    print(f"{indent}  @{key} = {val}")
        
        f.visititems(print_items)

# Inspect all HDF5 files
import glob
files = sorted(glob.glob('/home/gabriel.amici/testing_data/2026-03-19/*.hdf5'))

if not files:
    print("No HDF5 files found!")
    sys.exit(1)

for fpath in files:
    inspect_hdf5(fpath)

print("\n\nExpected structure from current motormov.py + utils.py:")
print("="*60)
print("Groups: scan-XXXX (one per step)")
print("Group attrs (flattened with '.'):")
print("  step, scan_name, scan_device, scan_motor, state, time")
print("  mirror.tx, mirror.ry, mirror.y1, mirror.y2, mirror.y3")
print("  mirror.cs_rx, mirror.cs_rz, mirror.cs_tx, mirror.cs_ty")
print("  mirror.photocollector")
print("  slit_A1.top, slit_A1.bottom, slit_A1.left, slit_A1.right")
print("  slit_B1.top, slit_B1.bottom, slit_B1.left, slit_B1.right")
print("  Storage ring current, ARIRANHA ID phase, etc.")
print("  Cold Finger temp, etc.")
print("Datasets (with attrs):")
print("  dvf_A1 (image) with @acq_time, @expo_time")
print("  dvf_B1 (image) with @acq_time, @expo_time, @z_pos, @lens_pos, etc.")
