#!/usr/bin/env python
"""Convert old HDF5 files to new format matching current motormov.py output."""
import h5py
import os
import glob

# Mapping from old 'scan_type' to new (scan_name, scan_device) based on motormov.py
SCAN_TYPE_MAP = {
    'mirror': ('mirror', 'mirror'),       # CAXMirrorMove: scan_name='mirror', scan_device='mirror'
    'slit_A1': ('slit', 'slit_A1'),       # CAXSlitMove: scan_name='slit', scan_device=slit_name
    'slit_B1': ('slit', 'slit_B1'),
    'caustic': ('caustic', 'dvf_B1'),     # CAXCausticMove: scan_name='caustic', scan_device='dvf_B1'
    'lens': ('lens', 'dvf_B1'),           # CAXLensMove: scan_name='lens', scan_device='dvf_B1'
}

def convert_file(old_path, output_dir):
    """Convert a single old HDF5 file to new format."""
    old_filename = os.path.basename(old_path)
    name, ext = os.path.splitext(old_filename)
    new_filename = f"{name}_converted{ext}"
    new_path = os.path.join(output_dir, new_filename)

    with h5py.File(old_path, 'r') as old_f, h5py.File(new_path, 'w') as new_f:
        # Copy file-level attributes (e.g., 'start time')
        for key, val in old_f.attrs.items():
            new_f.attrs[key] = val

        # Process each scan group (scan-XXXX)
        for grp_name in old_f.keys():
            if not isinstance(old_f[grp_name], h5py.Group):
                continue
            
            old_grp = old_f[grp_name]
            new_grp = new_f.create_group(grp_name)

            # --- Update group attributes ---
            new_attrs = dict(old_grp.attrs)

            # Replace old 'scan_type' with new 'scan_name' + 'scan_device'
            if 'scan_type' in new_attrs:
                old_scan_type = new_attrs['scan_type']
                # Decode bytes to string if needed (HDF5 stores strings as bytes)
                if isinstance(old_scan_type, bytes):
                    old_scan_type = old_scan_type.decode('utf-8')
                
                if old_scan_type in SCAN_TYPE_MAP:
                    scan_name, scan_device = SCAN_TYPE_MAP[old_scan_type]
                else:
                    # Infer for unknown types (e.g., old mirror scans might have 'mirror' as scan_type)
                    print(f"Warning: Unknown scan_type '{old_scan_type}' in {grp_name} of {old_filename}")
                    if old_scan_type.startswith('slit'):
                        scan_name, scan_device = 'slit', old_scan_type
                    else:
                        scan_name, scan_device = old_scan_type, old_scan_type
                
                new_attrs['scan_name'] = scan_name
                new_attrs['scan_device'] = scan_device
                del new_attrs['scan_type']  # Remove old key

            # Add 'scan_motor' for mirror/caustic/lens scans if missing (matches motormov.py)
            scan_name = new_attrs.get('scan_name')
            if scan_name in ('mirror', 'caustic', 'lens') and 'scan_motor' not in new_attrs:
                # Try to get motor from old attrs or infer from context
                if scan_name == 'mirror' and 'scan_motor' not in new_attrs:
                    # Old mirror files might have 'motor' attribute
                    if 'motor' in new_attrs:
                        new_attrs['scan_motor'] = new_attrs['motor']
                        del new_attrs['motor']
                elif scan_name == 'caustic':
                    new_attrs['scan_motor'] = 'z_pos'
                elif scan_name == 'lens':
                    new_attrs['scan_motor'] = 'lens_pos'

            # Write updated attributes to new group
            for key, val in new_attrs.items():
                new_grp.attrs[key] = val

            # --- Copy DVF datasets (dvf_A1, dvf_B1) with their attributes ---
            for dset_name in old_grp.keys():
                if not isinstance(old_grp[dset_name], h5py.Dataset):
                    continue
                
                old_dset = old_grp[dset_name]
                # Copy dataset with gzip compression (matches h5file.py behavior)
                new_dset = new_grp.create_dataset(
                    dset_name, 
                    data=old_dset[...], 
                    compression='gzip'
                )
                # Copy dataset attributes (expo_time, acq_time, lens_pos, etc.)
                for key, val in old_dset.attrs.items():
                    new_dset.attrs[key] = val

    print(f"Converted: {old_filename} -> {new_filename}")
    return new_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert old HDF5 files to new motormov.py format.')
    parser.add_argument('input_dir', help='Directory with old HDF5 files (e.g., /home/gabriel.amici/testing_data/2026-03-19/)')
    parser.add_argument('--output-dir', default=None, help='Output directory for converted files (default: input_dir/converted/)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, 'converted')
    os.makedirs(output_dir, exist_ok=True)

    # Find all HDF5 files
    old_files = sorted(glob.glob(os.path.join(args.input_dir, '*.hdf5')))
    if not old_files:
        print(f"No .hdf5 files found in {args.input_dir}")
        return

    print(f"Converting {len(old_files)} files from {args.input_dir} to {output_dir}...")
    for old_path in old_files:
        try:
            convert_file(old_path, output_dir)
        except Exception as e:
            print(f"Error converting {old_path}: {e}")

if __name__ == '__main__':
    main()
