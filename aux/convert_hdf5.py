#!/usr/bin/env python
"""Convert old HDF5 files to new format matching current motormov.py output."""
import argparse
import h5py
import os
import glob

# Mapping from old 'scan_type' to new
# (scan_name, scan_device) based on motormov.py
SCAN_TYPE_MAP = {
    # CAXMirrorMove: scan_name='mirror', scan_device='mirror'
    'mirror': ('mirror', 'mirror'),
    # CAXSlitMove: scan_name='slit', scan_device=slit_name
    'slit_A1': ('slit', 'slit_A1'),
    'slit_B1': ('slit', 'slit_B1'),
    # CAXCausticMove: scan_name='caustic', scan_device='dvf_B1'
    'caustic': ('caustic', 'dvf_B1'),
    # CAXLensMove: scan_name='lens', scan_device='dvf_B1'
    'lens': ('lens', 'dvf_B1'),
}


def _resolve_scan_type(old_scan_type, grp_name, old_filename):
    """Map old scan_type string to (scan_name, scan_device)."""
    if isinstance(old_scan_type, bytes):
        old_scan_type = old_scan_type.decode('utf-8')
    if old_scan_type in SCAN_TYPE_MAP:
        return SCAN_TYPE_MAP[old_scan_type]
    print(f"Warning: Unknown scan_type '{old_scan_type}'"
          f" in {grp_name} of {old_filename}")
    if old_scan_type.startswith('slit'):
        return 'slit', old_scan_type
    return old_scan_type, old_scan_type


def _update_scan_type_attrs(new_attrs, grp_name, old_filename):
    """Replace 'scan_type' with 'scan_name' and 'scan_device' in-place."""
    if 'scan_type' not in new_attrs:
        return
    scan_name, scan_device = _resolve_scan_type(
        new_attrs['scan_type'], grp_name, old_filename
    )
    new_attrs['scan_name'] = scan_name
    new_attrs['scan_device'] = scan_device
    del new_attrs['scan_type']


def _update_scan_motor(new_attrs):
    """Add 'scan_motor' for mirror/caustic/lens scans when missing."""
    scan_name = new_attrs.get('scan_name')
    if (scan_name not in ('mirror', 'caustic', 'lens') or
        'scan_motor' in new_attrs):
        return
    if scan_name == 'mirror':
        if 'motor' in new_attrs:
            new_attrs['scan_motor'] = new_attrs.pop('motor')
    elif scan_name == 'caustic':
        new_attrs['scan_motor'] = 'z_pos'
    elif scan_name == 'lens':
        new_attrs['scan_motor'] = 'lens_pos'


def _convert_group(old_f, new_f, grp_name, old_filename):
    """Convert a single scan group from old HDF5 format to new."""
    old_grp = old_f[grp_name]
    new_grp = new_f.create_group(grp_name)

    new_attrs = dict(old_grp.attrs)
    _update_scan_type_attrs(new_attrs, grp_name, old_filename)
    _update_scan_motor(new_attrs)
    for key, val in new_attrs.items():
        new_grp.attrs[key] = val

    for dset_name in old_grp.keys():
        if not isinstance(old_grp[dset_name], h5py.Dataset):
            continue
        old_dset = old_grp[dset_name]
        new_dset = new_grp.create_dataset(
            dset_name, data=old_dset[...], compression='gzip'
        )
        for key, val in old_dset.attrs.items():
            new_dset.attrs[key] = val


def convert_file(old_path, output_dir):
    """Convert a single old HDF5 file to new format."""
    old_filename = os.path.basename(old_path)
    name, ext = os.path.splitext(old_filename)
    new_filename = f"{name}_converted{ext}"
    new_path = os.path.join(output_dir, new_filename)

    with h5py.File(old_path, 'r') as old_f, h5py.File(new_path, 'w') as new_f:
        for key, val in old_f.attrs.items():
            new_f.attrs[key] = val
        for grp_name in old_f.keys():
            if isinstance(old_f[grp_name], h5py.Group):
                _convert_group(old_f, new_f, grp_name, old_filename)

    print(f"Converted: {old_filename} -> {new_filename}")
    return new_path


def cmd_line():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert old HDF5 files to new motormov.py format.'
        )
    parser.add_argument(
        'input_dir',
        help=("Directory with old HDF5 files"
              " (e.g., ~/testing_data/2026-03-19/)")
            )
    parser.add_argument(
        '--output-dir', default=None,
        help=("Output directory for converted files "
              "(default: input_dir/converted/)")
              )
    return parser.parse_args()


def main():
    """Main function to parse arguments and convert files."""
    args = cmd_line()

    output_dir = args.output_dir or os.path.join(args.input_dir, 'converted')
    os.makedirs(output_dir, exist_ok=True)

    # Find all HDF5 files
    old_files = sorted(glob.glob(os.path.join(args.input_dir, '*.hdf5')))
    if not old_files:
        print(f"No .hdf5 files found in {args.input_dir}")
        return

    print(f"Converting {len(old_files)} files"
          f" from {args.input_dir} to {output_dir}...")
    for old_path in old_files:
        try:
            convert_file(old_path, output_dir)
        except Exception as e:
            print(f"Error converting {old_path}: {e}")


if __name__ == '__main__':
    main()
