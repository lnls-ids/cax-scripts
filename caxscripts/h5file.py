import time
import h5py


class HDF5File:
    """Class for working with HDF5 files.

    Can be used to create new files and save data.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file.
    filedir : str
        Path to the HDF5 file directory. Defalts to the current directory.
    mode : str, {'r', 'r+', 'w', 'w-', 'x', 'a'}
        HDF5 file opening mode. Defaults to 'w'.
        * r        Readonly, file must exist (default)
        * r+       Read/write, file must exist
        * w        Create file, truncate if exists
        * w- or x  Create file, fail if exists
        * a        Read/write if exists, create otherwise
    """
    def __init__(self, filename, filedir='.', mode='w'):
        """Initialize the HDF5File object."""
        self.file = '/'.join([filedir, filename])

        with h5py.File(name=self.file, mode=mode) as f:
            local_time = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S UTC%Z", local_time)
            f.attrs['start time'] = formatted_time

    def save_metadata(self, metadata_dict):
        """Save metadata to the HDF5 file."""
        with h5py.File(name=self.file, mode='a') as f:
            f.attrs.update(metadata_dict)

    def save_dataset(self, dsetname, dsetdata,
                     dsetmetadata=None, grpname='', t0=None):
        """Save a dataset to the HDF5 file."""
        dsetname = '/'.join([grpname, dsetname])
        if dsetmetadata is None:
            dsetmetadata = {}
        with h5py.File(name=self.file, mode='a') as f:
            dset = f.create_dataset(name=dsetname, data=dsetdata,
                                    compression='gzip')
            dset.attrs.update(dsetmetadata)
            if t0:
                dset.attrs['ellapsed time'] = time.time() - t0

    def save_group(self, grpname, grpmetadata=None):
        """Save a group to the HDF5 file."""
        if grpmetadata is None:
            grpmetadata = {}
        with h5py.File(name=self.file, mode='a') as f:
            grp = f.create_group(name=grpname)
            grp.attrs.update(grpmetadata)


# todo: update with the implemented logic for groups
if __name__ == '__main__':

    directory = '.'

    h5file = HDF5File(filename='test.h5', filedir=directory)
    h5file.save_metadata(metadata_dict={
        'slit_top': 1,
        'slit_bottom': 2,
        'slit_left': 3,
        'slit_right': 4,
        'step': 0.1
    })

    metadata = {
        'slit_top': 41.5,
        'slit_bottom': 38,
        'slit_left': 35,
        'slit_right': 48.2
    }
    h5file.save_dataset(dsetname='movement-top-1',
                        dsetmetadata=metadata,
                        dsetdata=[[1, 2, 3], [4, 5, 6]],
                        t0=time.time())

    f = h5py.File(name=directory+'test.h5')
    print(dict(f.attrs))
    print(dict(f['movement-top-1'].attrs))
    # print(np.array(f['movement-top-1']))
    f.close()
