import numpy as np
import h5py
from matplotlib import pyplot as plt

FNAME = 'caustic_02.h5'
PIXEL_FACTOR = 0.48
X_START, X_END = 1800, 2500     # if none is set, it automatically takes the minimum and/or maximum
Y_START, Y_END = 800, 1500      # if none is set, it automatically takes the minimum and/or maximum
SAVE_FIG = False

def read_caustic(fname='test.h5'):
    
    f = h5py.File(fname, 'r')
    
    d = dict()
    
    d['zstart'] = f.attrs['zstart'] 
    d['zfin'] = f.attrs['zfin'] 
    d['zstep'] = f.attrs['zstep']
    d['positions'] = f.attrs['positions']
    d['nz'] = len(d['positions'])

    img0 = np.array(f['step_{0:03d}'.format(0)])
    
    d['ny'], d['nx'] = img0.shape

    caustic3D = np.zeros((d['nz'], d['ny'], d['nx']))
    
    for i in range(d['nz']):
        dataset = 'step_{0:03d}'.format(i)
        caustic3D[i] = np.array(f[dataset])

    return caustic3D, d

def calc_fwmh(x, y):
    """ FWMH functions.

    Args:
        x (numpy array): x array.
        y (numpy array): f(x) array.
    
    Return:
        fwmh (float).
    """
    idx_max = np.argmax(y)
    y_max = y[idx_max]
    y_mid = y_max/2

    idx_left = np.argmin(np.abs(y[:idx_max] - y_mid))
    idx_right = np.argmin(np.abs(y[idx_max:] - y_mid)) + idx_max

    return x[idx_right] - x[idx_left]


caustic, d = read_caustic(fname=FNAME)

nx = d['nx']
ny = d['ny']
zs = d['positions']

cx = np.sum(caustic, axis=1)[::-1].T
cy = np.sum(caustic, axis=2)[::-1].T

X_START = 0 if X_START == None else X_START
X_END = nx if X_END == None else X_END

Y_START = 0 if Y_START == None else Y_START
Y_END = ny if Y_END == None else Y_END

#### CROP 

cx_new = cx[X_START:X_END, :]
cy_new = cy[Y_START:Y_END, :]

### INTENSITY
intensity_along_z = np.sum(cx_new, axis=0)

### FWMHSs
FWMHs_x = list()
for i in range(cx_new.shape[1]):
    y = cx_new[:, i]
    x = np.arange(X_START, X_END, 1)
    fwmh_x = calc_fwmh(x=x, y=y)
    FWMHs_x.append(fwmh_x)
FWMHs_x = np.array(FWMHs_x)

FWMHs_y = list()
for i in range(cy_new.shape[1]):
    y = cy_new[:, i]
    x = np.arange(Y_START, Y_END, 1)
    fwmh_y = calc_fwmh(x=x, y=y)
    FWMHs_y.append(fwmh_y)
FWMHs_y = np.array(FWMHs_y)

### PLOTS
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9,5), sharex=True)
fig.suptitle('Caustic')

axs[0, 0].imshow(cx_new, aspect='auto', extent=[d['zfin'], d['zstart'], X_END, X_START])
axs[0, 0].set_ylabel('X [px]')

axs[1, 0].imshow(cy_new, aspect='auto', extent=[d['zfin'], d['zstart'], Y_END, Y_START])
axs[1, 0].set_xlabel('Z Screen Position [mm]')
axs[1, 0].set_ylabel('Y [px]')

axs[0, 1].plot(zs[::-1], intensity_along_z/intensity_along_z.max(), linewidth=3)
axs[0, 1].set_ylabel('Normalized Intensity')
axs[0, 1].grid(which='major', alpha=0.3)
axs[0, 1].grid(which='minor', alpha=0.1)
axs[0, 1].minorticks_on()

axs[1, 1].plot(zs[::-1], FWMHs_x * PIXEL_FACTOR, '-o', label='x', linewidth=3)
axs[1, 1].plot(zs[::-1], FWMHs_y * PIXEL_FACTOR, '-o', label='y', linewidth=3)
axs[1, 1].set_ylabel(r'$FWMH~[\mu m]$')
axs[1, 1].set_xlabel('Z Screen Position [mm]')
axs[1, 1].grid(which='major', alpha=0.3)
axs[1, 1].grid(which='minor', alpha=0.1)
axs[1, 1].legend()
axs[1, 1].minorticks_on()

fig.tight_layout()
if SAVE_FIG:
    fig.savefig('{}.png'.format(FNAME.split('.h5')[0]), dpi=400)
fig.show()