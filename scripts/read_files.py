import numpy as np
import matplotlib.pyplot as plt
import os

all_files = os.listdir('./')

name_files = [name_file for name_file in all_files if '.npy' in name_file]

name_files.sort()

files_selected = [np.load(name_file) for name_file in name_files]

zs = np.array([z.split('_')[3] for z in name_files], dtype='float')

projections_x = [np.array(file.sum(axis=0)) for file in files_selected]
projections_y = [np.array(file.sum(axis=1)) for file in files_selected]

xs = np.arange(len(projections_x[0]))
ys = np.arange(len(projections_y[0]))

def calc_fwmh(y, x):
    idx_max = np.argmax(y)
    y_max = y[idx_max]
    y_mid = y_max/2

    idx_left = np.argmin(np.abs(y[:idx_max] - y_mid))
    idx_right = np.argmin(np.abs(y[idx_max:] - y_mid)) + idx_max

    return x[idx_right] - x[idx_left]

factor = 0.48

fwmhs_x = np.array([calc_fwmh(y=projection, x=np.arange(len(projection))) for projection in projections_x]) * factor
fwmhs_y = np.array([calc_fwmh(y=projection, x=np.arange(len(projection))) for projection in projections_y]) * factor

plt.figure(figsize=(6,4))
plt.plot(zs, fwmhs_x, '-o',  label='x', linewidth=3)
plt.plot(zs, fwmhs_y, '-o', label='y', linewidth=3)
plt.xlabel('z [mm]')
plt.ylabel(r'FWMH [$\mu m]$')
plt.ylim((50, 110))
plt.legend()
plt.grid(which='major', alpha=0.3)
plt.grid(which='minor', alpha=0.1)
plt.minorticks_on()
plt.tight_layout()
plt.show()
