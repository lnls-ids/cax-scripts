
import h5py

import numpy as np
from scipy.optimize import curve_fit

from . import utils

def caustic_analysis(filename, filedir):

    file = '/'.join([filedir,filename])

    f = h5py.File(name=file, mode='r')

    positions = [f[scaname].attrs['z_pos'] for scaname in f]

    caustic3d = []
    for scaname in f:
        img = np.array(f[scaname]['dvf2'])
        caustic3d.append(img)

    causticx = np.sum(caustic3d, axis=1).T
    causticy = np.sum(caustic3d, axis=2).T

    fwhmsx = [utils.full_width(data=profilex,coords=positions)[0] for profilex in causticx.T]
    fwhmsy = [utils.full_width(data=profiley,coords=positions)[0] for profiley in causticy.T]

    causticx_params, covx = curve_fit(utils.caustic_func, positions, fwhmsx, p0=None)
    causticy_params, covy = curve_fit(utils.caustic_func, positions, fwhmsy, p0=None)

    resultsx = utils.caustic_processing(causticx_params, positions)
    resultsy = utils.caustic_processing(causticy_params, positions)

    return resultsx, resultsy