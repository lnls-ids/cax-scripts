#!/usr/bin/python3
# -*- coding: utf-8 -*-

from numba import jit
import numpy as np
from scipy.optimize import curve_fit

class beam_analysis ():
    '''This class contains the basic methods to calculate the center of
    mass and other statistical information from a direct beam image in
    EDF or numpy data format.

    '''
    def __init__ (self, beamdata, beamheader=None, boxsize=50):

        # The center of mass of the whole image.
        self.data = np.float64(beamdata)
        #(imcy, imcx), imm = self.matrix_center(self.data)

        # The header.
        self.header = beamheader

        # Box size.
        self.boxsize = boxsize
        
        # Pixel size definition.
        if beamheader is None:
            self.PSize1 = 48.e-6  # m
            self.PSize2 = 48.e-6  # m
        else:
            self.PSize1 = float(beamheader['PSize_1'])
            self.PSize2 = float(beamheader['PSize_2'])
        
        # The flattened histogram in x and y directions.
        self.flattened_profile()

        # Center of mass.
        self.Cx, self.massx = self.vector_mass_center(self.flatx)
        self.Cy, self.massy = self.vector_mass_center(self.flaty)
        
        # Narrow the array down to a box of size 'boxsize' around the center
        # formerly calculated and recalculate its center.
        R = 0.5 * self.boxsize
        ax, bx  = int(self.Cx - R), int(self.Cx + R)
        ay, by  = int(self.Cy - R), int(self.Cy + R)
        #az, bz  = int(imcy - R), int(imcy + R)
        #ax, bx  = int(imcx - R), int(imcx + R)

        # Calculate the center of mass of the box in pixels.
        #(cy, cx), self.Mass = self.matrix_center(self.data[az:bz, ax: bx])
        (cx, cy), self.Mass = self.matrix_center(self.data[ay:by, ax: bx])
        self.CMx = int(cx + ax)
        self.CMy = int(cy + ay)

        # Point of Normal Incidence (P.O.N.I.) of the beam, in
        # meters. It is equivalent to the center of mass.
        self.PONI = self.CMx * self.PSize1, self.CMy * self.PSize2
        
        # Define beam profile histogram in horizontal and vertical
        # directions passing through the center.
        self.profile_cut()

        # Standard deviation from raw data.
        self.xrange = np.array(range(len(self.flatx)))
        self.yrange = np.array(range(len(self.flaty)))
        self.StdDevRawX = self.std_dev_raw(self.xrange, self.flatx)
        self.StdDevRawY = self.std_dev_raw(self.yrange, self.flaty)
        
        # Full-width at half maximum from flattened data.
        self.flatfwhmx = self.fwhm_flat(self.flatx)
        self.flatfwhmy = self.fwhm_flat(self.flaty)

        # Gaussian fit.
        self.gaussfitx, self.gaussfity = self.gaussian_fit()

    #
    def vector_mass_center (self, data):
        # Total mass.
        mass = np.sum(data)
        # Center of mass.
        mc = np.sum(np.arange(1, data.size + 1) * data) / mass
        return np.float64(mc), np.float64(mass)
        
    #
    #@jit(nopython=True, parallel = True)
    def matrix_center (self, data):
        '''  Center of mass of a given numpy array "data". '''
        
        cx, cy, mass = 0.0, 0.0, 0.0
        NL = data[:,0].size   # Number of lines of data.
        NC = data[0].size     # Number of columns of data.

        # Run through elements of data and add weighted positions.
        for ii in range(NL):
            for jj in range(NC):
                # Skip if value is not well defined (= -1, usually masked
                # positions).
                if (data[ii, jj] == -1): continue
                # Weighted coordinates.
                cy += data[ii, jj] * ii
                cx += data[ii, jj] * jj
                mass += data[ii, jj]     # Total mass (integral) of the area.
        # Normalize the coordinates.
        cy /= mass
        cx /= mass

        # (Poni1, Poni2) correspond to axes (z, x). m is the total "mass" (the
        # integral of the image).
        return np.float64((cy, cx)), np.float64(mass)


    #@jit(nopython=True)
    def profile_cut (self):
        '''Beam profile in x and y direction, passing through the calculated
        center of the beam.

        '''
        R = 0.5 * self.boxsize    # Half-box size.
        # Vertical and horizontal limits.
        ax, bx = int(self.Cx - R), int(self.Cx + R)
        ay, by = int(self.Cy - R), int(self.Cy + R)

        # Histograms through the beam center in vertical and
        # horizontal directions. These are cuts of the total image,
        self.xCenterProfile = self.data[int(self.Cy), ax:bx+1]
        self.yCenterProfile = self.data[ay:by+1, int(self.Cx)]
        return

    #
    #@jit(nopython=True)
    def skewness (self, profile):
        '''Calculate the skewness of the given histogram. This function
        considers the 'profile' of the beam, in the vertical or
        horizontal direction, through the beam center, as a
        distribution and evaluates its assymmetry by the calulation of
        its skewness. The unbiased (sample) Fisher's definition is
        used here, which is the standardized third momentum.

        '''
        # The profile through the center.
        #vc  = profile[:, 1]
        vc  = profile[:]
        nvc = float(vc.size)   # Number of bins.
        # Average value of the 'distribution'.
        av = np.average(vc)
        # Third momentum of the distribution.
        v3 = np.array([ (x - av)**3  for x in vc ]) 
        m3 = np.average(v3)                          
        # Standard deviation of the sample.
        v2 = np.array([ (x - av)**2  for x in vc ])
        s2 = np.sum(v2) / (nvc - 1)                   
        # Skewness and "normalized" skewness (by interval size).
        sk = m3 / s2**1.5
        return sk, sk / float(nvc)

    #
    def flattened_profile (self):
        '''Projection (flattening) of the 2D histogram onto x and y directions.'''
        nx, ny = self.data.shape
        self.flatx = np.array([ np.sum(self.data[:, jj]) for jj in range(ny) ]) / nx
        self.flaty = np.array([ np.sum(self.data[jj, :]) for jj in range(nx) ]) / ny
        return

    #
    def std_dev_raw(self, x, y):
        '''Calculate the standard deviation from raw data.'''
        return np.sqrt(np.sum(y * np.square(x)) / np.sum(y) -
                       (np.sum(y * x) / np.sum(y))**2)
   
    #
    def fwhm_flat (self, ydata):
        '''Full width at half maximum (fwhm) from raw data.'''
        A0 = np.max(ydata)       # Max value.
        iA0 = np.argmax(ydata)   # Index of max value.
        # Positions (pixels) where y < A0 /2.
        iA1 = np.max(np.where(ydata[:iA0] < A0 / 2)[0])
        iA2 = np.max(np.asarray(A0 / 2 < ydata).nonzero())
        # Distance in indexes = # pixels.
        return np.abs(iA1 - iA0) + np.abs(iA2 - iA0)

    #
    def gauss (self, xdata, *p0):
        '''Gaussian function.'''
        if len(p0) == 1:
            A0, mu, sigma = p0[0]
        else:
            A0, mu, sigma = p0
        return A0 * np.exp(-(xdata - mu)**2 / (2. * sigma**2))
    
    # 
    def gaussian_fit (self):
        '''Fit a gaussian curve to data sets.'''

        self.Gaussfitx, self.Gaussfity = {}, {}
        
        # For x direction data.
        # Guess parameters: A0, mu, sigma.
        p0 = [np.max(self.flatx), self.CMx, self.flatfwhmx]
        try:
            prm, cov = curve_fit(self.gauss, self.xrange,
                                 self.flatx, p0=p0)
            self.Gaussfitx['A0'] = prm[0]
            self.Gaussfitx['mu'] = prm[1]
            self.Gaussfitx['sigma'] = prm[2]
            self.Gaussfitx['cov'] = cov
        except Exception as err:
            print(f" (gaussian_fit) ERROR fitting x-data: {err}")

        if prm is not None:
            self.Gaussfitx['fitcurve'] = self.gauss(self.xrange, prm)
        else:
            self.Gaussfitx['fitcurve'] = None
            
        # For y direction data.
        # Guess parameters: A0, mu, sigma.
        p0 = [np.max(self.flaty), self.CMy, self.flatfwhmy]
        try:
            prm, cov = curve_fit(self.gauss, self.yrange, self.flaty,
                                 p0=p0)
            self.Gaussfity['A0'] = prm[0]
            self.Gaussfity['mu'] = prm[1]
            self.Gaussfity['sigma'] = prm[2]
            self.Gaussfity['cov'] = cov
        except Exception as err:
            print(" (gaussain_fit) ERROR fitting y-data: {err}")
            return None, None
 
        if prm is not None:
            self.Gaussfity['fitcurve'] = self.gauss(self.yrange, prm)
        else:
            self.Gaussfity['fitcurve'] = None
    
        return self.Gaussfitx, self.Gaussfity
