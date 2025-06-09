#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''Get information from the direct beam measurement. 

Usage:
   beamprofile.py [-h]
   beamprofile.py <direct beam npy file(s)>
   beamprofile.py [-h] [-b <box size>] <direct beam npy file(s)>
   beamprofile.py [<some options below>] <direct beam npy file(s)>

Options:
   -h      : this help message
   -o      : write results to file "beam_center.info"
   -d      : date
   -e      : date in seconds since epoch
   -m      : mass (integral = total counts) of the beam inside box
   -n      : mass normalized by box area (intensity per pixel)
   -t      : measurement exposure time
   --cx    : horziontal (x) center position in meters 
   --cy    : vertical (z) center position in meters 
   --px    : horziontal (x) center position in pixels
   --py    : vertical (z) center position in pixels
   --xprof : beam intensity profile of a horizontal section passing through (cx, cy)
   --xprof : beam intensity profile of a vertical section passing through (cx, cy)
   --skewx : skewness of the horizontal profile of the beam, passing through (cx, cy)
   --skewy : skewness of the vertical profile of the beam, passing through (cx, cy)
   --flat  : the flattened profile histogram in x and y directions

This program gets information from the direct beam measurement, as
beam center coordinates in pixels and/or meters, the histogram profile
in x and z directions passing through the beam center and the skewness
of these intensity distributions.

With no options, the program returns all the information. When an
option is given, the program returns just what was asked, in the order
given by the parameters at command line.

All information is by default printed to standard output. To store it
in a file, use '-o'.

This program is based on a script belonging to a former suite named
XS_treatment developed for SAXS analysis by the author and a
collaborator (Dennys Reis @ IF-USP-BR).

'''

__version__ = '0.2'
__author__ = 'Arnaldo G. Oliveira-Filho'
__email__ = 'arnaldo.filho@lnls.br'
__license__ = 'GPL'
__date__ = '2025-05-05'
__status__ = 'Development'
__copyright__ = 'Copyright 2025 by LNLS-CNPEM'
__local__ = 'LNLS - CNPEM - Campinas - BR'

import calendar
import getopt
import fabio
import numpy as np
import os
import pyFAI
import re
import sys
import time
#
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axisartist import floating_axes

sys.path.append('/usr/local/lib/XS-treatment/')
from libbeamanalysis import beam_analysis as LBA

#
def reformat_date(filetime):
    # Check if filetime needs to be reformatted.
    if (re.match('[A-Za-z]{3}', filetime)): return filetime
    
    # Split date and time from filetime.
    sd = filetime.split()

    # Date
    dt = sd[0].split('-')
    D = [ int(d) for d in dt ]
    # Get week day and month in abbreviate format.
    weekday = calendar.day_abbr[calendar.weekday(*D)]
    month = calendar.month_abbr[int(dt[1])]
    # Time
    tm = sd[1]

    # Build time standard string.
    tss = '{} {} {} {} {}'.format(weekday, month, D[2], tm, D[0])
    return tss

#
def set_output (lba, filename):
    '''Set the format for output data got from "lba" object, containing
    data and header info.

    '''

    # outd = dictionary with data; outs = string for complete output.
    outd, outs = {}, ''

    # File name.
    outs += f'\n>>> File: {filename}\n'

    # Matrix data size.
    outd['DataSize'] = lba.data.shape
    outs += '\n {:<30} {} pixels'.format('Data size:', lba.data.shape)
    
    # Box size.
    #outd['boxsize'] = lba.boxsize
    #outs += ' {:<30} {} x {} pixels\n'.format('Box size:',
    #                                            lba.boxsize, lba.boxsize)
    
    # Date.
    outd['date'] = lba.header['Date']
    outs += '\n {:<30} {}\n'.format('File date:', outd['date'])
    # Date since epoch. Useful when comparing time evolution of frames. 
    outd['epoch'] = calendar.timegm(time.strptime(outd['date']))
    outs += ' {:<30} {}\n'.format('Epoch file date (s):', outd['epoch'])

    # Exposure time.
    outd['Exptime'] = lba.header['Exptime']
    outs += ' {:<30} {:.4f}\n'.format('Exposure time:', lba.header['Exptime'])
    
    # Center in pixels.
    #outd['center'] = (lba.CMy, lba.CMx)
    #outs += '\n {:<30} ({:.4f}, {:.4f})\n'.format('Center (x, y; pixels):',
    #                                  lba.CMy, lba.CMx)

    # Center from flattened histogram, in pixels.
    outs += '\n'
    outd['CenterFlat'] = (lba.Cx, lba.Cy)
    outs += ' {:<30} ({:.4f}, {:.4f})\n'.format('Center (flat, x, y; pixels):',
                                      lba.Cx, lba.Cy)
    outd['massx'] = lba.massx
    outd['massy'] = lba.massy
    outs += '\n {:<30} ({:.4f}, {:.4f})\n'.format('Mass (flat; x, y, in #):',
                                      lba.massx, lba.massy)
    
    # Poni.
    outd['poni'] = (lba.PONI[0], lba.PONI[1])
    outs += (' {:<30} ({:.4f}, {:.4f})\n'.format('P.O.N.I. (in m): ',
                                                 outd['poni'][0], outd['poni'][1]))

    # Integral mass.
    #outd['mass'] = lba.Mass
    #outs += '\n {:<30} {:.4f}\n'.format('Total mass (in #):', lba.Mass)
    # Normalized mass.
    #outd['normass'] = lba.Mass / (lba.boxsize * lba.boxsize)
    #outs += ' {:<30} {:.4f}\n'.format('Normalized mass (#/pixel):',
    #                              outd['normass'])

    # Beam profiles through the center.
    outd['xcenterprofile'] = lba.xCenterProfile
    outd['ycenterprofile'] = lba.yCenterProfile

    # Skewness.
    sk, skn = lba.skewness(lba.xCenterProfile)
    outd['xskew']  = sk
    outd['xskewN'] = skn
    outs += '\n {:<30} {:8.4f} \t (normalized = {:6.3g})\n'.format('Skewness (x):', sk, skn)
    sk, skn = lba.skewness(lba.yCenterProfile)
    outd['yskew']  = sk
    outd['yskewN'] = skn
    outs += ' {:<30} {:8.4f} \t (normalized = {:6.3g})'.format('Skewness (y):', sk, skn)
    outs += '\n'

    # Flattened histograms.
    outd['flatx'] = lba.flatx
    outd['flaty'] = lba.flaty
    #outs += '\n {:<30} {}'.format('Flatten (x):', lba.flatx)
    #outs += '\n {:<30} {}\n'.format('Flatten (y):', lba.flaty)

    # Standard deviation from raw data.
    outd['sdrawx'] = lba.StdDevRawX
    outd['sdrawy'] = lba.StdDevRawY
    outs += '\n {:<30} {:8.4f} '.format('Std. Dev. Raw (x):', lba.StdDevRawX)
    outs += '\n {:<30} {:8.4f} '.format('Std. Dev. Raw (y):', lba.StdDevRawY)
    
    # Full-width at half maximum from flattened data.
    outd['flatfwhmx'] = lba.flatfwhmx
    outd['flatfwhmy'] = lba.flatfwhmy
    outs += '\n'
    outs += '\n {:<30} {:8.4f} '.format('FWHM, flat (x):', lba.flatfwhmx)
    outs += '\n {:<30} {:8.4f} '.format('FWHM, flat (y):', lba.flatfwhmy)

    outs+= "\n\n Gaussian fitting for x direction:\n"
    outd['prmxA0'] = lba.gaussfitx['A0']
    outd['prmxmu'] = lba.gaussfitx['mu']
    outd['prmxsg'] = lba.gaussfitx['sigma']
    stddev = np.sqrt(np.diag(lba.gaussfitx['cov']))
    outs += (f"  x_A0    = {outd['prmxA0']:10.4f}   ({stddev[0]:6.3f})\n"
             f"  x_mu    = {outd['prmxmu']:10.4f}   ({stddev[1]:6.3f})\n"
             f"  x_sigma = {outd['prmxsg']:10.4f}   ({stddev[2]:6.3f})\n")
    
    outs += "\n Gaussian fitting for y direction:\n"
    outd['prmyA0'] = lba.gaussfitx['A0']
    outd['prmymu'] = lba.gaussfitx['mu']
    outd['prmysg'] = lba.gaussfitx['sigma']
    stddev = np.sqrt(np.diag(lba.gaussfitx['cov']))
    outs += (f"  y_A0    = {outd['prmyA0']:10.4f}   ({stddev[0]:6.3f})\n"
             f"  y_mu    = {outd['prmymu']:10.4f}   ({stddev[1]:6.3f})\n"
             f"  y_sigma = {outd['prmysg']:10.4f}   ({stddev[2]:6.3f})\n")
    
    outs += '\n'
    return outd, outs

#
def out_opts (opts, Data, zerotime):
    '''Compose the output from Data defined by argument options opts. Return the
    composed table.

    '''

    # Lambda isopt: returns 0 or 1 if option myopt is defined.
    isopt = lambda myopt : len([ True for o in opts if myopt in o[0] ])
    # Set relative time.
    zt = zerotime if (isopt('-r')) else 0
                
    outs = []
    for o in opts:
        if (o[0] == '-d'):
            outs.append(Data['date'])
        elif (o[0] == '-e'):
            outs.append(Data['epoch'] - zt)
        elif (o[0] == '-t'):
            outs.append(Data['Exptime'])
        elif (o[0] == '--cx'):
            outs.append(Data['CenterFlat'][1])
        elif (o[0] == '--cy'):
            outs.append(Data['CenterFlat'][0])
        elif (o[0] == '--px'):
            outs.append(Data['poni'][1])
        elif (o[0] == '--py'):
            outs.append(Data['poni'][0])
        elif (o[0] == '--skewx'):
            outs.append(Data['xskew'])
            outs.append(Data['xskewN'])
        elif (o[0] == '--skewy'):
            outs.append(Data['zskew'])
            outs.append(Data['zskewN'])
        elif (o[0] == '--yprof'):
            zout  = '# Z, C = {} \n'.format(Data['CenterFlat'])
            zout += '# time(s), z pixel pos., counts \n'
            for z in Data['ycenterprofile']:
                zout += '{} {} {}\n'.format(Data['epoch'] - zt,
                                            str(z[0]), str(z[1]))
            outs.append(zout)
        elif (o[0] == '--xprof'):
            xout  = '# X, C = {} \n'.format(Data['CenterFlat'])
            xout += '# epoch(s), x pixel pos., counts \n'
            for x in Data['xcenterprofile']:
                xout += '{} {} {}\n'.format(Data['epoch'] - zt,
                                            str(x[0]), str(x[1]))
            outs.append(xout)
        elif (o[0] == '--flat'):
            outs.append(Data['flatx'])
            outs.append(Data['flaty'])
        elif (o[0] == '-m'):
            outs.append(Data['mass'])
        elif (o[0] == '-n'):
            outs.append(Data['normass'])
        else:
            pass

    return outs

#
def print_table(tableData, printoutfile):
    ''' Print out the results from table.
    '''
    target = 'beam.info' if printoutfile else None
    output = target and open(target, 'a') or sys.stdout
    for tD in tableData:
        for dd in tD:
            #if len(dd) == 1:
            #    output.write(f'{dd} \n')
            #else:
            output.write(f'{dd:.4f} ')
        output.write('\n')
    if target is not None:
        output.close()
    return

#
def print_out(OutS, filename, printoutfile=False):
    '''Print out data table, if there were selected options.
    '''
    target = 'beam.info' if printoutfile else None
    output = target and open(target, 'a') or sys.stdout
    output.write(f'{OutS}')
    if printoutfile:
        output.close()
    return

##
def opening_screen(sep, *header, totwidth=60, bsize=3):
    '''Format opening text screen with information about the program.'''

    # Clear current terminal screen before start
    #os.system('clear')
    
    border = bsize * sep 
    skipline = f'{border}{border:>{totwidth-bsize}}\n'
    tblines = str(3 * '{}\n'.format(totwidth * sep))
    
    opscreen = f'{tblines}{skipline}'
    for item in header:
        opscreen += f'{border}{item:^{totwidth-2*bsize}}{border}\n{skipline}'

    opscreen += tblines
    return opscreen


#
def splash_screen ():
    # Title and opening screen
    title      = 'Beam analysis for Carcara-X beamline'
    subheading = '@ IDS - LNLS - CNPEM - Campinas - Brazil'
    opscreen = opening_screen('*', title, subheading, __author__,
                              __email__, __date__,
                              __copyright__, __version__,
                              __status__)
    print(opscreen)

#
def header_read (header=None, filedate=None):
    '''Read the header of "datafile" if it has one, otherwise, create a
    stub.

    '''
    if header == None:
        header = {}
        # Pixel size is set to 1.
        header['PSize_1'] = 1
        header['PSize_2'] = 1
        # Date of file creation.
        header['Date'] = filedate
        # Exposure time
        header['Exptime'] = 1
        return header
    else:
        return datafile.header

#
def gaussian_fit (BD, xdata, ydata, C0, ax='x'):
    '''Performs a gaussian fit using the BD (LBA) class method. C0 is a
    guess from formerly calculated center.'''
    # Guess for A0 coefficient.
    A0 = np.max(ydata)
    fwhm = BD.flatfwhmx if ax == 'x' else BD.flatfwhmy 
    prm, cov = BD.gaussian_fit(xdata, ydata, A0, C0, fwhm / 2)

    if prm is not None:
        stddev = np.sqrt(np.diag(cov, 0))   # Std. dev.
        yfit = BD.gauss(xdata, *prm)        # Fit a gaussian.
        print(f"  {ax}_A0    = {prm[0]:10.4f}   ({stddev[0]:6.3f})\n"
              f"  {ax}_mu    = {prm[1]:10.4f}   ({stddev[1]:6.3f})\n"
              f"  {ax}_sigma = {prm[2]:10.4f}   ({stddev[2]:6.3f})\n")
    else:
        print(" Gaussian fitting aborted,")
        yfit = None
    return prm, cov, yfit, fwhm, s_raw

#
def histograms_fit (BD):
    ''' Fit a gaussian to x data.'''

    # Fit a gaussian to x data.
    print("\n Gaussian fitting for x:")
    xprm, xstd, xfit, xsraw, xfwhm = gaussian_fit(BD, BD.xrange, BD.flatx, BD.Cx, ax='x')
    
    # Fit a gaussian to y data.
    print("\n Gaussian fitting for y:")
    yprm, ystd, yfit, ysraw, yfwhm = gaussian_fit(BD, BD.yrange, BD.flaty, BD.Cy, ax='y')
        
    return ((xprm, xstd, BD.xrange, xfit, xsraw, xfwhm),
            (yprm, ystd, BD.yrange, yfit, ysraw, yfwhm))

#
def histograms_plot (BD):
    fig, ax = plt.subplots(figsize=(10,7))
    
    # The relocation of axes.
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1)
    ax_histy = divider.append_axes("right", 2.4, pad=0.1)
    
    ax.invert_yaxis()
    ax.imshow(BD.data)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    ax_histx.plot(BD.xrange, BD.flatx)
    #
    ax_histy.invert_yaxis()
    ax_histy.plot(BD.flaty, BD.yrange)

    if BD.Gaussfitx['fitcurve'] is not None:
        ax_histx.plot(BD.xrange, BD.Gaussfitx['fitcurve'])
        
    if BD.Gaussfity['fitcurve'] is not None:
       ax_histy.plot(BD.Gaussfity['fitcurve'], BD.yrange)

    #ax_histx.set_yticks([0, 50, 100])
    #ax_histy.set_xticks([0, 50, 100])
    #ax[1].set_xlabel("x (pixels)")
    #ax[1].set_ylabel("counts")
    #ax[2].set_xlabel("y (pixels)")
    #ax[2].set_ylabel("counts")
    plt.tight_layout(h_pad=1.2)
    plt.show()

    
#
def main ():
    '''Get command line options, open file(s) and calculate info with class methods
    accordingly to demand.'''

    # Get output option.
    try:
        line = getopt.getopt(sys.argv[1:], "hdemnrtoqb:",
                             ['cx', 'cy', 'px', 'py',
                              'xprof', 'yprof', 'skewx', 'skewy',
                              'projfit', 'projshow'])
    except getopt.GetoptError as err:
        print ('\n\n ERROR: ', str(err),'\b.')
        sys.exit(1)
    #
    opts, files = line[0], line[1]

    # Initialize variable for initial time and for final data.
    zerotime  = 0
    tableData = []

    # Define a default box size.
    boxsize = 500
    otheropts, printoutfile, quietsplash = False, False, False
    projfit, projshow = False, False
    for o in opts:
        if (o[0] == '-h'):
            help('beamprofile')
            return
        elif (o[0] == '-b'):
            boxsize = int(o[1])
        elif (o[0] == '-o'):
            printoutfile = True
        elif (o[0] == '-q'):
            quietsplash = True
        elif o[0] == '--projshow':
            projshow = True
            projfit = True
        elif o[0] == '--projfit':
            projfit = True
        else:
            otheropts = True

    # Print out a splash screen.
    if not quietsplash:
        splash_screen()
        print('### Center of mass coordinates, P.O.N.I., and'
              + '\n### other statistical information'
              + ' from beam image.')
            
    # Check input file.
    if len(files) == 0:
        print (' ERROR: no file given.')
        sys.exit(1)

    # Open data file.
    for FL in files:
        # Read data and header (if available) from file. If it is an
        # EDF file, open it with fabio and get header information,
        # otherwise creates a stub header.
        if re.search("edf", FL, flags=re.IGNORECASE):
            BD = fabio.open(FL)
            Bdata = BD.data
            Bheader = BD.header
        else:
            # If file contains only numpy data.
            filedate = time.ctime(os.path.getctime(FL))
            Bdata = np.load(FL)
            Bheader = header_read(None, filedate=filedate)
        
        # Instantiate beam information.
        BD = LBA(Bdata, Bheader, boxsize)

        # Set dictionary and strings for output data.
        OutData, OutStr = set_output(BD, FL)
        
        # Initial time is taken from first sample.
        if zerotime == 0:
            zerotime = OutData['epoch']

        # Print out file information or accumulate required data.
        if otheropts:
            tableData.append(out_opts(opts, OutData, zerotime))
        else:
            print_out(OutStr, FL, printoutfile)
    
        # If there were argument options, print data.
        if len(tableData) != 0:
            print_table(tableData, printoutfile)

        # Fit and/or show projectedh istograms.
        #if projfit:
        #    Xres, Yres = histograms_fit(BD)
            
        if projshow:
            histograms_plot(BD)

            
    return
                
##
if __name__ == '__main__':
    main()
