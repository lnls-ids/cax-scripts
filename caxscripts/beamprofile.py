#!/usr/bin/env python3
"""Get information from the direct beam measurement.

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
   --cx    : horizontal (x) center position in meters 
   --cy    : vertical (z) center position in meters 
   --px    : horizontal (x) center position in pixels
   --py    : vertical (z) center position in pixels
   --xprof : beam intensity profile of a horizontal section passing
        through (cx, cy)
   --yprof : beam intensity profile of a vertical section passing
        through (cx, cy)
   --skewx : skewness of the horizontal profile of the beam, passing
        through (cx, cy)
   --skewy : skewness of the vertical profile of the beam, passing
        through (cx, cy)
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
"""
__version__ = '0.2'
__author__ = 'Arnaldo G. Oliveira-Filho'
__email__ = 'arnaldo.filho@lnls.br'
__license__ = 'GPL'
__date__ = '2025-05-05'
__status__ = 'Development'
__copyright__ = 'Copyright 2025 by LNLS-CNPEM'
__local__ = 'LNLS - CNPEM - Campinas - BR'

from typing import Any
import calendar
import getopt
import fabio
import numpy as np
import os
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


def reformat_date(filetime: str) -> str:
    """Reformat date string from filetime to standard format."""
    # Check if filetime needs to be reformatted.
    if (re.match('[A-Za-z]{3}', filetime)): return filetime

    # Split date and time from filetime.
    sd = filetime.split()

    # Date
    dt = sd[0].split('-')
    dd = [int(d) for d in dt]
    # Get week day and month in abbreviate format.
    weekday = calendar.day_abbr[calendar.weekday(*dd)]
    month = calendar.month_abbr[int(dt[1])]
    # Time
    tm = sd[1]

    # Build time standard string.
    tss = f"{weekday} {month} {dd[2]:02d} {tm} {dd[0]}"
    return tss


def set_output(lba: LBA, filename: str) -> tuple[dict[str, Any], str]:
    """Set the format for output data got from "lba" object.

    It contains data and header info.
    """
    # outd = dictionary with data; outs = string for complete output.
    outd, outs = {}, ''

    # File name.
    outs += f'\n>>> File: {filename}\n'

    # Matrix data size.
    outd['DataSize'] = lba.data.shape
    outs += '\n {:<30} {} pixels'.format('Data size:', lba.data.shape)

    # Box size.
    # outd['boxsize'] = lba.boxsize
    # outs += ' {:<30} {} x {} pixels\n'.format('Box size:',
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
    # outd['center'] = (lba.CMy, lba.CMx)
    # outs += '\n {:<30} ({:.4f}, {:.4f})\n'.format('Center (x, y; pixels):',
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
                                                 outd['poni'][0],
                                                 outd['poni'][1]))

    # Integral mass.
    # outd['mass'] = lba.Mass
    # outs += '\n {:<30} {:.4f}\n'.format('Total mass (in #):', lba.Mass)
    #  Normalized mass.
    # outd['normass'] = lba.Mass / (lba.boxsize * lba.boxsize)
    # outs += ' {:<30} {:.4f}\n'.format('Normalized mass (#/pixel):',
    #                               outd['normass'])

    # Beam profiles through the center.
    outd['xcenterprofile'] = lba.xCenterProfile
    outd['ycenterprofile'] = lba.yCenterProfile

    # Skewness.
    sk, skn = lba.skewness(lba.xCenterProfile)
    outd['xskew']  = sk
    outd['xskewN'] = skn
    outs += (f"\n {'Skewness (x):':<30} {sk:8.4f} "
             f"\t (normalized = {skn:6.3g})\n")
    sk, skn = lba.skewness(lba.yCenterProfile)
    outd['yskew']  = sk
    outd['yskewN'] = skn
    outs += (f"\n {'Skewness (y):':<30} {sk:8.4f} "
             f"\t (normalized = {skn:6.3g})\n")
    outs += '\n'

    # Flattened histograms.
    outd['flatx'] = lba.flatx
    outd['flaty'] = lba.flaty
    # outs += '\n {:<30} {}'.format('Flatten (x):', lba.flatx)
    # outs += '\n {:<30} {}\n'.format('Flatten (y):', lba.flaty)

    # Standard deviation from raw data.
    outd['sdrawx'] = lba.StdDevRawX
    outd['sdrawy'] = lba.StdDevRawY
    outs += f'\n {"Std. Dev. Raw (x):":<30} {lba.StdDevRawX:8.4f} '
    outs += f'\n {"Std. Dev. Raw (y):":<30} {lba.StdDevRawY:8.4f} '

    # Full-width at half maximum from flattened data.
    outd['flatfwhmx'] = lba.flatfwhmx
    outd['flatfwhmy'] = lba.flatfwhmy
    outs += f"\n {"FWHM, flat (x):":<30} {lba.flatfwhmx:8.4f} "
    outs += f"\n {"FWHM, flat (y):":<30} {lba.flatfwhmy:8.4f} "
 
    outs += "\n\n Gaussian fitting for x direction:\n"
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


options = {
    '-d'      : 'date',
    '-e'      : 'epoch',
    '-t'      : 'Exptime',
    '--cx'    : 'CenterFlat[1]',
    '--cy'    : 'CenterFlat[0]',
    # '--px'    : 'poni[1]',
    # '--py'    : 'poni[0]',
    '--skewx' : 'xskew',
    '--skewy' : 'yskew',
    # '--yprof' : 'ycenterprofile',
    # '--xprof' : 'xcenterprofile',
    # '--flat'  : 'flatx, flaty',
    '-m'      : 'mass',
    '-n'      : 'normass'
}


def is_opt(myopt: str) -> bool:
    """Returns 0 or 1 if option myopt is defined."""
    return any(myopt in o for o in options)


def coord_out(
        data: dict[str, Any],
        direction: str,
        zerotime: float) -> str:
    """."""
    centerflat = data['CenterFlat']
    centerprofile = (data['xcenterprofile']
                     if direction == 'x' else data['ycenterprofile'])
    zout  = (f"# {direction.upper()}, C = {centerflat} \n"
                "# time(s), z pixel pos., counts \n")
    for z in centerprofile:
        zout += f"{data['epoch'] - zerotime} {z[0]} {z[1]}\n"
    return zout


def out_opts(
        opts: list[tuple[str, Any]],
        data: dict[str, Any],
        zerotime: float) -> list[Any]:
    """Compose the output from Data defined by argument options opts.

    Return the composed table.
    """
    # Set relative time.
    zt = zerotime if (is_opt('-r')) else 0

    # General variables.
    outs = [data[options[o[0]]] for o in opts]

    # Complete output for skewN.
    if is_opt('--skewx'):
        outs.append(data['xskewN'])
    if is_opt('--skewy'):
        outs.append(data['yskewN'])

    # Special cases.
    for o in opts:
        if (o[0] == '--px'):
            outs.append(data['poni'][1])
        elif (o[0] == '--py'):
            outs.append(data['poni'][0])
        elif (o[0] == '--skewx'):
            outs += [data['xskew'], data['xskewN']]
        elif (o[0] == '--skewy'):
            outs += [data['yskew'], data['yskewN']]
        elif (o[0] == '--yprof'):
            outs.append(coord_out(data, 'y', zt))
        elif (o[0] == '--xprof'):
            outs.append(coord_out(data, 'x', zt))
        elif (o[0] == '--flat'):
            outs += [data['flatx'], data['flaty']]
        else:
            pass

    return outs


def print_table(
        tabledata: list[list[float]],
        printoutfile: bool = False) -> None:
    """Print out the results from table."""
    target = 'beam.info' if printoutfile else None
    output = target and open(target, 'a') or sys.stdout
    for td in tabledata:
        for dd in td:
            output.write(f'{dd:.4f} ')
        output.write('\n')
    if target is not None:
        output.close()
    return


def print_out(outs: str, printoutfile: bool = False) -> None:
    """Print out data table, if there were selected options."""
    target = 'beam.info' if printoutfile else None
    output = target and open(target, 'a') or sys.stdout
    output.write(f'{outs}\n')
    if printoutfile:
        output.close()


def opening_screen(
        sep: str,
        *header: str,
        totwidth: int = 60,
        bsize: int = 3) -> str:
    """Format opening text screen with information about the program."""
    # Clear current terminal screen before start
    # os.system('clear')

    border = bsize * sep 
    skipline = f'{border}{border:>{totwidth-bsize}}\n'
    tblines = str(3 * '{}\n'.format(totwidth * sep))

    opscreen = f'{tblines}{skipline}'
    for item in header:
        opscreen += f'{border}{item:^{totwidth-2*bsize}}{border}\n{skipline}'

    opscreen += tblines
    return opscreen


#
def splash_screen() -> None:
    """Title and opening screen."""
    title      = 'Beam analysis for Carcara-X beamline'
    subheading = '@ IDS - LNLS - CNPEM - Campinas - Brazil'
    opscreen = opening_screen('*', title, subheading, __author__,
                              __email__, __date__,
                              __copyright__, __version__,
                              __status__)
    print(opscreen)


def header_read(
        header: dict[str, Any] | None = None,
        filedate: str | None = None) -> dict[str, Any]:
    """Read the header of 'datafile' or create a stub.

    Args:
        header: dictionary with header information, if available.
        filedate: string with date of file creation, if available.

    Returns:
        header: dictionary with header information.
    """
    if header is None:
        header = {}
        # Pixel size is set to 1.
        header['PSize_1'] = 1
        header['PSize_2'] = 1
        # Date of file creation.
        header['Date'] = filedate
        # Exposure time
        header['Exptime'] = 1
    return header


def gaussian_fit(
        bdata: LBA,
        xdata: np.ndarray,
        ydata: np.ndarray,
        c0: float,
        ax: str = 'x') -> tuple[
            np.ndarray, np.ndarray, np.ndarray, float, np.ndarray
            ]:
    """Performs a gaussian fit using the BD (LBA) class method.

    c0 is a guess from formerly calculated center.
    """
    # Guess for A0 coefficient.
    a0 = np.max(ydata)
    fwhm = bdata.flatfwhmx if ax == 'x' else bdata.flatfwhmy 
    prm, cov = bdata.gaussian_fit(xdata, ydata, a0, c0, fwhm / 2)

    if prm is not None:
        stddev = np.sqrt(np.diag(cov, 0))   # Std. dev.
        yfit = bdata.gauss(xdata, *prm)        # Fit a gaussian.
        print(f"  {ax}_A0    = {prm[0]:10.4f}   ({stddev[0]:6.3f})\n"
              f"  {ax}_mu    = {prm[1]:10.4f}   ({stddev[1]:6.3f})\n"
              f"  {ax}_sigma = {prm[2]:10.4f}   ({stddev[2]:6.3f})\n")
    else:
        print(" Gaussian fitting aborted,")
        yfit = None
    return prm, cov, yfit, fwhm, stddev


def histograms_fit(bdata: LBA) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Fit a gaussian to x data."""
    # Fit a gaussian to x data.
    print("\n Gaussian fitting for x:")
    res = gaussian_fit(
        bdata, bdata.xrange, bdata.flatx, bdata.Cx, ax='x'
        )
    xprm, xstd, xfit, xsraw, xfwhm = res

    # Fit a gaussian to y data.
    print("\n Gaussian fitting for y:")
    res = gaussian_fit(
        bdata, bdata.yrange, bdata.flaty, bdata.Cy, ax='y'
        )
    yprm, ystd, yfit, ysraw, yfwhm = res

    return ((xprm, xstd, bdata.xrange, xfit, xsraw, xfwhm),
            (yprm, ystd, bdata.yrange, yfit, ysraw, yfwhm))


def histograms_plot(bd: LBA) -> None:
    """Plot the flattened histograms in x and y directions."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # The relocation of axes.
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1)
    ax_histy = divider.append_axes("right", 2.4, pad=0.1)

    ax.invert_yaxis()
    ax.imshow(bd.data)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax_histx.plot(bd.xrange, bd.flatx)
    #
    ax_histy.invert_yaxis()
    ax_histy.plot(bd.flaty, bd.yrange)

    if bd.Gaussfitx['fitcurve'] is not None:
        ax_histx.plot(bd.xrange, bd.Gaussfitx['fitcurve'])

    if bd.Gaussfity['fitcurve'] is not None:
        ax_histy.plot(bd.Gaussfity['fitcurve'], bd.yrange)

    # x_histx.set_yticks([0, 50, 100])
    # x_histy.set_xticks([0, 50, 100])
    # x[1].set_xlabel("x (pixels)")
    # x[1].set_ylabel("counts")
    # x[2].set_xlabel("y (pixels)")
    # x[2].set_ylabel("counts")
    plt.tight_layout(h_pad=1.2)
    plt.show()


def cmd_line() -> tuple[dict[str, Any], list[str]]:
    """Get command line options."""
    cmdopts = {
        'cx'       : None,
        'cy'       : None,
        'px'       : None,
        'py'       : None,
        'xprof'    : None,
        'yprof'    : None,
        'skewx'    : None,
        'skewy'    : None,
        'projfit'  : False,
        'projshow' : False,
        'boxsize'  : 500,
        'otheropts': False,
        'printoutfile': False,
        'quietsplash': False,
        'files'    : [],
    }

    try:
        line = getopt.getopt(
            sys.argv[1:],
            "hdemnrtoqb:",
            list(cmdopts.keys())
            )
    except getopt.GetoptError as err:
        print('\n\n ERROR: ', str(err),'\b.')
        sys.exit(1)
    #
    opts, files = line[0], line[1]

    # Define a default box size.
    for o in opts:
        if (o[0] == '-h'):
            help('beamprofile')
            return
        elif (o[0] == '-b'):
            cmdopts['boxsize'] = int(o[1])
        elif (o[0] == '-o'):
            cmdopts['printoutfile'] = True
        elif (o[0] == '-q'):
            cmdopts['quietsplash']  = True
        elif o[0] == '--projshow':
            cmdopts['projshow']     = True
            cmdopts['projfit']      = True
        elif o[0] == '--projfit':
            cmdopts['projfit']      = True
        else:
            cmdopts['otheropts']    = True

    return cmdopts, files


def main() -> None:
    """Get command line options, open file(s) and calculate info."""
    # Get output option.
    cmdopts, files = cmd_line()

    # Print out a splash screen.
    if not cmdopts.get('quietsplash', False):
        splash_screen()
        print('### Center of mass coordinates, P.O.N.I., and'
              + '\n### other statistical information'
              + ' from beam image.')

    # Check input file.
    if not cmdopts.get('files', False):
        print(' ERROR: no file given.')
        sys.exit(1)

    # Open data file.
    for fl in cmdopts['files']:
        # Read data and header (if available) from file. If it is an
        # EDF file, open it with fabio and get header information,
        # otherwise creates a stub header.
        if re.search("edf", fl, flags=re.IGNORECASE):
            bd      = fabio.open(fl)
            bdata   = bd.data
            bheader = bd.header
        else:
            # If file contains only numpy data.
            filedate = time.ctime(os.path.getctime(fl))
            bdata = np.load(fl)
            bheader = header_read(None, filedate=filedate)

        # Instantiate beam information.
        bd = LBA(bdata, bheader, cmdopts['boxsize'])

        # Set dictionary and strings for output data.
        outdata, outstr = set_output(bd, fl)

        # Initial time is taken from first sample.
        zerotime = outdata['epoch']

        # Print out file information or accumulate required data.
        # Initialize variable for initial time and for final data.
        if cmdopts['otheropts']:
            tabledata = []
            tabledata.append(out_opts(cmdopts, outdata, zerotime))
            # If there were argument options, print data.
            if len(tabledata) != 0:
                print_table(tabledata, cmdopts['printoutfile'])
        else:
            print_out(outstr, cmdopts['printoutfile'])

        # Fit and/or show projectedh istograms.
        if cmdopts['projfit']:
            xres, yres = histograms_fit(bd)

        if cmdopts['projshow']:
            histograms_plot(bd)


if __name__ == '__main__':
    main()
