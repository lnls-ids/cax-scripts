#!/usr/bin/env python3

# Import necessary libraries.

import argparse
import re
import string
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

from scipy.optimize import curve_fit

# os.environ["QT_QPA_PLATFORM"] = "xcb"

PIXEL_SIZE = 0.48   # µm
DVF = 'dvf_B1'


OBSERVABLE_KEYS = [
    ('roix_fit_mean', PIXEL_SIZE),
    ('roiy_fit_mean', PIXEL_SIZE),
    ('roix_fit_sigma', PIXEL_SIZE),
    ('roiy_fit_sigma', PIXEL_SIZE),
    ('roix_fwhm', PIXEL_SIZE),
    ('roiy_fwhm', PIXEL_SIZE),
    ('z_pos', 1.0),
]

# Default initial guesses and maximum function evaluations for caustic fitting.
P0_x   = np.array([ 50.0, 15.0, 150.0])
P0_y   = np.array([100.0, 15.0, 700.0])
MAXFEV = 10000

"""
Functions for blocking snapshots and averaging over each set.
"""

def average_snapshots(
        snapshots  : list,
        observable : str,
        unit       : float = PIXEL_SIZE,
        ) -> tuple:
    """Average observable over snapshots.

    Args:
        snapshots (list) : List of snapshots.
        observable (str) : The key of the observable to average.

    Returns:
        tuple: A tuple containing the average value and the standard deviation of the observable over the snapshots.
    """
    total = []
    for sn in snapshots:
        val = sn[DVF].attrs[observable]
        if isinstance(val, (list, np.ndarray)):
            val = val[0]
        try:
            val = float(val)
        except (TypeError, ValueError):
            print(f"WARNING: Could not convert value '{val}'"
                  f"to float for observable '{observable}'.\n"
                  f" Skipping snapshot '{sn}' for observable '{observable}'.")
            val = 0.0
        total.append(float(val))
    return np.mean(total) * unit, np.std(total) * unit


def block_snapshots(data : "h5py.File") -> list:
    """Separate snapshots in blocks based on step identifiers.

    Args:
        data ("h5py.File") : HDF5 file containing snapshot data, with keys as snapshot names.

    Returns:
        list: A list of blocks, where each block is a list of snapshots corresponding to the same step identifier.
    """
    # Sort data list.
    sdata = sorted(list(data))
    step_last = sdata[0]

    steps = []
    snapshots = []
    for name in sdata:
        step_curr = name.split('-')[1]
        step_last = step_last.split('-')[1]
        if step_curr == step_last:
            snapshots.append(data[name])
        else:
            steps.append(snapshots[:])
            snapshots = [data[name]]
        step_last = name

    nsnaps = int(max([f"{len(st):3d}" for st in steps]))
    # stlengths = [f"{len(st):3d}" for st in steps]
    # print("Steps:  ", ' '.join([f"{ii:3d}" for ii in range(len(steps))]))
    # print("Lengths:", ' '.join(stlengths))
    # print("\n### nsnaps: ", nsnaps, "\n###\n")

    return steps, nsnaps


def calculate_observables(
        steps : list,
        observable_keys : list
        ) -> dict:
    """Calculate observables for each block of snapshots.

    Args:
        steps (list) : List of blocks of snapshots, where each block is a list of snapshots corresponding to the same step identifier.
        observable_keys (list) : List of observable keys to calculate.

    Returns:
        dict: A dictionary where keys are observable names and values are numpy arrays of averaged observables for each block.
    """
    observables = {}
    for key, unit in observable_keys:
        observables[key] = []
        for step in steps[1:]:
            observables[key].append(
                average_snapshots(step, key, unit)
                )
        observables[key] = np.array(observables[key])
    return observables


def caustic_func(
        z   : np.ndarray,
        z0  : float,
        amp : float,
        s0  : float
        ) -> np.ndarray:
    """Calculate caustic function.

    The caustic envelope is characterized as a hyperboloid.

    Args:
        z   (np.ndarray) : Array of propagation positions.
        z0  (float)      : Waist position.
        amp (float)      : Amplitude.
        s0  (float)      : Spread factor.

    Returns:
        np.ndarray: Calculated caustic function values.
    """
    return amp * np.sqrt(1 + ((z - z0) / s0)**2)


def caustic_processing(
        params : list,
        positions : np.ndarray
        ) -> tuple:
    """Process caustic scan results.

    Args:
        params (list) : List of caustic function parameters [z0, amp, s0].
        positions (np.ndarray) : Array of propagation positions [m].

    Returns:
        tuple: waist position z0 [m], depth of focus dof [m], and Rayleigh range zr [m].
    """
    z = np.linspace(positions[0], positions[-1], 2000)
    widths_c_fit = caustic_func(z, *params)

    z0  = params[0]
    dof = np.ptp(z[widths_c_fit <= 1.1 * np.min(widths_c_fit)])/2
    zr  = np.ptp(z[widths_c_fit <= np.sqrt(2) * np.min(widths_c_fit)])/2

    return z0, dof, zr


def observable_set_calc(
        ddir  : str,
        files : list
        ) -> dict:
    """Calculate observables for each file.

    Args:
        ddir (str) : Directory containing the data files.
        files (list) : List of data files to process.

    Returns:
        dict: Dictionary containing observables for each ry value.
    """
    observables = {}
    for cfile in files:
        # Read data set.
        wfile = f"{ddir}/{cfile}"
        cdatum = h5py.File(wfile, "r")

        # 
        print(f"* Processing file: {cfile}")
        steps, nsnaps = block_snapshots(cdatum)
        print(f"  Total steps found: {len(steps)}.\n")
        cdat_scan1 = list(cdatum.keys())[1]
        try:
            ry = cdatum[cdat_scan1].attrs["mirror.ry"][0]
        except KeyError:
            ry = cdatum["scan-0001"].attrs["mirror.ry"][0]

        # Calculate observables for each averaged block of snapshots.
        observables[ry] = (
            calculate_observables(steps, OBSERVABLE_KEYS)
            )
    dwell_time = cdatum[cdat_scan1].attrs["dwell_time"]

    return observables, nsnaps, dwell_time


def hyperbola_fitting(
        observables: dict,
        p0_x       : np.ndarray = P0_x,
        p0_y       : np.ndarray = P0_y,
        maxfev     : int        = MAXFEV,
        obsx       : str        = "roix_fit_sigma",
        obsy       : str        = "roiy_fit_sigma",
        ) -> tuple:
    """Fitting of hyperbolae.
    
    Args:
        observables (dict) : Dictionary containing observables for each ry      
                             value.

        # Parameters for fitting: default is defined by header 
        # general parameters.
        p0_x (np.ndarray)  : Initial guess for x-direction caustic parameters.
        p0_y (np.ndarray)  : Initial guess for y-direction caustic parameters.
        maxfev (int)       : Maximum number of function evaluations for
                             curve fitting.

        # Observables - default is sigma.
        obsx (str)         : Observable key for x-direction fitting.
        obsy (str)         : Observable key for y-direction fitting.

    Returns:
        tuple: Dictionaries of fitted caustic parameters for x and y directions.    
    """
    caustic_x_params = {}
    caustic_y_params = {}


    cxp = ', '.join([f"{v:.4f}" for v in p0_x])
    cyp = ', '.join([f"{v:.4f}" for v in p0_y])
    print(f"Initial guess for x: {cxp}")
    print(f"Initial guess for y: {cyp}\n")

    obsx = "roix_fit_sigma"
    obsy = "roiy_fit_sigma"

    for key, value in observables.items():
        positions = value["z_pos"][:, 0]
        sx = value[obsx][:, 0]
        sy = value[obsy][:, 0]

        # print(f"\n## Fitting for ry = {key:.4f}")
        try:
            caustic_x_params[key], _ = curve_fit(
                caustic_func, positions, sx, p0=p0_x, maxfev=maxfev
                )
            cxp = ', '.join([f"{v:.4f}" for v in caustic_x_params[key]])
            # print(f" * fitted parameters x: {cxp}")
        except RuntimeError as e:
            print(f"Error fitting ry = {key:.4f} for x: {e}")
            caustic_x_params[key] = None

        try:
            caustic_y_params[key], _ = curve_fit(
                caustic_func, positions, sy, p0=p0_y, maxfev=maxfev
                )
            cyp = ', '.join([f"{v:.4f}" for v in caustic_y_params[key]])
            # print(f" * fitted parameters y: {cyp}")
        except RuntimeError as e:
            print(f"Error fitting ry = {key:.4f} for y: {e}")
            caustic_y_params[key] = None

    return caustic_x_params, caustic_y_params


def save_caustic_parameters(
        caustic_x_params : dict,
        caustic_y_params : dict,
        coeffs_x         : tuple = None,
        coeffs_y         : tuple = None,
        dwell_time       : float = 1.0,
        nsnapshots       : int = 1,
        ddir             : str = ".",
        ) -> None:
    """Save caustic fitting parameters to a file."""
    file_basename =(
        "all_passes"
        + f"-snapshots_{nsnapshots:02d}"
        + f"-dwell_{float(dwell_time):2.1f}s"
    )
    filename = f"{ddir}/{file_basename}.txt"

    with open(filename, "w") as f:
        f.write("# Caustic fitting parameters\n\n")
        for key in caustic_x_params:
            f.write(f"#### ry = {key:.4f}\n")
            if caustic_x_params[key] is not None:
                z0, a, s0 = caustic_x_params[key]
                f.write(" x:"
                        f" z0 = {z0:12.4f},"
                        f" A  =  {a:10.4f},"
                        f" s0 = {s0:10.4f}\n"
                        )
            if caustic_y_params[key] is not None:
                z0, a, s0 = caustic_y_params[key]
                f.write(" y:"
                        f" z0 = {z0:12.4f},"
                        f" A  =  {a:10.4f},"
                        f" s0 = {s0:10.4f}\n\n"
                        )

        if coeffs_x is not None:
            f.write("\n# Coefficients for z0 fit, x:"
                    f" a = {coeffs_x[0]:.4f}, b = {coeffs_x[1]:.4f}\n")
        if coeffs_y is not None:
            f.write("# Coefficients for z0 fit, y:"
                    f" a = {coeffs_y[0]:.4f}, b = {coeffs_y[1]:.4f}\n")


def plot_observables(
        observables: dict,
        caustic_x_params: dict,
        caustic_y_params: dict,
        nsnapshots      : int   = 1,
        dwell_time      : float = 1.0,
        obsx            : str   = 'roix_fit_sigma',
        obsy            : str   = 'roiy_fit_sigma',
        raw             : bool  = False,
        fitted          : bool  = False,
        ) -> None:
    """Plot caustic fitting results and observables."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))


    prm  = (
        f"{len(observables)} passes,"
        f" {nsnapshots} snapshot(s)/step,"
        f" dwell = {dwell_time:2.1f} s"
        )

    fig.suptitle(f'Caustic: {prm}')

    nax = 1 if obsx == "roix_fit_sigma" else 2

    rys = sorted(observables.keys())
    colormap = plt.get_cmap("jet")(np.linspace(0, 1, len(rys)))

    for ii, ry in enumerate(rys):
        observable = observables[ry]

        zp = observable['z_pos'][:, 0]

        cx     = observable['roix_fit_mean'][:, 0]
        cx_err = observable['roix_fit_mean'][:, 1]
        sx     = observable[obsx][:, 0]
        sx_err = observable[obsx][:, 1]
        # fx     = observable[obsx][:, 0]
        # fx_err = observable[obsx][:, 1]

        cy     = observable['roiy_fit_mean'][:, 0]
        cy_err = observable['roiy_fit_mean'][:, 1]
        sy     = observable[obsy][:, 0]
        sy_err = observable[obsy][:, 1]
        # fy     = observable[obsy][:, 0]
        # fy_err = observable[obsy][:, 1]

        # Plot centroid.
        axes[0, 0].errorbar(
            zp, cx, yerr=cx_err, label=f'cx (ry = {ry:.4f})',
            color=colormap[ii]
            )
        axes[1, 0].errorbar(
            zp, cy, yerr=cy_err, label=f'cy (ry = {ry:.4f})',
            color=colormap[ii]
            )

        # Plot sigmas, raw.
        if raw:
            axes[0, 1].errorbar(
                zp, sx, yerr=sx_err, label=f'sx (ry = {ry:.4f})',
                color=colormap[ii]
                )
            axes[1, 1].errorbar(
                zp, sy, yerr=sy_err, label=f'sy (ry = {ry:.4f})',
                color=colormap[ii]
                )

        # axes[0, 2].errorbar(
        #     zp, fx, yerr=fx_err, label=f'fx (ry = {ry:.4f})'
        #     )
        # axes[1, 2].errorbar(
        #     zp, fy, yerr=fy_err, label=f'fy (ry = {ry:.4f})'
        #     )

        # Plot sigmas, fitted.
        if fitted:
            if caustic_x_params[ry] is not None:
                sx_fit = caustic_func(zp, *caustic_x_params[ry])
                axes[0, nax].plot(
                    zp, sx_fit, label=f'sx fit (ry = {ry:.4f})',
                    color=colormap[ii]
                    )

            if caustic_y_params[ry] is not None:
                sy_fit = caustic_func(zp, *caustic_y_params[ry])
                axes[1, nax].plot(
                    zp, sy_fit, label=f'sy fit (ry = {ry:.4f})',
                    color=colormap[ii]
                    )

    # Plot fitted z0 parameters.
    rys = sorted(observables.keys())
    z0_x = [
        caustic_x_params[ry][0]
        if caustic_x_params[ry] is not None
        else np.nan
        for ry in rys
        ]
    z0_y = [
        caustic_y_params[ry][0]
        if caustic_y_params[ry] is not None
        else np.nan
        for ry in rys
        ]

    z0_x_fit, coeffs_x = line_fit(np.array(rys), np.array(z0_x))
    z0_y_fit, coeffs_y = line_fit(np.array(rys), np.array(z0_y))
    print("Coefficients for z0 fit, x:",
          f"a = { coeffs_x[0]}, b = {coeffs_x[1]}")
    print("Coefficients for z0 fit, y:",
          f"a = { coeffs_y[0]}, b = {coeffs_y[1]}")
    axes[0, 2].plot(
        rys, z0_x, 'o-', label=f'z0_x (ry = {ry:.4f})',
        )
    axes[0, 2].plot(
        rys, z0_x_fit, 'o-', label='z0_x fit',
        )
    axes[0, 2].plot(
        rys, z0_y, 'o-', label=f'z0_y (ry = {ry:.4f})',
        )
    axes[0, 2].plot(
        rys, z0_y_fit, '^-', label='z0_y fit',
        )

    ounit = r'[$\mu$m]'
    axes[0, 0].set_ylabel(rf'Centroid $x$ {ounit}')
    axes[1, 0].set_ylabel(rf'Centroid $y$ {ounit}')

    axes[0, 1].set_ylabel(rf'$\sigma_x$ {ounit}')
    axes[1, 1].set_ylabel(rf'$\sigma_y$ {ounit}')

    axes[0, 2].set_ylabel('$z_0,x$ [mm]')
    axes[1, 2].set_ylabel('$z_0,y$ [mm]')

    for nax in np.array(np.meshgrid(range(2), range(3))).T.reshape(-1, 2):
        axes[tuple(nax)].legend()
        axes[tuple(nax)].grid(True)
        axes[tuple(nax)].set_xlabel(r'$z$ [mm]')

    axes[0, 2].set_xlabel(r'$R_y$ [mrad]')
    axes[1, 2].set_xlabel(r'$R_y$ [mrad]')

    plt.tight_layout()
    plt.show()
    return fig, (coeffs_x, coeffs_y)


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    """Perform a linear fit to the given data.

    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.

    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    coeffs = np.polyfit(x, y, 1)
    calc_points = np.polyval(coeffs, x)
    return calc_points, coeffs


def save_figure(
        fig        : "plt.Figure",
        wdir       : str = ".",
        nsnapshots : int = 1,
        dwell_time : float = 1.0
        ) -> None:
    """Save the figure to the specified directory with the given filename."""
    figname = (
        "all_passes"
        + f"-snapshots_{nsnapshots:02d}"
        + f"-dwell_{float(dwell_time):2.1f}s"
        + ".png"
        )
    print("Saving figure as: ", figname)
    fig.savefig(f"{wdir}/{figname}", dpi=300)


def cmd_line() -> argparse.Namespace:
    import argparse
    parser = argparse.ArgumentParser(
        description="Process caustic scan results."
        )
    parser.add_argument(
        "-d", "--dir", type=str, required=True,
        help="Data directory"
        )
    parser.add_argument(
        "-p0x", type=str, required=False, help="Initial guess for p0_x."
        )
    parser.add_argument(
        "-p0y", type=str, required=False, help="Initial guess for p0_y."
        )

    group_rf = parser.add_mutually_exclusive_group()
    group_rf.add_argument(
        "-r", action='store_true', required=False,
        help="Plot only raw data."
        )
    group_rf.add_argument(
        "-f", action='store_true', required=False,
        help="Plot only fitted data."
        )

    args = parser.parse_args()

    if not args.f and not args.r:
        args.f = True
        args.r = True

    return args


def _isinteger(s: str) -> bool:
    '''Tests whether a string is a positive integer number.
     
    Returns:
    bool: True if the string represents a positive integer, False otherwise.
    '''
    if (s == ''):
        return False
    for c in s:
        if (c not in (string.digits)):
            return False
    return True


def _read_range (ifls: int, nfls: int) -> list:
    '''Ask interval range.

    Args:
        ifls (int): Initial file number in the list.
        nfls (int): Number of files in the list.

    Returns:
        list: List of chosen file numbers.
    '''
    
    while (True):
        # Read numbers corresponding to file list.
        s = input(' ? > ')
        
        if  (s == ''):
            continue
        elif (s == 'q'):
            sys.exit(0)
            
        try:
            for c in s:
                if (c not in (string.digits + '-' + ' ')):
                    raise Exception (' Please, type only digits '
                                     'and the character \"-\".')   
        except Exception as err:
            print (err)
            continue

        # Test whether chosen files are suitable
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'\s*-\s*', '-', s)
        s = re.sub(r'-+', '-', s)
        ss = s.split()
        ts = []
        
        for rg in ss:
            if (re.search('-', rg)):
                try:
                    ffs = rg.split('-')
                    for num in ffs:
                        if (not _isinteger(num)):
                            raise Exception(' Expression not recognized.')
                        a = int(ffs[0])
                        b = int(ffs[1])
                        if (a > b):
                            raise Exception(' Range limits are out of order.')
                except Exception as err:
                    print (err, 'Please, type again.')
                    break
                fs = [ x for x in range(a, b + 1) ]
            else:
                fs = [ int(rg) ]
            ts = ts + fs

        #
        if (ts == []): continue

        if min(ts) < ifls or max(ts) > nfls:
            print(f' Error: numbers must be within the interval'
                  f' {ifls} to {nfls}.')
            continue
        
        if 0 in ts:
            if not len(ts) == 1:
                print (' Error: "0" option must be chosen alone, '
                       'not within an interval.')
                continue

        return ts


def choose_files(ddir : str) -> list:
    """Prompt the user to choose a file from the list of available files.

    Args:
        ddir (str): Data directory containing the files.

    Returns:
        list: List containing the selected files, or an empty list
        if invalid choice.
    """
    listfiles = sorted([
        f for f in os.listdir(ddir) if f.endswith('.hdf5')
        ])

    print("Avaliable files:")
    for ii, f in enumerate(listfiles):
        print(f"\t[{ii + 1:2d}]: {f}")
    print("\t[ a]: All files")
    print("\t[ q]: Quit")

    choices = input(
        "Enter the number of the files to choose separated by spaces: "
        ).split()
    selected_files = []

    if choices[0].upper() == "Q":
        print("Quitting.")
        sys.exit(0)

    if choices[0].upper() == "A":
        return listfiles
 
    for choice in choices:
        idx = int(choice) - 1
        if 0 <= idx < len(listfiles):
            selected_files.append(listfiles[idx])
        else:
            print(f"Invalid choice: {choice}")
    return selected_files


def main() -> None:
    """Main function to process caustic scan results."""
    # Get directory containing the data files.
    args = cmd_line() 
    if args.dir is None:
        print("Data directory is required.")
        return

    # Select data files.
    files = choose_files(args.dir)
    file_list = '\n * '.join(files)
    print(f"\n Chosen data file:\n * {file_list}\n")

    # Calculate observables.
    observables, nsnapshots, dwell_time = observable_set_calc(args.dir, files)
    print(f"Number of snapshots per step: {nsnapshots}\n"
          f"Dwell time between snapshots: {dwell_time}")

    # Fit hyperbolic caustics.
    p0_x = (
        np.array([float(x) for x in args.p0x.split(',')])
        if args.p0x is not None
        else P0_x
        )
    p0_y = (
        np.array([float(y) for y in args.p0y.split(',')])
        if args.p0y is not None
        else P0_y
        )
    caustic_x_params, caustic_y_params = hyperbola_fitting(
        observables, p0_x, p0_y
        )

    # Plot graphs.
    fig, (coeffs_x, coeffs_y) = plot_observables(
        observables,
        caustic_x_params,
        caustic_y_params,
        nsnapshots = nsnapshots,
        dwell_time = dwell_time,
        raw        = args.r,
        fitted     = args.f,
        )

    # Save results.
    save_caustic_parameters(
        caustic_x_params,
        caustic_y_params,
        coeffs_x,
        coeffs_y,
        dwell_time,
        nsnapshots,
        )
    save_figure(fig, nsnapshots=nsnapshots, dwell_time=dwell_time)


if __name__ == "__main__":
    main()