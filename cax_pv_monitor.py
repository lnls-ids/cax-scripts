#!/usr/bin/env python
"""Fetch data from Carcara's environment PVs and plot them.

PVs relative to water flux, temperature, pressure and FWHM are used to
monitor Carcara state. Fittings are applied when suitable.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from datetime import datetime
from typing import cast
from scipy.optimize import curve_fit

# Warnings filter to ignore deprecation.
# Python version is waiting for upgrade in siriuspy.
# Use a regex match to target this exact deprecation warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pkg_resources is deprecated as an API.*"
    )

from caxscripts.config import Config                 # noqa: E402
from siriuspy.clientarch import Time, PVDataSet      # noqa: E402

# Carcara PV prefixes.
cfg = Config()

PVFLUX  = cfg.PVFLUX
PVTEMP  = cfg.PVTEMP
PVPRESS = cfg.PVPRESS
PVSR    = cfg.SRPV['Storage ring current']
PVFWHM  = cfg.PVFWHM

DEFAULT_UPDATE_INTERVAL = 60     # [s]
DEFAULT_EXTEND_DAYS     = 3      # [days]
DO_RESCALE_TIME         = False


def _parse_dates(dt: datetime) -> list:
    """Parse dates from date string."""
    year, month, day = dt.year, dt.month, dt.day
    hour, minute = dt.hour, dt.minute
    return [year, month, day, hour, minute]


def get_pvdata(
        pvnames: list,
        initdate: datetime,
        enddate: datetime,
        timeout: int,
        update: float = DEFAULT_UPDATE_INTERVAL
        ) -> tuple:
    """Now ruff doesn't bother me."""
    if isinstance(pvnames, str):
        pvnames = [pvnames]

    idt = _parse_dates(initdate)
    edt = _parse_dates(enddate)

    pvs_data = PVDataSet(pvnames)
    pvs_data.timeout = timeout
    pvs_data.time_start = Time(*idt, 0)
    pvs_data.time_stop  = Time(*edt, 0)
    pvs_data.update(mean_sec=update)

    t0 = pvs_data[pvnames[0]].timestamp[0]
    return pvs_data, t0, pvs_data[pvnames[0]].timestamp


def data_output(pvs: dict, wdir: str) -> list:
    """Write data to file."""
    print("\n >>>>> Writing data to files...")

    filenames = list()
    for pvname, pv in pvs.items():
        t0 = 0   # pv.timestamp[0]
        pvtimeval = np.dstack((pv.timestamp - t0, pv.value))[0]
        lasttime = datetime.fromtimestamp(pv.timestamp[-1])
        lasttime = datetime.strftime(lasttime, "%Y%m%d_%H%M%S")
        fn = f"{wdir}/{pvname}_{lasttime}.txt"
        print(f" {fn}", end=", ")
        filenames.append(fn)
        np.savetxt(fn, pvtimeval, fmt=("%.2f", "%.8e"))
    print(" done.")
    return filenames


def negexp(t: np.ndarray, *p0: float) -> np.ndarray:
    """Exponential model."""
    c0, a0, tau = p0
    return c0 + a0 * np.exp(-t / tau)


def time_scale(
        times: np.ndarray,
        rescale: bool = False
        ) -> np.ndarray:
    """Rescale time interval."""
    if not rescale:
        return times

    first = datetime.fromtimestamp(times[0])
    last  = datetime.fromtimestamp(times[-1])
    dt    = last - first
    lapse = dt.days + dt.seconds / (24*60*60)
    return np.linspace(0, lapse, len(times))


def exponential_fit(filenames: list) -> tuple:
    """Fit a decaying exponential to the flux data."""
    prm: list = list(range(2))
    cov: list = list(range(2))
    data = list()
    for idx, pv in enumerate(filenames):
        print(f"\n##### Reading data from {pv}...", end="")
        data.append(np.genfromtxt(pv))
        times, fluxes = data[idx][:, 0], data[idx][:, 1]
        print(" done.")

        # Rescale time.
        times = time_scale(times, rescale=True)

        p0 = (100, 10, 1.0)
        try:
            prm[idx], cov[idx] = curve_fit(negexp, times, fluxes, p0=p0)
        except Exception as err:
            print(f" WARNING: when trying to fit exponential: {err}\n"
                  " Curve will be drawn with standard values, just as"
                  " a reference.")
            prm[idx] = p0
            cov[idx] = np.diag((1, 1))
    return data, prm, cov


def plot_data(
        data: list,
        prm: list,
        pvnames: list,
        ddays=30,
        sr_data=None,
        sr_name="SR current [mA]"
        ) -> None:
    """Plot data and fittings.

    Args:
        data    (list) : List of data arrays (flux vs. time).
        prm     (list) : List of parameters for the exponential fit.
        pvnames (list) : List of PV names.
        ddays   (int)  : Number of days to extend the exponential fit.
                 (default: 30)
        sr_data (array): SR current data array (time vs. value).
        sr_name (str)  : SR current axis label.
    """
    fig, ax_raw = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
    ax = cast(np.ndarray, ax_raw)   # Avoids lint useless complaints.
    st = ['b-', 'g-']

    ax0, ax1 = ax[0], ax[1]
    t0 = data[0][0, 0]
    times = data[0][:, 0]
    tf = t0 + times[-1]
    times = time_scale(times, rescale=True)
    ninterval = int(min(len(times) / 25, 25))
    times_data  = times[::ninterval]
    timeinterval = (f"from {datetime.fromtimestamp(t0)} to "
                    f"{datetime.fromtimestamp(tf)}")

    for idx in range(2):
        fluxes = data[idx][:, 1]

        fluxfit = negexp(times, *prm[idx])
        c0, a0, tau = prm[idx]

        title = f"{pvnames[idx]}: ({timeinterval})"
        # Take sample points from data.
        fluxes_data = fluxes[::ninterval]
        ax0[idx].plot(times_data, fluxes_data, st[idx], label="data")
        ax0[idx].plot(times, fluxfit, 'y-',
                    label=f"{c0:.2f} + {a0:.2f} exp(-t/{tau:.2f})")

        exttime = np.arange(0, ddays, 1/ninterval)
        fluxfit = negexp(exttime, *prm[idx])
        ax0[idx].plot(exttime, fluxfit, 'r-', label="extended fit")
        ax0[idx].set_xlabel("days")
        ax0[idx].set_ylabel("flux [mL / min]")
        # ylim = max(fluxes_data) * 1.2
        # ax0[idx].set_ylim(ylim)
        ax0[idx].set_title(title)
        ax0[idx].legend(loc="upper right")
        ax0[idx].grid()

    axr = ax1[0]  # .twinx()
    if sr_data is not None:
        sr_vals  = sr_data[:, 1]
        # sr_step  = max(1, int(min(len(times_data) / 50, 50)))
        axr.plot(
            times_data,
            sr_vals[::ninterval],
            'k--', alpha=0.6, label="SR current"
        )
        axr.set_ylabel(sr_name)
        axr.legend(loc="upper right")
        axr.grid(True)
    else:
        axr.set_visible(False)

    ax1[1].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_simple(
        data: list,
        pvnames: list,
        yunit="value",
        logy=False,
        sr_data=None,
        sr_name="SR current [mA]"
        ) -> None:
    """Plot PV data without fitting.

    Args:
        data    (list): List of data arrays (time vs. value).
        pvnames (list): List of PV names.
        yunit   (str) : Label/unit for y axis.
        logy   (bool): Use logarithmic scale in y axis.
        sr_data (array): SR current data array (time vs. value).
        sr_name (str)  : SR current axis label.
    """
    nplots = len(data)
    ncols = 2
    nrows = int(np.ceil(nplots / ncols)) + 1
    fig, axes_raw = plt.subplots(
        nrows, ncols, figsize=(16, max(4, 3.2 * nrows))
    )
    axes = cast(np.ndarray, axes_raw)   # Avoids lint useless complaints.
    axes = np.atleast_1d(axes).ravel()

    # Define time interval and x axis label.
    times = data[0][:, 0]
    t0 = datetime.fromtimestamp(times[0])
    tf = datetime.fromtimestamp(times[-1])
    time_interval = time_scale(times, rescale=DO_RESCALE_TIME)
    if DO_RESCALE_TIME:
        ninterval = max(1, int(min(len(time_interval) / 50, 50)))
        time_interval = time_interval[::ninterval]
        xlabel = "days"
    else:
        ninterval = 1
        xlabel = "time [s]"

    for idx in range(nplots):
        vals = data[idx][:, 1]
        ax   = axes[idx]
        ax.plot(
            time_interval,
            vals[::ninterval],
            'b-', label=pvnames[idx]
            )

        # t0 = datetime.fromtimestamp(times[0])
        # t1 = datetime.fromtimestamp(times[-1])
        ax.set_title(f"{pvnames[idx]}: (from {t0} to {tf})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(yunit)
        if logy:
            ax.set_yscale("log", nonpositive="clip")
        ax.grid()
        ax.legend(loc="upper left")

    axr = axes[nplots]  # .twinx()
    if sr_data is not None:
        sr_times = sr_data[:, 0]
        sr_vals  = sr_data[:, 1]
        sr_tdays = time_scale(sr_times)
        sr_step  = max(1, int(min(len(sr_tdays) / 50, 50)))
        # axr      = ax.twinx()
        axr.plot(
            sr_tdays[::sr_step],
            sr_vals[::sr_step],
            'k--', alpha=0.6, label="SR current"
        )
        axr.set_xlabel(xlabel)
        axr.set_ylabel(sr_name)
        axr.legend(loc="upper right")
        axr.grid(True)

    for idx in range(nplots + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def cmd_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read relevant PV data from Carcara\'s EPICS archiver.'
        )

    parser.add_argument(
        '-f', '--flux',
        action='store_true',
        help="Fetch data from Carcara\'s flux PVs (mirror and mask), "
            " fit a decaying exponential and plot them."
            )

    parser.add_argument(
        '-t', '--temperature',
        default=False,
        action='store_true',
        help=("Fetch data from Carcara\'s temperature PVs and plot them.")
              )

    parser.add_argument(
        '-p', '--pressure',
        default=False,
        action='store_true',
        help=("Fetch data from Carcara\'s pressure PVs and plot them.")
              )

    parser.add_argument(
        '-i', '--init_date',
        type=str,
        required=True,
        help=("Initial date in ISO format (YYYY-MM-DD [HH:MM]),"
              " without seconds.")
    )

    parser.add_argument(
        '-e', '--end_date',
        type=str,
        required=True,
        help=("End date in ISO format (YYYY-MM-DD [HH:MM]),"
              " without seconds.")
    )

    parser.add_argument(
        '-g', '--plot-graph',
        default=False,
        action='store_true',
        help="Plot graph of the data and fittings. (default: False)"
    )

    parser.add_argument(
        '-x', '--extend-days',
        type=int,
        default=DEFAULT_EXTEND_DAYS,
        help=("Interval to extend the exponential fit in days."
              f" (default: {DEFAULT_EXTEND_DAYS})")
    )

    parser.add_argument(
        '-d', '--directory',
        type=str,
        default="./logs/",
        help="Directory to write output data files. (default: ./logs/)"
    )

    parser.add_argument(
        '-u', '--update-interval',
        type=float,
        default=DEFAULT_UPDATE_INTERVAL,
        help=("Update interval for fetching PV data in seconds."
              f" (default: {DEFAULT_UPDATE_INTERVAL})")
    )

    parser.add_argument(
        '-w', '--fwhm',
        default=False,
        action='store_true',
        help="FWHM x and y values."
    )

    args = parser.parse_args()

    # Rearrange date order if needed.
    date1, date2 = args.init_date, args.end_date
    d1 = datetime.fromisoformat(date1)
    d2 = datetime.fromisoformat(date2)
    args.init_date, args.end_date = min(d1, d2), max(d1, d2)

    # Check that at least one of the options -f, -t or -p is set.
    if (args.flux is False and
        args.temperature is False and
        args.pressure is False and
        args.fwhm is False):
        parser.error(" At least one of the options"
                     " -f, -t, -p or -w must be set.")

    # Check whether working directory exists.
    if not os.path.isdir(args.directory):
        parser.error(f"Directory {args.directory} does not exist.")

    return args


def pv_name_list(args) -> list:
    """Return list of PVs to fetch based on command line arguments."""
    pvnameslist = list()

    if args.flux:
        pvnameslist.append(PVFLUX)
    if args.temperature:
        pvnameslist.append(PVTEMP)
    if args.pressure:
        pvnameslist.append(PVPRESS)
    if args.fwhm:
        pvnameslist.append(PVFWHM)

    return pvnameslist


def main() -> None:
    """Main function."""
    # Read command line arguments.
    args = cmd_args()

    # Working directory to write output data files.
    wdir = args.directory

    # Initial and end dates in iso format (no seconds).
    idt, edt = args.init_date, args.end_date

    # Define PV groups to fetch.
    pvnameslist = pv_name_list(args)

    # Write out data to files and plot them if needed.
    for pvname in pvnameslist:
        pvnames_fetch = pvname + [PVSR]
        pv_data, _, _ = get_pvdata(
            pvnames_fetch,
            idt, edt,
            timeout=30,
            update=args.update_interval
            )

        # Print storage ring current as reference for each queried group.
        sr_vals = pv_data[PVSR].value
        print(
            f"\n >>>>> SR current [{PVSR}]: "
            f"\n\tmin  = {np.nanmin(sr_vals):6.3f} mA"
            f"\n\tmean = {np.nanmean(sr_vals):6.3f} mA"
            f"\n\tmax  = {np.nanmax(sr_vals):6.3f} mA"
            "\n\n >>>>> Number of samples:"
            f" {len(pv_data[pvname[0]].timestamp)}"
            )

        # Get PV data for the queried group.
        pvs = {pv: pv_data[pv] for pv in pvnames_fetch}

        # Write out data to files.
        filenames = data_output(pvs, wdir)
        # pv_to_file = dict(zip(pvnames_fetch, filenames, strict=True))
        pv_to_file = dict(zip(pvnames_fetch, filenames))  # noqa: B905
        grp_files  = [pv_to_file[pv] for pv in pvname]
        sr_file    = pv_to_file[PVSR]
        sr_data    = np.genfromtxt(sr_file)

        # Flux case: fit and optionally plot with fit curves.
        if pvname == PVFLUX:
            data, prm, cov = exponential_fit(grp_files)
            for idx in [0, 1]:
                print(f"\n  >>>>> Results from Fitting C + A x exp(-t/tau) "
                    f"for PV {idx + 1}:\n C = {prm[idx][0]:.4f}, "
                    f"A = {prm[idx][1]:.4f}, tau = {prm[idx][2]:.4f}"
                    f"\n (half life = {prm[idx][2] * np.log(2):.2f})\n"
                    f"\n covariance matrix =\n{cov[idx]}\n")

            if args.plot_graph:
                plot_data(
                    data, prm, pvname,
                    args.extend_days,
                    sr_data=sr_data,
                    sr_name="SR current [mA]"
                )

        # Temperature and pressure: simple plot only.
        else:
            yunit, logy = "Unknown", False
            if args.plot_graph:
                data = [np.genfromtxt(fname) for fname in grp_files]

                if pvname == PVTEMP:
                    yunit, logy = "temperature [ºC]", False
                elif pvname == PVPRESS:
                    yunit, logy = "pressure [mbar]", True
                elif pvname == PVFWHM:
                    yunit, logy = "FWHM [px]", False

                plot_simple(
                    data, pvname, yunit=yunit, logy=logy,
                    sr_data=sr_data, sr_name="SR current [mA]"
                )


if __name__ == "__main__":
    main()
