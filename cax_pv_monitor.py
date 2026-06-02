#!/usr/bin/env python
"""Fetch data from Carcara's water flux PVs, fit exponentials and plot them."""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
# import re
import os
from scipy.optimize import curve_fit
from siriuspy.clientarch import Time, PVDataSet
from caxscripts.config import Config
# import sys
import argparse

cfg = Config()

PVFLUX  = cfg.PVFLUX
PVTEMP  = cfg.PVTEMP
PVPRESS = cfg.PVPRESS
PVSR    = cfg.SRPV['Storage ring current']


def _parse_dates(dt):
    """Parse dates from date string."""
    year, month, day = dt.year, dt.month, dt.day
    hour, minute = dt.hour, dt.minute
    return [year, month, day, hour, minute]


def get_pvdata(pvnames, initdate, enddate, timeout):
    """Now ruff doesn't bother me."""
    if isinstance(pvnames, str):
        pvnames = [pvnames]

    idt = _parse_dates(initdate)
    edt = _parse_dates(enddate)

    pvs_data = PVDataSet(pvnames)
    pvs_data.timeout = timeout
    pvs_data.time_start = Time(*idt, 0)
    pvs_data.time_stop  = Time(*edt, 0)
    pvs_data.update(mean_sec=60)

    t0 = pvs_data[pvnames[0]].timestamp[0]
    return pvs_data, t0, pvs_data[pvnames[0]].timestamp


def data_output(pvs, wdir):
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


def negexp(t, *p0):
    """Exponential model."""
    c0, a0, tau = p0
    return c0 + a0 * np.exp(-t / tau)


def time_rescale(times):
    """Rescale time interval."""
    first = datetime.fromtimestamp(times[0])
    last = datetime.fromtimestamp(times[-1])
    dt = last - first
    lapse = dt.days + dt.seconds / (24*60*60)
    return np.linspace(0, lapse, len(times))


def exponential_fit(filenames):
    """Fit a decaying exponential to the flux data."""
    prm, cov = list(range(2)), list(range(2))
    data = list()
    for idx, pv in enumerate(filenames):
        print(f"\n##### Reading data from {pv}...", end="")
        data.append(np.genfromtxt(pv))
        times, fluxes = data[idx][:, 0], data[idx][:, 1]
        print(" done.")

        # Rescale time.
        times = time_rescale(times)

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
    data, prm, pvnames, ddays=30, sr_data=None, sr_name="SR current [mA]"
):
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
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    st = ['b-', 'g-']

    for idx in range(2):
        times, fluxes = data[idx][:, 0], data[idx][:, 1]
        timeinterval = (f"from {datetime.fromtimestamp(times[0])} to "
                        f"{datetime.fromtimestamp(times[-1])}")
        times = time_rescale(times)

        fluxfit = negexp(times, *prm[idx])
        c0, a0, tau = prm[idx]

        title = f"{pvnames[idx]}: ({timeinterval})"
        # Take sample points from data.
        ninterval = int(min(len(times) / 25, 25))
        times_data  = times[::ninterval]
        fluxes_data = fluxes[::ninterval]
        ax[idx].plot(times_data, fluxes_data, st[idx], label="data")
        ax[idx].plot(times, fluxfit, 'y-',
                    label=f"{c0:.2f} + {a0:.2f} exp(-t/{tau:.2f})")

        exttime = np.arange(0, ddays, 1/ninterval)
        fluxfit = negexp(exttime, *prm[idx])
        ax[idx].plot(exttime, fluxfit, 'r-', label="extended fit")
        ax[idx].set_xlabel("days")
        ax[idx].set_ylabel("flux [mL / min]")
        ylim = max(fluxes_data) * 1.2
        ax[idx].set_ylim(0, ylim)
        ax[idx].set_title(title)
        ax[idx].legend(loc="upper left")

        if sr_data is not None:
            sr_times = sr_data[:, 0]
            sr_vals = sr_data[:, 1]
            sr_tdays = time_rescale(sr_times)
            sr_step = max(1, int(min(len(sr_tdays) / 50, 50)))
            axr = ax[idx].twinx()
            axr.plot(
                sr_tdays[::sr_step], sr_vals[::sr_step],
                'k--', alpha=0.6, label="SR current"
            )
            axr.set_ylabel(sr_name)
            axr.legend(loc="upper right")

        ax[idx].grid()

    plt.tight_layout()
    plt.show()


def plot_simple(
    data, pvnames, yunit="value", logy=False,
    sr_data=None, sr_name="SR current [mA]"
):
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
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(16, max(4, 3.2 * nrows))
    )
    axes = np.atleast_1d(axes).ravel()

    for idx in range(nplots):
        times = data[idx][:, 0]
        vals = data[idx][:, 1]

        tdays = time_rescale(times)
        ninterval = max(1, int(min(len(tdays) / 50, 50)))

        ax = axes[idx]
        ax.plot(
            tdays[::ninterval], vals[::ninterval], 'b-', label=pvnames[idx]
        )

        t0 = datetime.fromtimestamp(times[0])
        t1 = datetime.fromtimestamp(times[-1])
        ax.set_title(f"{pvnames[idx]}: (from {t0} to {t1})")
        ax.set_xlabel("days")
        ax.set_ylabel(yunit)
        if logy:
            ax.set_yscale("log", nonpositive="clip")
        if sr_data is not None:
            sr_times = sr_data[:, 0]
            sr_vals = sr_data[:, 1]
            sr_tdays = time_rescale(sr_times)
            sr_step = max(1, int(min(len(sr_tdays) / 50, 50)))
            axr = ax.twinx()
            axr.plot(
                sr_tdays[::sr_step], sr_vals[::sr_step],
                'k--', alpha=0.6, label="SR current"
            )
            axr.set_ylabel(sr_name)
            axr.legend(loc="upper right")
        ax.grid()
        ax.legend(loc="upper left")

    for idx in range(nplots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read relevant PV data from Carcara\'s EPICS archiver.'
        )
    parser.add_argument(
        '-f', '--flux', action='store_true',
        help="Fetch data from Carcara\'s flux PVs (mirror and mask), "
            " fit a decaying exponential and plot them."
            )

    parser.add_argument(
        '-t', '--temperature', default=False, action='store_true',
        help=("Fetch data from Carcara\'s temperature PVs and plot them.")
              )

    parser.add_argument(
        '-p', '--pressure', default=False, action='store_true',
        help=("Fetch data from Carcara\'s pressure PVs and plot them.")
              )

    parser.add_argument(
        '-i', '--init_date', type=str, required=True,
        help=("Initial date in ISO format (YYYY-MM-DD [HH:MM]),"
              " without seconds.")
    )

    parser.add_argument(
        '-e', '--end_date', type=str, required=True,
        help=("End date in ISO format (YYYY-MM-DD [HH:MM]),"
              " without seconds.")
    )

    parser.add_argument(
        '-g', '--plot-graph', default=False, action='store_true',
        help="Plot graph of the data and fittings. (default: False)"
    )

    parser.add_argument(
        '-x', '--extend-days', type=int, default=30,
        help="Interval to extend the exponential fit in days. (default: 30)"
    )

    parser.add_argument(
        '-d', '--directory', type=str, default="./logs/",
        help="Directory to write output data files. (default: ./logs/)"
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
        args.pressure is False):
        parser.error("At least one of the options -f, -t or -p must be set.")

    # Check whether working directory exists.
    if not os.path.isdir(args.directory):
        parser.error(f"Directory {args.directory} does not exist.")

    return args


def main():
    """Main function."""
    # Read command line arguments.
    args = cmd_args()

    # Working directory to write output data files.
    wdir = args.directory

    # Initial and end dates in iso format (no seconds).
    idt, edt = args.init_date, args.end_date

    # Define PV groups to fetch.
    pvnameslist = list()

    if args.flux:
        pvnameslist.append(PVFLUX)
    if args.temperature:
        pvnameslist.append(PVTEMP)
    if args.pressure:
        pvnameslist.append(PVPRESS)

    # Write out data to files and plot them if needed.
    for pvnames in pvnameslist:
        pvnames_fetch = pvnames + [PVSR]
        pvs_data, _, _ = get_pvdata(pvnames_fetch, idt, edt, timeout=30)

        # Print storage ring current as reference for each queried group.
        sr_vals = pvs_data[PVSR].value
        print(
            f"\n >>>>> SR current [{PVSR}] [mA]: "
            f"min={np.nanmin(sr_vals):.3f}, "
            f"mean={np.nanmean(sr_vals):.3f}, "
            f"max={np.nanmax(sr_vals):.3f}"
        )

        pv = {pv: pvs_data[pv] for pv in pvnames_fetch}

        # Write out data to files.
        filenames = data_output(pv, wdir)
        # pv_to_file = dict(zip(pvnames_fetch, filenames, strict=True))
        pv_to_file = dict(zip(pvnames_fetch, filenames))
        grp_files = [pv_to_file[pv] for pv in pvnames]
        sr_file = pv_to_file[PVSR]
        sr_data = np.genfromtxt(sr_file)

        # Flux case: fit and optionally plot with fit curves.
        if pvnames == PVFLUX:
            data, prm, cov = exponential_fit(grp_files)
            for idx in [0, 1]:
                print(f"\n  >>>>> Results from Fitting C + A x exp(-t/tau) "
                    f"for PV {idx + 1}:\n C = {prm[idx][0]:.4f}, "
                    f"A = {prm[idx][1]:.4f}, tau = {prm[idx][2]:.4f}"
                    f"\n (half life = {prm[idx][2] * np.log(2):.2f})\n"
                    f"\n covariance matrix =\n{cov[idx]}\n")

            if args.plot_graph:
                plot_data(
                    data, prm, pvnames, args.extend_days,
                    sr_data=sr_data, sr_name="SR current [mA]"
                )

        # Temperature and pressure: simple plot only.
        else:
            if args.plot_graph:
                data = [np.genfromtxt(fname) for fname in grp_files]

                if pvnames == PVTEMP:
                    yunit, logy = "temperature", False
                else:
                    yunit, logy = "pressure [mbar]", True

                plot_simple(
                    data, pvnames, yunit=yunit, logy=logy,
                    sr_data=sr_data, sr_name="SR current [mA]"
                )


if __name__ == "__main__":
    main()
