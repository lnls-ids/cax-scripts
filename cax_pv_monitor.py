#!/opt/mamba_files/mamba/envs/sirius/bin/python
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

def _parse_dates(dt):
    """Parse dates from date string."""
    year, month, day = dt.year, dt.month, dt.day
    hour, minute = dt.hour, dt.minute
    return [year, month, day, hour, minute]


def get_pvdata(pvnames, initdate, enddate, timeout):
    """Now ruff doesn't bother me."""
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
        fn = f"{wdir}/CAX_{pvname}_{lasttime}.txt"
        print(f" {fn}", end=", ")
        filenames.append(fn)
        np.savetxt(fn, pvtimeval, fmt="%.2f")
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

        p0 = (60, 75, 5.0)
        try:
            prm[idx], cov[idx] = curve_fit(negexp, times, fluxes, p0=p0)
        except Exception as err:
            print(f" WARNING: when trying to fit exponential: {err}\n"
                  " Curve will be drawn with standard values, just as"
                  " a reference.")
            prm[idx] = p0
            cov[idx] = np.diag((1, 1))
    return data, prm, cov


def plot_data(data, prm, pvnames):
    """Plot data and fittings."""
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
        ax[idx].plot(times[::25], fluxes[::25], st[idx], label="data")
        ax[idx].plot(times, fluxfit, 'y-',
                    label=f"{c0:.2f} + {a0:.2f} exp(-t/{tau:.2f})")

        exttime = np.arange(0, 50)
        fluxfit = negexp(exttime, *prm[idx])
        ax[idx].plot(exttime, fluxfit, 'r-', label="extended fit")
        ax[idx].set_xlabel("days")
        ax[idx].set_ylabel("flux [mL / min]")
        ax[idx].set_ylim(0, 150)
        ax[idx].set_title(title)
        ax[idx].legend()
        ax[idx].grid()

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

    # Define PVs.
    pvnames = PVFLUX
    # for pvnames in [PVFLUX]:
    for pvnames in [PVFLUX, PVTEMP, PVPRESS]:
        pvs_data, _, _ = get_pvdata(pvnames, idt, edt, timeout=30)
        pv = {pv: pvs_data[pv] for pv in pvnames}

    # Write out data to files.
    filenames = data_output(pv, wdir)

    # Fit a decaying exponential to each PV data.
    data, prm, cov = exponential_fit(filenames)
    for idx in [0, 1]:
        print(f"\n  >>>>> Results from Fitting C + A x exp(-t/tau) "
            f"for PV {idx + 1}:\n C = {prm[idx][0]:.4f}, "
            f"A = {prm[idx][1]:.4f}, tau = {prm[idx][2]:.4f}"
            f"\n (half life = {prm[idx][2] * np.log(2):.2f})\n"
            f"\n covariance matrix =\n{cov[idx]}\n")

    if args.plot_graph:
        plot_data(data, prm, pvnames)


if __name__ == "__main__":
    main()
