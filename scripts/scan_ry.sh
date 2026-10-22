#!/bin/bash

# Set environment variables for EPICS and Python libraries.
export EPICS_PVA_ADDR_LIST="10.0.38.59:62000 10.30.13.22 10.30.14.19 10.39.50.46"
export EPICS_CA_ADDR_LIST="10.0.38.59:62000 10.30.13.22 10.30.14.19 10.39.50.46"
export PYDM_EPICS_LIB=PYEPICS
export EPICS_HOST_ARCH=linux-x86_64
export EPICS_CA_MAX_ARRAY_BYTES=100000000

# Demands that Python output is unbuffered for real-time logging.
export PYTHONUNBUFFERED=1

# Activate the micromamba environment for Sirius.
if [[ "${CONDA_DEFAULT_ENV:-}" == "sirius" ]]; then
    echo "Running inside an activated environment: $CONDA_PREFIX"
else
    echo "No activated environment detected."
    echo "Activate the 'sirius' environment before running this script."
    exit 1
fi

# Display usage information for the script.
help ()
{
    echo -e "Usage:\n scan_ry.sh <ry_min> <ry_max> <n_steps>"
    echo "  ry_min   : Minimum value of r_y"
    echo "  ry_max   : Maximum value of r_y"
    echo "  n_steps  : Number of steps between ry_min and ry_max"
}

# Function to set the r_y value of the mirror.
set_ry()
{
    ry=$1
    echo "Setting r_y to $ry ..."
    env python3 - "$ry" <<'EOF'
import sys
from siriuspy.devices import CAXCtrl
cax = CAXCtrl()
ry = float(sys.argv[1])
cax.mirror.ry = ry
if abs(cax.mirror.ry - ry) > 1e-3:
    print(f"Failed to set r_y to {ry}")
    sys.exit(1)
EOF
}

# Parse and validate command-line arguments.
args=$@
if [ "$#" -ne 3 ]; then
    help
    exit 1
fi

ry_min=$1
ry_max=$2
n_steps=$3

if (( $(echo "$ry_min >= $ry_max" | bc -l) )); then
    echo "Error: ry_min must be less than ry_max."
    exit 1
fi

echo "Starting scan with the following parameters:"
echo "  ry_min   : $ry_min"
echo "  ry_max   : $ry_max"
echo "  n_steps  : $n_steps"
echo "Scan started..."

parameter_file="caustic_parameters"


# Loop over the steps and perform the scan.
for ((ii=0; ii<=n_steps; ii++))
do
    # Calculate the current r_y value.
    ry=$(echo "$ry_min + ($ry_max - $ry_min) * $ii / $n_steps" | bc -l)
    echo -e "\n ### Step $(($ii + 1)): ry = $ry"
    set_ry $ry

    # Check if the command was successful.
    if [ $? -ne 0 ]; then
        echo "Error: Failed to set r_y to $ry"
        exit 1
    fi

    # Wait for the system to stabilize before running the caustic scan.
    echo "Waiting for the system to stabilize..."
    sleep 10
    echo "Done. Start scanning..."

    # Update the output file name.
    p=$(($ii % 2))
    if [ $p -eq 0 ]; then
        sed -i "s/_pass.*/_pass$ii/" caustic_parameters

        # Run the caustic scan.
        caxmirrorscan-cli -p caustic_parameters
    else
        sed -i "s/_pass.*/_pass$ii/" caustic_parameters

        # Run the caustic scan.
        caxmirrorscan-cli -p caustic_parameters_back
    fi

    echo "Step $(($ii + 1)) completed."
done

echo "Scan completed."
