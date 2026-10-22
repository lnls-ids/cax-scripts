#!/bin/bash

# Set environment variables for EPICS and Python libraries.
export EPICS_PVA_ADDR_LIST="10.0.38.59:62000 10.30.13.22 10.30.14.19 10.39.50.46"
export EPICS_CA_ADDR_LIST="10.0.38.59:62000 10.30.13.22 10.30.14.19 10.39.50.46"
export PYDM_EPICS_LIB=PYEPICS
export EPICS_HOST_ARCH=linux-x86_64
export EPICS_CA_MAX_ARRAY_BYTES=100000000

# Activate the micromamba environment for Sirius.
micromamba activate sirius

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

echo "Starting scan with the following parameters:"
echo "  ry_min   : $ry_min"
echo "  ry_max   : $ry_max"
echo "  n_steps  : $n_steps"
echo "Scan started..."

# Loop over the steps and perform the scan.
for ((ii=0; ii<=n_steps; ii++))
do
    # Calculate the current r_y value.
    ry=$(echo "$ry_min + ($ry_max - $ry_min) * $ii / $n_steps" | bc -l)
    echo -e "\n ### Step $ii: ry = $ry"
    set_ry $ry

    # Wait for the system to stabilize before running the caustic scan.
    sleep 10 

    # Update the output file name.
    p=$(($ii % 2))
    if [ $p -eq 1 ]; then
        sed -i "s/_pass.*/_pass$ii/" caustic_parameters

        # Run the caustic scan.
        caxmirrorscan-cli -p caustic_parameters
    else
        sed -i "s/_pass.*/_pass$ii/" caustic_parameters

        # Run the caustic scan.
        caxmirrorscan-cli -p caustic_parameters_back
    fi

done
echo "Scan completed."
