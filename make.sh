#!/usr/bin/env bash
set -euo pipefail

# export CXXFLAGS="-std=c++11"
# export CFLAGS="-std=c99"

PYTHON=${PYTHON:-"python"}
CC=${CC:-"gcc"}
CXX=${CXX:-"g++"}
export CC CXX
# Older conda environments can ship a linker shim and libstdc++ incompatible
# with the host. Link the C++ runtime statically unless explicitly overridden.
export LDSHARED=${LDSHARED:-"$CXX -shared -static-libstdc++"}
cd models/ops

echo "Building bbox op..."
python setup_ssd.py build_ext --inplace
rm -rf build

echo "Building rcnn op..."
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup_rcnn.py build_ext --inplace
rm -rf build
