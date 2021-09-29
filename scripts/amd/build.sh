#!/bin/bash
set -ex
clear

BUILD_DIR=/tmp/pytorch

if [true]; then
    CUR_FILE=$1
    chmod -R 777 $CUR_FILE
    cp -rf --parents $CUR_FILE $BUILD_DIR

    cd $BUILD_DIR/build
    cmake --build . --target install --config Release -- -j 16
else
    cd $BUILD_DIR
    export MAX_JOBS=16
    pip uninstall torch -y
    python tools/amd_build/build_amd.py
    VERBOSE=1 USE_ROCM=1 python3 setup.py develop | tee DEBUG_BUILD.log
fi
