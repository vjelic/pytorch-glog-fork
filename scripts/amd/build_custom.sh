#!/bin/bash
set -ex

# export PYTORCH_ROCM_ARCH=gfx908
# export PYTORCH_ROCM_ARCH=gfx1030
export PYTORCH_ROCM_ARCH="gfx908;gfx1030"

BUILD_DIR=/tmp/pytorch
bash scripts/amd/prep.sh $BUILD_DIR

cp_to_build_dir() {
    local CUR_FILE=$1
    chmod -R 777 $CUR_FILE
    cp -rf --parents $CUR_FILE $BUILD_DIR
}

build_develop() {
    pip uninstall torch -y

    cd $BUILD_DIR
    export MAX_JOBS=16
    python tools/amd_build/build_amd.py
    VERBOSE=1 USE_ROCM=1 python3 setup.py develop | tee BUILD_DEVELOP.log
}

if true; then
    # "c10/macros/Macros.h"
    # "aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh"
    # "aten/src/ATen/native/cuda/block_reduce.cuh"
    # "aten/src/ATen/native/cuda/RangeFactories.cu"
    # "aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh"
    # "aten/src/ATen/native/cuda/MemoryAccess.cuh"
    # "aten/src/ATen/native/cuda/ROCmLoops.cuh"
    # "aten/src/ATen/native/cuda/Loops.cuh"
    # "aten/src/ATen/native/hip/Loops.cuh"
    # "aten/src/ATen/native/cuda/ScatterGatherKernel.cu"
    # "aten/src/ATen/native/cuda/LinearAlgebra.cu"

    # FILE_LIST=(
    #    "test/test_cuda.py"
    # )
    # for FILE in "${FILE_LIST[@]}"; do
    #     cp_to_build_dir $FILE
    # done

    # cd $BUILD_DIR/build
    # cmake --build . --target install --config Release -- -j 16
    build_develop
else
    build_develop
fi
