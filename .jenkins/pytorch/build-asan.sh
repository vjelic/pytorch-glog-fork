#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

echo "Clang version:"
clang --version

# hipify sources
python tools/amd_build/build_amd.py

# patch fbgemm to work around build failure
pushd third_party/fbgemm
patch -p1 -i ../../.jenkins/pytorch/fbgemm.patch
popd

python tools/stats/export_test_times.py

# detect_leaks=0: Python is very leaky, so we need suppress it
# symbolize=1: Gives us much better errors when things go wrong
export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1:symbolize=1:detect_odr_violation=0
if [ -n "$(which conda)" ]; then
  export CMAKE_PREFIX_PATH=/opt/conda
fi

# otherwise any program run at build time will fail
export LD_LIBRARY_PATH=/opt/rocm/llvm/lib/clang/17.0.0/lib/linux
export HSA_XNACK=1

# TODO: Make the ASAN flags a centralized env var and unify with USE_ASAN option
export CC="/opt/rocm/llvm/bin/clang"
export CXX="/opt/rocm/llvm/bin/clang++"
export LDSHARED="/opt/rocm/llvm/bin/clang --shared -fuse-ld=lld"
export LDFLAGS="-fuse-ld=lld"
#export CFLAGS="-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-address-use-after-scope -shared-libasan"
#export CXXFLAGS="-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-address-use-after-scope -shared-libasan"
export CFLAGS="-ggdb -fsanitize=address -shared-libasan"
export CXXFLAGS="-ggdb -fsanitize=address -shared-libasan"
#export USE_ASAN=1
export USE_CUDA=0
export USE_ROCM=1
export USE_MKLDNN=0
python setup.py bdist_wheel
python -mpip install "$(echo dist/*.whl)[opt-einsum]"
exit 0

# Test building via the sdist source tarball
python setup.py sdist
mkdir -p /tmp/tmp
pushd /tmp/tmp
tar zxf "$(dirname "${BASH_SOURCE[0]}")/../../dist/"*.tar.gz
cd torch-*
python setup.py build --cmake-only
popd

print_sccache_stats

assert_git_not_dirty
