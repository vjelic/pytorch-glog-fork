#!/bin/bash

set -ex

# Optionally install python from soruce, into /opt/conda
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then

  echo "deb-src http://archive.ubuntu.com/ubuntu/ focal main" > /etc/apt/sources.list.d/python3-build-dep.list

  apt-get update
  apt-get build-dep -y python3
  apt-get install -y pkg-config
  apt-get install -y build-essential gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev

  git clone --recursive https://github.com/python/cpython -b $ANACONDA_PYTHON_VERSION
  pushd cpython
  ./configure --with-pydebug --prefix=/opt/conda
  make -j
  make -j install
  popd
  rm -rf cpython

  pushd /opt/conda/bin
  ln -s python3 python
  ln -s pip3 pip
  popd

  export PATH="/opt/conda/bin:$PATH"

  # Install PyTorch python deps, as per https://github.com/pytorch/pytorch README
  # DO NOT install cmake here as it would install a version newer than 3.13, but
  # we want to pin to version 3.13.
  CONDA_COMMON_DEPS="astunparse pyyaml mkl==2022.0.1 mkl-include==2022.0.1 setuptools cffi future six"
  if [ "$ANACONDA_PYTHON_VERSION" = "3.10" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    pip install numpy==1.21.2 ${CONDA_COMMON_DEPS}
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.9" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    pip install numpy==1.19.2 ${CONDA_COMMON_DEPS}
  elif [ "$ANACONDA_PYTHON_VERSION" = "3.8" ]; then
    # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
    pip install numpy==1.18.5 ${CONDA_COMMON_DEPS}
  else
    # Install `typing_extensions` for 3.7
    pip install numpy==1.18.5 ${CONDA_COMMON_DEPS} typing_extensions
  fi

  # after MKL is installed using pip, the *.so's are *.so.2 and are missing symlinks to *.so names
  pushd /opt/conda/lib
  for f in libmkl*.so.2
  do
      ln -s $f ${f::-2}
  done
  popd

  # TODO: This isn't working atm
  #pip install nnpack -c killeent

  # Install some other packages, including those needed for Python test reporting
  pip install -r /opt/conda/requirements-ci.txt

  # Update scikit-learn to a python-3.8 compatible version
  if [[ $(python -c "import sys; print(int(sys.version_info >= (3, 8)))") == "1" ]]; then
    pip install -U scikit-learn
  else
    # Pinned scikit-learn due to https://github.com/scikit-learn/scikit-learn/issues/14485 (affects gcc 5.5 only)
    pip install scikit-learn==0.20.3
  fi

fi
