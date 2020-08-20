#!/bin/bash

set -ex

[ -n "$BLIS_VERSION" ]

LIBFLAME_VERSION=${LIBFLAME_VERSION:-5.2.0}

pushd /tmp

# Download and install specific blis version in /usr/local
curl -Os --retry 3 "https://github.com/flame/blis/archive/$BLIS_VERSION.tar.gz"
tar xzf $BLIS_VERSION.tar.gz
pushd blis-$BLIS_VERSION
./configure --enable-blas --enable-shared --enable-static -t openmp x86_64
make -j
sudo make install
sudo ldconfig
popd
rm -rf blis-$BLIS_VERSION

# Download and install an accompanying LAPACK version in /usr/local
curl -Os --retry 3 https://github.com/flame/libflame/archive/$LIBFLAME_VERSION.tar.gz
tar xzf $LIBFLAME_VERSION.tar.gz
pushd libflame-$LIBFLAME_VERSION
./configure --enable-dynamic-build --enable-lapack2flame --enable-max-arg-list-hack --enable-supermatrix --disable-ldim-alignment --enable-multithreading=openmp --disable-autodetect-f77-ldflags --disable-autodetect-f77-name-mangling
make -j
sudo make install
sudo ldconfig
popd
pushd libflame-$LIBFLAME_VERSION

popd
