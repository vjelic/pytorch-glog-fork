#!/bin/bash

set -ex

[ -n "$DEVTOOLSET_VERSION" ]

#yum install -y centos-release-scl
#yum install -y devtoolset-$DEVTOOLSET_VERSION

dnf install -y rpmdevtools

#echo "source scl_source enable devtoolset-$DEVTOOLSET_VERSION" > "/etc/profile.d/devtoolset-$DEVTOOLSET_VERSION.sh"
