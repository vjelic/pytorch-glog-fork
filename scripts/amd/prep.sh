#!/bin/bash
set -ex

BUILD_DIR=/tmp/pytorch

rm -rf $BUILD_DIR
cp -rf /var/lib/jenkins/pytorch /tmp
chmod -R 777 $BUILD_DIR
ls $BUILD_DIR
