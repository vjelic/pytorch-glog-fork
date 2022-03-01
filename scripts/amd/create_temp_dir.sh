#!/bin/bash
set -ex

# SOURCE_DIR=/var/lib/jenkins/pytorch
SOURCE_DIR=$(pwd)
BUILD_DIR="${1:-"/tmp/pytorch"}"

rm -rf $BUILD_DIR
cp -rf $SOURCE_DIR $BUILD_DIR
chmod -R 777 $BUILD_DIR
ls $BUILD_DIR
