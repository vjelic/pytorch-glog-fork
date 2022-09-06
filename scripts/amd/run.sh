clear

set -x
# set -e

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# copy cur dir to tmp dir
TMP_DIR=/tmp/pytorch
# TMP_DIR=/var/lib/jenkins/pytorch
bash scripts/amd/create_temp_dir.sh $TMP_DIR | tee $LOG_DIR/create_temp_dir.log

# cp_to_temp() {
# 	file_name=$1

# 	TARGET_FILE=$TMP_DIR/$file_name
# 	cp -rf $file_name $TARGET_FILE

# 	if [ -f "$TARGET_FILE" ]; then
# 		echo "Found $TARGET_FILE."
# 	else
# 		echo "Did not Find $TARGET_FILE."
# 	fi
# }

# cp_to_temp aten/src/ATen/native/Convolution.cpp
# cp_to_temp aten/src/ATen/native/ConvUtils.h
# cp_to_temp aten/src/ATen/native/miopen/Conv_miopen.cpp
# cp_to_temp test/test_nn.py
# cp_to_temp scripts/amd

# build pytorch
pip uninstall torch -y
# export PYTORCH_ROCM_ARCH="gfx1030"
# export PYTORCH_ROCM_ARCH="gfx908;gfx1030"
export PYTORCH_ROCM_ARCH="gfx908"
ln -s /opt/rocm/rccl/lib/librccl.so /usr/lib/librccl.so
cd $TMP_DIR
bash .jenkins/pytorch/build.sh 2>&1 | tee $LOG_DIR/build.log

# test
# bash scripts/amd/build.sh | tee $LOG_DIR/build.log
# bash scripts/amd/build_torchvision.sh | tee $LOG_DIR/build_torchvision.log
bash scripts/amd/test.sh $LOG_DIR 2>&1 |tee $LOG_DIR/test.log
# bash scripts/amd/benchmark.sh 2>&1 | tee $LOG_DIR/benchmark.log
