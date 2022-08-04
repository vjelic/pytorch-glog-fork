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
bash scripts/amd/create_temp_dir.sh $TMP_DIR | tee $LOG_DIR/create_temp_dir.log


# build pytorch
pip uninstall torch -y
export PYTORCH_ROCM_ARCH="gfx908"
# export PYTORCH_ROCM_ARCH="gfx1030"
# export PYTORCH_ROCM_ARCH="gfx908;gfx1030"
cd $TMP_DIR
bash .jenkins/pytorch/build.sh 2>&1 | tee $LOG_DIR/build.log

# bash scripts/amd/build.sh | tee $LOG_DIR/build.log
# bash scripts/amd/build_torchvision.sh | tee $LOG_DIR/build_torchvision.log
bash scripts/amd/test.sh $LOG_DIR 2>&1 |tee $LOG_DIR/test.log
# bash scripts/amd/benchmark.sh 2>&1 | tee $LOG_DIR/benchmark.log
