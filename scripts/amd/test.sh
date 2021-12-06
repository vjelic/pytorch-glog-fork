set -ex
# clear

# export HIP_VISIBLE_DEVICES=0
# export HIP_HIDDEN_FREE_MEM 500
# export HIP_TRACE_API=1
# export HIP_DB=api+mem+copy
# export HIP_API_BLOCKING=1
# export HIP_LAUNCH_BLOCKING_KERNELS kernel1,kernel2,...
# export HCC_DB 0x48a
# export HCC_SERIALIZE_KERNEL=3
# export HCC_SERIALIZE_COPY=3

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# bash scripts/amd/copy.sh

export PYTORCH_TEST_WITH_ROCM=1

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
# PYTORCH_DIR="/tmp/pytorch"
PYTORCH_DIR=$(pwd)

cd $PYTORCH_DIR/test
git branch

# tests
# python test_fx.py --verbose
# python distributed/optim/test_zero_redundancy_optimizer.py TestZeroRedundancyOptimizerDistributed.test_zero_model_parallel_without_bucket_view -v
python distributed/optim/test_zero_redundancy_optimizer.py TestZeroRedundancyOptimizerDistributed.test_zero_model_parallel_with_bucket_view -v
# python run_test.py -i distributed/optim/test_zero_redundancy_optimizer.py -v

