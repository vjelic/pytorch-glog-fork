echo "testing"

# export HIP_VISIBLE_DEVICES=0
# export HIP_HIDDEN_FREE_MEM 500
# export HIP_TRACE_API=1
# export HIP_DB=api+mem+copy
# export HIP_API_BLOCKING=1
# export HIP_LAUNCH_BLOCKING_KERNELS kernel1,kernel2,...
# export HCC_DB 0x48a
# export HCC_SERIALIZE_KERNEL=3
# export HCC_SERIALIZE_COPY=3

export AMD_LOG_LEVEL=3
export HIP_LAUNCH_BLOCKING=1

# sh scripts/amd/check_warp.sh

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
PYTORCH_DIR="/tmp/pytorch"
# PYTORCH_DIR=$(pwd)
cd $PYTORCH_DIR/test

# PYTORCH_TEST_WITH_ROCM=1 pytest --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_type_promotion_cuda_float32
# PYTORCH_TEST_WITH_ROCM=1 python distributed/algorithms/test_join.py --verbose TestJoin.test_join_kwargs
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py  -c --verbose TestGradientsCUDA.test_forward_mode_AD_linalg_tensorinv_cuda_float64
bash scripts/amd/run_individually.sh
