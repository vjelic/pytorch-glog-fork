echo "testing"

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
# PYTORCH_DIR="/tmp/pytorch"
PYTORCH_DIR=`pwd`

cd $PYTORCH_DIR/test

# PYTORCH_TEST_WITH_ROCM=1 pytest --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_type_promotion_cuda_float32
PYTORCH_TEST_WITH_ROCM=1 python distributed/algorithms/test_join.py --verbose TestJoin.test_join_kwargs