git clone https://github.com/pytorch/pytorch /tmp/pytorch
cd /tmp/pytorch/test

PYTORCH_TEST_WITH_ROCM=1 python3.6 test_spectral_ops.py TestFFTCUDA -v

# PYTORCH_TEST_WITH_ROCM=1 python test_spectral_ops.py TestFFTCUDA --verbose \
#     2>&1 | tee ../scripts/amd/test_spectral.log