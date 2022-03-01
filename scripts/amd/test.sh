# set -ex
# clear

ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
# rm -rf $LOG_DIR
# mkdir -p $LOG_DIR
# chmod -R 777 $LOG_DIR

export PYTORCH_TEST_WITH_ROCM=1

# deps
pip3 install ninja
sudo apt install binutils-dev

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

# sh scripts/amd/check_warp.sh

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
PYTORCH_DIR="/tmp/pytorch"
# PYTORCH_DIR=$(pwd)
cd $PYTORCH_DIR/test

# navi 21 failing tests
python3 distributed/test_distributed_spawn.py --verbose 2>&1 | tee $LOG_DIR/test_distributed_spawn.log # segfault

python3 test_jit_fuser_te.py --verbose 2>&1 | tee $LOG_DIR/test_jit_fuser_te.log #segfaults

python3 test_quantization.py --verbose 2>&1 | tee $LOG_DIR/test_quantization.log # segfaults


# passing tests

# python3 test_nn.py --verbose 2>&1 | tee $LOG_DIR/test_nn.log # 22 mismatchs down to 19
# python3 test_nn.py --verbose 2>&1 TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64 | tee $LOG_DIR/TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64.log
# TEST_LOG=$LOG_DIR/TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64.log
# grep -oP -iw "Kernel Name:\K.*" $TEST_LOG | tee $LOG_DIR/TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64_kernel_names.log
# sort $LOG_DIR/TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64_kernel_names.log | uniq | tee $LOG_DIR/TestNNDeviceTypeCUDA.test_embedding_bag_device_cuda_int64_int64_float64_kernel_names_uniq.log

# python3 distributed/test_c10d_gloo.py --verbose 2>&1 | tee $LOG_DIR/test_c10d_gloo.log

# python3 distributed/test_c10d_nccl.py --verbose 2>&1 | tee $LOG_DIR/test_c10d_nccl.log

# python3 distributed/test_c10d_spawn_gloo.py --verbose 2>&1 | tee $LOG_DIR/test_c10d_spawn_gloo.log

# python3 distributed/test_data_parallel.py --verbose 2>&1 | tee $LOG_DIR/test_data_parallel.log

# python3 distributions/test_constraints.py --verbose 2>&1 | tee $LOG_DIR/test_constraints.log

# python3 test_autograd.py --verbose 2>&1 | tee $LOG_DIR/test_autograd.log

# python3 test_binary_ufuncs.py --verbose 2>&1 | tee $LOG_DIR/test_binary_ufuncs.log

# python3 test_cpp_api_parity.py --verbose 2>&1 | tee $LOG_DIR/test_cpp_api_parity.log

# python3 test_cuda.py --verbose 2>&1 | tee $LOG_DIR/test_cuda.log

# python3 test_functional_autograd_benchmark.py --verbose 2>&1 | tee $LOG_DIR/test_functional_autograd_benchmark.log

# python3 test_indexing.py --verbose 2>&1 | tee $LOG_DIR/test_indexing.log

# python3 test_linalg.py --verbose 2>&1 | tee $LOG_DIR/test_linalg.log

# python3 test_modules.py --verbose 2>&1 | tee $LOG_DIR/test_modules.log

# python3 test_namedtensor.py --verbose 2>&1 | tee $LOG_DIR/test_namedtensor.log

# python3 test_ops.py --verbose 2>&1 | tee $LOG_DIR/test_ops.log

# python3 test_reductions.py --verbose 2>&1 | tee $LOG_DIR/test_reductions.log

# python3 test_shape_ops.py --verbose 2>&1 | tee $LOG_DIR/test_shape_ops.log

# python3 test_sort_and_select.py --verbose 2>&1 | tee $LOG_DIR/test_sort_and_select.log

# python3 test_sparse.py --verbose 2>&1 | tee $LOG_DIR/test_sparse.log

# python3 test_sparse_csr.py --verbose 2>&1 | tee $LOG_DIR/test_sparse_csr.log

# python3 test_spectral_ops.py --verbose 2>&1 | tee $LOG_DIR/test_spectral_ops.log

# python3 test_tensor_creation_ops.py --verbose 2>&1 | tee $LOG_DIR/test_tensor_creation_ops.log

# python3 test_torch.py --verbose 2>&1 | tee $LOG_DIR/test_torch.log

# python3 test_type_promotion.py --verbose 2>&1 | tee $LOG_DIR/test_type_promotion.log
# python3 test_type_promotion.py --verbose TestTypePromotionCUDA.test_sparse_sub_cuda 2>&1 | tee $LOG_DIR/test_type_promotion.log

# python3 test_unary_ufuncs.py --verbose 2>&1 | tee $LOG_DIR/test_unary_ufuncs.log
# python3 test_unary_ufuncs.py --verbose TestUnaryUfuncsCUDA.test_mvlgamma_argcheck_cuda 2>&1 | tee $LOG_DIR/TestUnaryUfuncsCUDA.test_mvlgamma_argcheck_cuda.log

# python3 test_utils.py --verbose 2>&1 | tee $LOG_DIR/test_utils.log

# python3 test_view_ops.py --verbose 2>&1 | tee $LOG_DIR/test_view_ops.log
# python3 test_view_ops.py --verbose 2>&1 TestOldViewOpsCUDA.test_python_types_cuda | tee $LOG_DIR/TestOldViewOpsCUDA.test_python_types_cuda.log

# python3 test_vmap.py --verbose 2>&1 | tee $LOG_DIR/test_vmap.log
# python3 test_vmap.py --verbose 2>&1 TestVmapBatchedGradientCUDA.test_median_cuda | tee $LOG_DIR/TestVmapBatchedGradientCUDA.test_median_cuda.log
# python3 test_vmap.py --verbose 2>&1 TestVmapBatchedGradientCUDA.test_trace_cuda | tee $LOG_DIR/TestVmapBatchedGradientCUDA.test_trace_cuda.log
