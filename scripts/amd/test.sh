set -ex
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

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

bash scripts/amd/copy.sh

# sh scripts/amd/check_warp.sh

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
PYTORCH_DIR="/tmp/pytorch"
# PYTORCH_DIR=$(pwd)
cd $PYTORCH_DIR/test

# PYTORCH_TEST_WITH_ROCM=1 pytest --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_type_promotion_cuda_float32
# PYTORCH_TEST_WITH_ROCM=1 python distributed/algorithms/test_join.py --verbose TestJoin.test_join_kwargs
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py   --verbose TestGradientsCUDA.test_forward_mode_AD_linalg_tensorinv_cuda_float64
# bash scripts/amd/run_individually.sh

export BACKEND="nccl"
export WORLD_SIZE="2"
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py  --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_2D_Input
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py  --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_No_Affine
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py  --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_Single_Input_Per_Process
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestDistributed.test_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestDistributed.test_qat_data_parallel
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_backward_per_tensor_cachemask_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_fixed_qparams_fq_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_forward_per_tensor
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_forward_per_tensor_cachemask_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_fq_module_per_tensor
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_learnable_backward_per_channel_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_learnable_backward_per_tensor_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_learnable_forward_per_tensor_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFakeQuantizeOps.test_numerical_consistency_per_tensor
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuant.test_fused_backward_op_fake_quant_off
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuant.test_fused_obs_fake_quant_backward_op
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuantModule.test_compare_fused_obs_fq_oss_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuantModule.test_fused_obs_fq_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestFusedObsFakeQuantModule.test_fused_obs_fq_moving_avg_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestObserver.test_state_dict_respects_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestQuantizeFx.test_qat_prepare_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py  --verbose TestQuantizeFxModels.test_qat_functional_linear

# "Cannot find Symbol" errors
# PYTORCH_TEST_WITH_ROCM=1 python test_cpp_api_parity.py  --verbose TestCppApiParity.test_torch_nn_CrossEntropyLoss_2d_indices_target_smoothing_weight_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose TestCuda.test_autocast_torch_need_autocast_promote
# PYTORCH_TEST_WITH_ROCM=1 python test_indexing.py   --verbose TestIndexingCUDA.test_take_along_dim_cuda_float32
# PYTORCH_TEST_WITH_ROCM=1 python test_jit.py   --verbose TestModels.test_mnist_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_linalg.py   --verbose TestLinalgCUDA.test_lu_cuda_complex128
# PYTORCH_TEST_WITH_ROCM=1 python test_namedtensor.py   --verbose TestNamedTensor.test_reduction_fns
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py   --verbose TestNN.test_AdaptiveLogSoftmax
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py   --verbose TestCommonCUDA.test_out_gather_cuda_float32
# PYTORCH_TEST_WITH_ROCM=1 python test_reductions.py   --verbose TestReductionsCUDA.test_median_nan_values_cuda_float16
# PYTORCH_TEST_WITH_ROCM=1 python test_sort_and_select.py   --verbose TestSortAndSelectCUDA.test_topk_integral_cuda_int64
# PYTORCH_TEST_WITH_ROCM=1 python test_torch.py  --verbose TestTorchDeviceTypeCUDA.test_gather_backward_deterministic_path_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py --verbose TestCommonCUDA.test_variant_consistency_eager_matrix_exp_cuda_complex64
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py --verbose TestCommonCUDA.test_variant_consistency_eager_unfold_cuda_complex64

# PYTORCH_TEST_WITH_ROCM=1 python test_cpp_api_parity.py  --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_indexing.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_jit.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_linalg.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_namedtensor.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_ops.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_reductions.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_sort_and_select.py   --verbose
# PYTORCH_TEST_WITH_ROCM=1 python test_torch.py --verbose

# hang
PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose TestCuda.test_caching_pinned_memory |& tee /dockerx/pytorch_rocm/test_caching_pinned_memory.log
# PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose TestCuda.test_caching_pinned_memory_multi_gpu |& tee /dockerx/pytorch_rocm/test_caching_pinned_memory_multi_gpu.log
# PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose TestCuda.test_copy_streams |& tee /dockerx/pytorch_rocm/test_copy_streams.log
# PYTORCH_TEST_WITH_ROCM=1 python test_cuda.py  --verbose TestCuda.test_cusparse_multiple_threads_same_device
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py --verbose TestNNDeviceTypeCUDA.test_embedding_bag_2D_padding_idx_cuda_bfloat16 |& tee /dockerx/pytorch_rocm/test_embedding_bag_2D_padding_idx_cuda_bfloat16.log


# segfault

