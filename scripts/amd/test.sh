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
# bash scripts/amd/run_individually.sh

export BACKEND="nccl"
export WORLD_SIZE="2"
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py -c --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_2D_Input
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py -c --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_No_Affine
# PYTORCH_TEST_WITH_ROCM=1 python distributed/test_distributed_spawn.py -c --verbose TestDistBackendWithSpawn.test_DistributedDataParallel_SyncBatchNorm_Single_Input_Per_Process
# PYTORCH_TEST_WITH_ROCM=1 python quantization/core/test_workflow_module.py  -c --verbose quantization.core.test_workflow_module.TestDistributed.test_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestDistributed.test_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestDistributed.test_qat_data_parallel
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_backward_per_tensor_cachemask_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_fixed_qparams_fq_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_forward_per_tensor
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_forward_per_tensor_cachemask_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_fq_module_per_tensor
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_learnable_backward_per_channel_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_learnable_backward_per_tensor_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_learnable_forward_per_tensor_cuda
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFakeQuantizeOps.test_numerical_consistency_per_tensor
#  PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuant.test_fused_backward_op_fake_quant_off
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuant.test_fused_obs_fake_quant_backward_op
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuant.test_fused_obs_fake_quant_moving_avg
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuantModule.test_compare_fused_obs_fq_oss_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuantModule.test_fused_obs_fq_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestFusedObsFakeQuantModule.test_fused_obs_fq_moving_avg_module
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestObserver.test_state_dict_respects_device_affinity
# PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestQuantizeFx.test_qat_prepare_device_affinity
PYTORCH_TEST_WITH_ROCM=1 python test_quantization.py -c --verbose TestQuantizeFxModels.test_qat_functional_linear