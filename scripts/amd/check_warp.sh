# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

cd scripts/amd/
rm -rf ./check_warp
/opt/rocm/bin/hipcc check_warp.cu -o check_warp
./check_warp
