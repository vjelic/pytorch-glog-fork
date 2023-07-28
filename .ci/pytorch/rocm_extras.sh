#!/bin/bash

set -ex

# This file contains ROCm-specific logic that's needed for QA to run tests successfully on ROCm
export CI=1
export PYTORCH_TEST_WITH_ROCM=1
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Manually extract the test times for ROCm since default steps may not find the BUILD_ENVIRONMENT key in the json that's hosted upstream
wget https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json
ROCM_BUILD_ENVIRONMENT_IN_JSON=$(grep -m1 "rocm" test-times.json | cut -d \" -f 2)
BUILD_ENVIRONMENT=$ROCM_BUILD_ENVIRONMENT_IN_JSON python tools/stats/export_test_times.py
