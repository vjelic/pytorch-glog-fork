# python -u batchnorm_nhwc_enable_miopen_in_runtime.py

import os
import torch

PYTORCH_MIOPEN_SUGGEST_NHWC = "PYTORCH_MIOPEN_SUGGEST_NHWC"

# enable MIOpen logging
os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
# os.environ["AMD_LOG_LEVEL"] = "3"

print("START")
m = torch.nn.BatchNorm2d(100).to("cuda:0")

# random NCHW tensor on GPU
input = torch.randn(20, 100, 5, 4).to("cuda:0")

# change tensor dims to NHWC
input = input.to(memory_format=torch.channels_last)
print(f"input.shape={input.shape}")
print(f"input.stride={input.stride()}")

print("Disable MIOpen")
if PYTORCH_MIOPEN_SUGGEST_NHWC in os.environ:
    os.environ.clear(PYTORCH_MIOPEN_SUGGEST_NHWC)
print(f"PYTORCH_MIOPEN_SUGGEST_NHWC={os.getenv(PYTORCH_MIOPEN_SUGGEST_NHWC, default=None)}. No 'MIOpen(HIP):' output expected")
# call batch norm
output = m(input)
print()

print("Enable MIOpen")
os.environ[PYTORCH_MIOPEN_SUGGEST_NHWC] = "1"
print(f"PYTORCH_MIOPEN_SUGGEST_NHWC={os.getenv(PYTORCH_MIOPEN_SUGGEST_NHWC, default=None)}. Expected 'MIOpen(HIP):' output")
# call batch norm
output = m(input)
print()

print("Disable MIOpen")
os.environ[PYTORCH_MIOPEN_SUGGEST_NHWC] = "0"
print(f"PYTORCH_MIOPEN_SUGGEST_NHWC={os.getenv(PYTORCH_MIOPEN_SUGGEST_NHWC, default=None)}. No 'MIOpen(HIP):' output expected")
# call batch norm
output = m(input)
print()

print(f"output.shape={output.shape}")
print(f"output.stride={output.stride()}")
print("END")

