import torch
import torch.utils.benchmark

# https://github.com/pytorch/pytorch/blob/master/torch/utils/benchmark/op_fuzzers/binary.py
from torch.utils.benchmark.op_fuzzers.binary import BinaryOpFuzzer


for i, (tensors, tensor_properties, _) in enumerate(BinaryOpFuzzer(seed=0, cuda=True).take(n=5)):
  timer = torch.utils.benchmark.Timer("x + y", globals=tensors)
  for k, v in tensor_properties.items():
    print(f"{k}:  {v}")
  print(timer.blocked_autorange(min_run_time=2))
  print()