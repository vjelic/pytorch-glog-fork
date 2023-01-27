#pragma once

#include <ATen/ATen.h>
#include <hip/hip_bfloat16.h>

namespace at {
namespace cuda {
namespace detail {

void out_of_place_fp32_to_bf16(void* in, void* out, int nElements, hipStream_t stream);

} // namespace detail
} // namespace cuda
} // namespace at

