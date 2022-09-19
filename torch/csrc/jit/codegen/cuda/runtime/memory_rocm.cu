// Utility macro for this file
#define DEVICE_INLINE __device__ inline

namespace rocm {

// Global to SMEM load that is synchronous
template <typename dtype, int len>
DEVICE_INLINE void cpSync(Array<dtype, len, len>* smem_ptr, Array<dtype, len, len>* gmem_ptr) {
#pragma unroll
    for (int i = 0; i < len; ++i) {
        smem_ptr->array[i] = gmem_ptr->array[i];
    }
}

} // namespace rocm

#undef DEVICE_INLINE
