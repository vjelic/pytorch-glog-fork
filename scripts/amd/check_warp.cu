#include <hip/hip_runtime.h>

__global__ void Empty(float A[], int param) {
    printf("[GPU] warpSize: %d, __AMDGCN_WAVEFRONT_SIZE: %d\n", warpSize, __AMDGCN_WAVEFRONT_SIZE);
}

int main() {
    printf("[HOST] warpSize: %d, __AMDGCN_WAVEFRONT_SIZE: %d\n", warpSize, __AMDGCN_WAVEFRONT_SIZE);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Empty), dim3(1), dim3(1), 0, 0, nullptr, 0);
    hipDeviceSynchronize();
    
    hipDeviceProp_t deviceProp;
    if (hipSuccess != hipGetDeviceProperties(&deviceProp, 0)) {
        printf("Get device properties failed.\n");
        return 1;
    } else {
        printf("The warp size is %d.\n", deviceProp.warpSize);
        printf("The gcn arch is %s\n", deviceProp.gcnArchName);
        return 0;
    }

    return 0;
}

   