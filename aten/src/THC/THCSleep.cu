#include <THC/THCSleep.h>


__global__ void spin_kernel(int64_t cycles)
{
  printf("spin_kernel\n");
  printf("cycles: %d\n", cycles);
  // see concurrentKernels CUDA sampl
  int64_t start_clock = clock64();
  printf("start_clock: %d\n", start_clock);
  int64_t clock_offset = 0;
  int64_t end_clock = start_clock;
  while (clock_offset < cycles)
  {
    end_clock = clock64();
    printf("end_clock: %d\n", end_clock);
    clock_offset = end_clock - start_clock;
    printf("clock_offset: %d\n", clock_offset);
  }
}

void THC_sleep(THCState* state, int64_t cycles)
{
  dim3 grid(1);
  dim3 block(1);
  spin_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
