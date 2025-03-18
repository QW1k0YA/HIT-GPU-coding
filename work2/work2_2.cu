#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

void __global__ what_is_id(unsigned int* const block,
                           unsigned int* const thread,
                           unsigned int* const warp,
                           unsigned int* const calc_thread) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;

    warp[thread_idx] = threadIdx.x / WARP_SIZE;

    calc_thread[thread_idx] = thread_idx;

    if (thread_idx == 0 ||
        block[thread_idx - 1] != block[thread_idx] ||
        warp[thread_idx - 1] != warp[thread_idx]) {
        printf("blockId:%d, warp:%d\n", block[thread_idx], warp[thread_idx]);
    }
}
int main() {
    cudaDeviceProp prop;
    int device;
    int threadSize = 1024;
    int blockSize = 128;
    int gridSize = 2;
    unsigned int h_block[1024];
    unsigned int h_thread[1024];
    unsigned int h_warp[1024]; // Not typically used in this way; warp size is fixed by GPU hardware
    unsigned int *d_block, *d_thread, *d_warp, *d_calc_thread;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    // Printing warp size for demonstration purposes
    printf("warpSize: %d\n", prop.warpSize);

    // Allocate device memory
    cudaMalloc((void**)&d_block, threadSize * sizeof(unsigned int));
    cudaMalloc((void**)&d_thread, threadSize * sizeof(unsigned int));
    cudaMalloc((void**)&d_warp, threadSize * sizeof(unsigned int));
    cudaMalloc((void**)&d_calc_thread, threadSize * sizeof(unsigned int));

    // Initialize host arrays (not strictly necessary as they will be overwritten by cudaMemcpy, but good practice)
    for (int i = 0; i < threadSize; i++) {
        h_block[i] = 0;
        h_thread[i] = 0;
        h_warp[i] = 0; // Again, not typically used; just for demonstration
    }

    // Copy host arrays to device
    cudaMemcpy(d_block, h_block, threadSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_thread, h_thread, threadSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_warp, h_warp, threadSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    what_is_id<<<gridSize, blockSize>>>(d_block, d_thread, d_warp, d_calc_thread);

    // Note: No cudaMemcpy back to host is performed here; typically, you would copy the results back to verify them.

    // Free device memory
    cudaFree(d_block);
    cudaFree(d_thread);
    cudaFree(d_warp);
    cudaFree(d_calc_thread);

    return 0;
}