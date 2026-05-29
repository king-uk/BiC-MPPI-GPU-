#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s (%d)\n", cudaGetErrorString(error), (int)error);
        return 1;
    }
    printf("Device count: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s, compute capability %d.%d\n", i, prop.name, prop.major, prop.minor);
    }
    curandGenerator_t gen;
    curandStatus_t status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        printf("curandCreateGenerator failed with status %d\n", (int)status);
        return 1;
    }
    printf("curandCreateGenerator succeeded!\n");
    curandDestroyGenerator(gen);
    return 0;
}
