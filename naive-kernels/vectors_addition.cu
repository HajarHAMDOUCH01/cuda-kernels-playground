#include <iostream>
#include <cuda_runtime.h>

__global__ void initVectors(float *a, float *b, int n){
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n){
        a[idx] = idx * 1.0f;
        b[idx] = idx * 2.0f;
    }
}

__global__ void addVectors(const float *a, const float *b, float *c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    const int N = 50000000;
    const int size = N * sizeof(float);

    // Allocate host memory 
    float *h_c = new float[N];

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Separate timing for each kernel
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventCreate(&start2); cudaEventCreate(&stop2);
    cudaEventCreate(&start3); cudaEventCreate(&stop3);

    // Time initVectors kernel
    cudaEventRecord(start1);
    initVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    cudaEventRecord(stop1);
    
    // Time addVectors kernel
    cudaEventRecord(start2);
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop2);
    
    // Time memory copy
    cudaEventRecord(start3);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop3);
    
    cudaEventSynchronize(stop3);
    
    float ms1, ms2, ms3;
    cudaEventElapsedTime(&ms1, start1, stop1);
    cudaEventElapsedTime(&ms2, start2, stop2);
    cudaEventElapsedTime(&ms3, start3, stop3);
    
    printf("\n=== Timing Results ===\n");
    printf("initVectors kernel:  %f ms\n", ms1);
    printf("addVectors kernel:   %f ms\n", ms2);
    printf("Memory copy D->H:    %f ms\n", ms3);
    printf("Total kernel time:   %f ms\n", ms1 + ms2);
    printf("Total time:          %f ms\n", ms1 + ms2 + ms3);

    // Verify results
    for (int i = 0; i < 10; i++){
        std::cout << "h_c[" << i << "] = " << h_c[i] << std::endl;
    }

    bool success = true;
    for (int i = 0; i < N; i++){
        float expected = i * 3.0f;
        if (fabs(h_c[i] - expected) > 1e-5){
            success = false;
            std::cout << "Error at idx " << i << ": expected " << expected << ", got " << h_c[i] << std::endl;
            break;
        }
    }
    if (success):
        print("true")
    print("false")

    // Cleanup
    cudaEventDestroy(start1); cudaEventDestroy(stop1);
    cudaEventDestroy(start2); cudaEventDestroy(stop2);
    cudaEventDestroy(start3); cudaEventDestroy(stop3);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_c;

    return 0;
}