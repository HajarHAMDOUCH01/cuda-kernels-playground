#include <iostream>
#include <cuda_runtime.h>

__global__ void initVectors(int *vector, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){ // each thread initializes an element in the vector
        vector[idx] = idx;
    }
}

__global__ void reduce_seq_addressing(int *g_in_data, int *g_out_data){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        // reduce_seq_addressing -- check out the reverse loop above
        if (tid < s){   // then, we check threadID to do our computation
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }
}

int main(){
    int N = 1000;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int n = N;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *d_vector;
    cudaMalloc(&d_vector, N * sizeof(int));

    // Time initialization
    cudaEventRecord(start);
    initVectors<<<blocksPerGrid, threadsPerBlock>>>(d_vector, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float initTime = 0;
    cudaEventElapsedTime(&initTime, start, stop);
    
    int *d_in = d_vector;
    int *d_out;

    // Time reduction
    cudaEventRecord(start);
    
    int totalIterations = 0;
    while (n > 1) {
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        cudaMalloc(&d_out, blocksPerGrid * sizeof(int));

        reduce_seq_addressing<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out, n);

        cudaDeviceSynchronize();

        if (d_in != d_vector)
            cudaFree(d_in);

        d_in = d_out;
        n = blocksPerGrid;
        totalIterations++;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float reduce_seq_addressingTime = 0;
    cudaEventElapsedTime(&reduce_seq_addressingTime, start, stop);
    
    int result;
    cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate bandwidth
    int totalDataRead = N * sizeof(int); // Initial read
    int totalDataWritten = N * sizeof(int); // Writes during reduction
    float totalDataGB = (totalDataRead + totalDataWritten) / (1024.0f * 1024.0f * 1024.0f);
    float totalTimeSeconds = (initTime + reduce_seq_addressingTime) / 1000.0f;
    float bandwidthGBps = totalDataGB / totalTimeSeconds;

    // Results
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Parallel Reduction Profiling" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Array Size (N):          " << N << std::endl;
    std::cout << "Threads Per Block:       " << threadsPerBlock << std::endl;
    std::cout << "Reduction Iterations:    " << totalIterations << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Initialization Time:     " << initTime << " ms" << std::endl;
    std::cout << "Reduction Time:          " << reduce_seq_addressingTime << " ms" << std::endl;
    std::cout << "Total Time:              " << (initTime + reduce_seq_addressingTime) << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Effective Bandwidth:     " << bandwidthGBps << " GB/s" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Sum Result:              " << result << std::endl;
    std::cout << "Expected:  " << (N * (N - 1) / 2) << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    cudaFree(d_vector);
    cudaFree(d_in);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}