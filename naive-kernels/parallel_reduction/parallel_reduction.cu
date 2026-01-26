#include <iostream>
#include <cuda_runtime.h>

__global__ void initVectors(int *vector, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){ // each thread initializes an element in the vector
        vector[idx] = idx;
    }
}

__global__ void reduce(int *g_in_data, int *g_out_data, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;// this is the thread's idx in the block it is in
    unsigned int i = blockIdx.x * blockDim.x + tid;// this is for tracking elements of the data within the block the thread is in

    // Load or pad with 0
    sdata[tid] = (i < n) ? g_in_data[i] : 0;// each thread copies elements of its block -IN PARALLEL : in threads and in blocks-
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {// threads within the block are multiplies of 2 , 4 , etc
            sdata[tid] += sdata[tid + s]; 
        }
        __syncthreads();
    }

    if (tid == 0) {// the element 0 of each block has the reduction output
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
    
    int *d_in = d_vector; // Fixed: initialize d_in
    int *d_out;

    // Time reduction
    cudaEventRecord(start);
    
    int totalIterations = 0;
    while (n > 1) {
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        cudaMalloc(&d_out, blocksPerGrid * sizeof(int));

        reduce<<<blocksPerGrid, threadsPerBlock,
                threadsPerBlock * sizeof(int)>>>(d_in, d_out, n);

        cudaDeviceSynchronize();

        if (d_in != d_vector)
            cudaFree(d_in);

        d_in = d_out;
        n = blocksPerGrid;
        totalIterations++;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float reduceTime = 0;
    cudaEventElapsedTime(&reduceTime, start, stop);
    
    int result;
    cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate bandwidth
    int totalDataRead = N * sizeof(int); // Initial read
    int totalDataWritten = N * sizeof(int); // Writes during reduction
    float totalDataGB = (totalDataRead + totalDataWritten) / (1024.0f * 1024.0f * 1024.0f);
    float totalTimeSeconds = (initTime + reduceTime) / 1000.0f;
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
    std::cout << "Reduction Time:          " << reduceTime << " ms" << std::endl;
    std::cout << "Total Time:              " << (initTime + reduceTime) << " ms" << std::endl;
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