#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

__global__ void row_normalize_kernel(float *d_in, float *d_out, int n, int m, float EPSILON) {
    extern __shared__ float shared[]; // dynamic shared memory per block
    float* row_data = shared;          // one row per block

    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row >= m || col >= n) return;

    // Load element into shared memory
    int idx = row * n + col;
    row_data[threadIdx.x] = d_in[idx];
    __syncthreads();

    // Compute mean using reduction
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += row_data[i];
    mean /= n;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = row_data[i] - mean;
        var += val * val;
    }
    var /= n;
    float stdev = sqrtf(var + EPSILON);

    // Normalize
    d_out[idx] = (d_in[idx] - mean) / stdev; // parallel on each element on the matrix
}


int main(){
    const float EPSILON = 1e-6f;
    const int m = 2048;  
    const int n = 2048;   
    const int size = m * n;
    const int num_runs = 1000;
    
    float *h_in = new float[size];
    float *h_out = new float[size];
    dim3 blockDim(2048, 1);  
    dim3 gridDim(1, 2048);  

    
    srand(time(0));

    for (int i = 0; i < size; i++)
    { 
        h_in[i] = (float)rand() / RAND_MAX * 10.0f;
    }

    // float h_in[16] = {
    //     1.23f, 4.56f, 7.89f, 0.12f,
    //     3.45f, 6.78f, 9.01f, 2.34f,
    //     5.67f, 8.90f, 1.23f, 4.56f,
    //     7.89f, 0.12f, 3.45f, 6.78f
    // };

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++)
    //         std::cout << h_in[i*n + j] << "\t";
    //     std::cout << std::endl;
    // }
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // int blockSize = 256;
    // int gridSize = (m + blockSize - 1) / blockSize;
    
    row_normalize_kernel<<<gridDim, blockDim, n * sizeof(float)>>>(d_in, d_out, n, m, EPSILON);

    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++){
        row_normalize_kernel<<<gridDim, blockDim, n * sizeof(float)>>>(d_in, d_out, n, m, EPSILON);

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / num_runs;
    
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_out[i * n + j] << "\t"; 
            break;
        }
        std::cout << std::endl;
    }
    std::cout << "time : " << avg_time << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out;
    delete[] h_in;
    
    return 0;
}