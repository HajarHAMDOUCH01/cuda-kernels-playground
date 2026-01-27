#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

__global__ void naive_kernel(float *d_in, float *d_out, int n, int m, float EPSILON){
    float mean = 0.0f;
    float var = 0.0f;
    int rowidx = blockIdx.x * blockDim.x + threadIdx.x; // a thread for each row 
    
    if (rowidx >= m) return;

    for (int col = 0; col < n; col++){
        int elementidx = rowidx * n + col;
        mean += d_in[elementidx]; 
    }
    mean /= n;

    for (int col = 0; col < n; col++){
        int elementidx = rowidx * n + col;
        var += (d_in[elementidx] - mean) * (d_in[elementidx] - mean);
    }
    var /= n;

    float stdev = sqrt(var + EPSILON);
    for (int col = 0; col < n; col++){
        int elementidx = rowidx * n + col;
        d_out[elementidx] = (d_in[elementidx] - mean) / stdev;
    }
}

int main(){
    const float EPSILON = 1e-6f;
    const int m = 1024;  
    const int n = 1024;   
    const int size = m * n;
    const int num_runs = 1000;
    
    float *h_in = new float[size];
    float *h_out = new float[size];
    
    srand(time(0));
    for (int i = 0; i < size; i++){
        h_in[i] = (float)rand() / RAND_MAX * 10.0f;
    }
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;
    
    naive_kernel<<<gridSize, blockSize>>>(d_in, d_out, n, m, EPSILON);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++){
        naive_kernel<<<gridSize, blockSize>>>(d_in, d_out, n, m, EPSILON);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / num_runs;
    
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "time : " << avg_time << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    
    return 0;
}