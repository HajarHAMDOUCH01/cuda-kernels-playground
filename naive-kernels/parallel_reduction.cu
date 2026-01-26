#include <iostream>
#include<cuda_runtime.h>


__global__ void initVectors(int *vector, int n){
    idx = blockIdx.x * blockDim + threadIdx;
    if (idx < n){ // eahc thread initailizes an element in the vector
        vector[idx] = idx * 1.0f;
    }
}

// does sum over an array elements
__global__ void reduce(int *g_in_data, int *g_out_data){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x; // this is the thread's idx in the block it is in
    unsigned int i = int blockIdx.x * blockDim + threadIdx; // this is for tracking elements of the data within the block the thread is in

    sdata[tid] = g_in_data[i]; // each thread copies elemnts of its block -IN PARALLEL : in threads and in blocks-
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2){ // this is the steps
        if (tid % (2*s) == 0){ // threads within the block are multiplies of 2 , 4 , etc
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0){ // the element 0 of each block has the reduction output
        g_out_data[blockIdx.x] = sdata[0];
    }
}

int main(){
    N = 1000;
    int threadsPerBlock = 256;
    int n = N;

    int *d_in = d_vector;
    int *d_out;

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
    }
    int result;
    cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum = " << result << std::endl;

}
