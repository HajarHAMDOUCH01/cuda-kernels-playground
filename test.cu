#include <iostream>
#include <cuda_runtime.h>

// cuda kernel pour initialization des vecteurs
__global__ void initVectors(float *a, float *b, int n){
    // formule pour calculer l'indice global d'un thread
    // blockIdx.x  : l'indice du block au quel appartient ce thread
    // blockDim.x  : le nombre de threads par block
    // threadIdx.x : l'indice du thread dans son block
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // seulement les n threads vont faire les operations
    // les autres sont lancer mais ils font rien
    if (idx < n){
        a[idx] = idx * 1.0f;
        b[idx] = idx * 2.0f;
    }
}

// cuda kernel pour addition des vecteurs
__global__ void addVectors(const float *a, const float *b, float *c, int n){
    // pointers are in cpu pointing to vectors allocated in gpu
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    const int N = 1000;
    const int size = N * sizeof(float);

    // allocate host memory 
    float *h_c = new float[N];

    // allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1 ) / threadsPerBlock;

    // launching a kernel to initialize vectors in gpu
    initVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("ERROR %s\n", cudaGetErrorString(err));
    }
    
    // launch kernel addition vectors
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // copy result back to host for verification
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 10; i++){
        std::cout << "h_c[" << i << "] = " << h_c[i] << std::endl;
    }

    // verfy results 
    bool success = true;
    for (int i=0; i < N; i++){
        float expected = i * 1.0f + i * 2.0f; // verification element par element
        if (fabs(h_c[i] - expected) > 1e-5){
            success = false;
            std::cout << "faaaalse" << std::endl;
            std::cout << "error in idx : " << i << ", expected " << expected << ", but got " << h_c[i] << std::endl;
            break;
        }
    }
    if(success) {
        std::cout << "trueeeee" << std::endl;
    }
    // get the gpu info 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "compute capability :" << prop.major << "." << prop.minor << std::endl;
    std::cout << "total global memory :" << prop.totalGlobalMem / (1024*1024) << "MB" << std::endl;

    //clean up 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_c;


    return 0;
}

