#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#include <cstring>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>


#include <cublas_v2.h>

template <typename T>
__global__ void add_alpha_kernel(const T* __restrict__ A,
                                 T* __restrict__ B,
                                 T alpha,
                                 int n)
{
    // Grid-stride loop for any n
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        B[i] = A[i] + alpha;
    }
}

template <typename T>
void add_alpha(const T* d_A, T* d_B, T alpha, int n, cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    // Optionally cap blocks to a multiple of SMs; simple version:
    add_alpha_kernel<T><<<blocks, threads, 0, stream>>>(d_A, d_B, alpha, n);
}


// random initialize a buffer of float
void random_init(float* buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


void verify(float* a, size_t m, size_t n, float alpha, float* result) {
    size_t verify_size = std::min(m * n, (size_t)1000);
    for (size_t i = 0; i < verify_size; i++)
    {
        if (std::abs(a[i] + alpha - result[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << ": expected " << (a[i] + alpha) << ", got " << result[i] << "\n";

            exit(EXIT_FAILURE);
        }
        // else{
        //     std::cout << "a[" << i << "] = " << a[i] << ", result[" << i << "] = " << result[i] << "\n";
        // }
    }
    // std::cout << "Verification passed!" << std::endl;
}

// --- Example usage ---
int main(int argc, char* argv[]) {
    int iter = 50;
    int warmup_iter=5;
    int n = 4096 * 14336 / 2;
    float *A[iter+warmup_iter], *d_A[iter+warmup_iter], *d_B[iter+warmup_iter];


    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--n" && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            iter = atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--batch <batch_size>] [--hidden <hidden_size>] [--output <output_size>] [--iter <iter>]" << std::endl;
            return 0;
        }
    }

    std::cout << "n: " << n << ", iter: " << iter << std::endl;

    for (int i = 0; i < iter+warmup_iter; i++)
    {
        // cudaMalloc(&d_A[i], n*sizeof(float));
        A[i] = (float*)malloc(n*sizeof(float));
        random_init(A[i], n);

        cudaMalloc(&d_A[i], n*sizeof(float));
        
        cudaMalloc(&d_B[i], n*sizeof(float));
    }
    
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter + warmup_iter; i++)
    {
        if (i == warmup_iter)
        {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        float alpha = 1.5f;
        cudaMemcpy(d_A[i], A[i], n*sizeof(float), cudaMemcpyHostToDevice);
        add_alpha<float>(d_A[i], d_B[i], alpha, n);
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
    float latency_per_iter = elapsed.count() / iter;
    std::cout << "Average time per iteration: " << latency_per_iter << " us" << " (" << iter << "iters)" << std::endl;


    cudaDeviceSynchronize();
    for (size_t i = 0; i < iter + warmup_iter; i++)
    {
        free(A[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
    }
    
    
    return 0;
}