#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>


#include <cublas_v2.h>
#include "helper.hpp"




// random initialize a buffer of float
void random_init(float* buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}




// --- Example usage ---
// C = A @ B_T : [B * F]
// A : [B * D]
// B : [F * D]
int main(int argc, char* argv[]) {
    int batch_size = 2;
    int hidden_size = 4096;
    int expert_size = 1024;
    int iter = 20;
    int warmup_iter=5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--batch" && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } if (arg == "--hidden" && i + 1 < argc) {
            hidden_size = atoi(argv[++i]);
        } else if (arg == "--expert" && i + 1 < argc) {
            expert_size = atof(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            iter = atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--batch <batch_size>] [--hidden <hidden_size>] [--output <output_size>] [--iter <iter>]" << std::endl;
            return 0;
        }
    }

    std::cout << "Batch size: " << batch_size << ", hidden_size: " << hidden_size << ", expert_size: " << expert_size << ", iter: " << iter << std::endl;

    uint16_t *A[iter+warmup_iter];
    __nv_bfloat16 *d_A[iter+warmup_iter];

    uint16_t *B[iter+warmup_iter];
    __nv_bfloat16 *d_B[iter+warmup_iter];

    __nv_bfloat16 *d_C[iter+warmup_iter];

    for (int i = 0; i < iter+warmup_iter; i++)
    {
        // cudaMalloc(&d_A[i], n*sizeof(float));
        A[i] = (uint16_t*) malloc(batch_size * hidden_size *sizeof(uint16_t));
        random_bf16(A[i], batch_size * hidden_size, 0.02f, 42+i);
        CUDA_CHECK(cudaMalloc(&d_A[i], batch_size * hidden_size * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_A[i], A[i], batch_size * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        B[i] = (uint16_t*) malloc(hidden_size * expert_size * sizeof(uint16_t));
        random_bf16(B[i], hidden_size * expert_size, 0.02f, 42+i);
        CUDA_CHECK(cudaMalloc(&d_B[i], hidden_size * expert_size * sizeof(__nv_bfloat16)));

        cudaMalloc(&d_C[i], batch_size * expert_size * sizeof(__nv_bfloat16));
        
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    // float alpha = 1.5f;
    for (int i = 0; i < iter + warmup_iter; i++)
    {
        if (i == warmup_iter)
        {
            cudaDeviceSynchronize();
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        cudaMemcpy(d_B[i], B[i], hidden_size * expert_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        gemm_rowmajor_bf16(handle, d_A[i], d_B[i], d_C[i], batch_size, hidden_size, expert_size);
        // bool ok = verify_bf16_gemm_cpu_sample(d_A[i], d_B[i], d_C[i], batch_size, hidden_size, expert_size,
        //                                   /*num_samples=*/10,
        //                                   /*atol=*/5e-2f, /*rtol=*/5e-2f,
        //                                   /*seed=*/1234ULL);
        // if (!ok) {
        //     std::cout << "bf16_gemm failed" << std::endl;
        // }
        
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
        free(B[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }


    // open the file in append mode
    std::ofstream outfile;
    outfile.open("results/native.csv", std::ios_base::app);
    outfile << int(hidden_size * expert_size * sizeof(__nv_bfloat16)) <<  "," << (elapsed.count() / iter) << std::endl;
    outfile.close();
    
    
    return 0;
}