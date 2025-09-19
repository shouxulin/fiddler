// #include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <cstdlib> // for atoi, atof
#include <sstream> // for stringstream
#include <iomanip> // for put_time
// #include "nvToolsExt.h"
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

#include "helper.hpp"

/*
gate = x @ W1ᵀ → [B, d_ff_new]
up = x @ W3ᵀ → [B, d_ff_new]
h = (SiLU(gate) ⊙ up) → [B, d_ff_new]
out = h @ W2ᵀ → [B, d_model]
*/

int main(int argc, char* argv[]) {

    int B = 2;
    int D = 4096;
    int F = 1024;
    int iter = 20;
    int warmup_iter=5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--b" && i + 1 < argc) {
            B = atoi(argv[++i]);
        } if (arg == "--d" && i + 1 < argc) {
            D = atoi(argv[++i]);
        } else if (arg == "--f" && i + 1 < argc) {
            F = atof(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            iter = atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--batch <batch_size>] [--hidden <hidden_size>] [--output <output_size>] [--iter <iter>]" << std::endl;
            return 0;
        }
    }

    std::cout << "B: " << B << ", D: " << D << ", F: " << F << ", iter: " << iter << std::endl;


    uint16_t *X;
    __nv_bfloat16 *X_dev;
    uint16_t *W_1[iter+warmup_iter];
    uint16_t *W_2[iter+warmup_iter];
    uint16_t *W_3[iter+warmup_iter];
    __nv_bfloat16 *W_1_dev[iter+warmup_iter];
    __nv_bfloat16 *W_2_dev[iter+warmup_iter];
    __nv_bfloat16 *W_3_dev[iter+warmup_iter];
    __nv_bfloat16 *gate;
    __nv_bfloat16 *up;
    __nv_bfloat16 *h;
    __nv_bfloat16 *out;

    X = (uint16_t*)malloc(B * D * sizeof(uint16_t));
    random_bf16(X, B * D, 0.02f, 42);
    CUDA_CHECK(cudaMalloc(&X_dev, B * D * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(X_dev, X, B * D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    for (int i = 0; i < iter+warmup_iter; i++)
    {
        W_1[i] = (uint16_t*)malloc(D * F * sizeof(uint16_t));
        random_bf16(W_1[i], D * F, 0.02f, 43);
        CUDA_CHECK(cudaMalloc(&W_1_dev[i], D * F * sizeof(__nv_bfloat16)));

        W_2[i] = (uint16_t*)malloc(D * F * sizeof(uint16_t));
        random_bf16(W_2[i], D * F, 0.02f, 43);
        CUDA_CHECK(cudaMalloc(&W_2_dev[i], D * F * sizeof(__nv_bfloat16)));

        W_3[i] = (uint16_t*)malloc(D * F * sizeof(uint16_t));
        random_bf16(W_3[i], D * F, 0.02f, 43);
        CUDA_CHECK(cudaMalloc(&W_3_dev[i], D * F * sizeof(__nv_bfloat16)));
    }


    CUDA_CHECK(cudaMalloc(&gate, (size_t)B * F * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&up, (size_t)B * F * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&h, (size_t)B * F * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&out, (size_t)B * D * sizeof(__nv_bfloat16)));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    char str [20];



    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < warmup_iter + iter; i++)
    {   
        if (i == warmup_iter)
        {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        sprintf(str, "Iter %d", i);
        nvtxRangePush(str);

        nvtxRangePush("H2D");
        CUDA_CHECK(cudaMemcpy(W_1_dev[i], W_1[i], D * F * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(W_2_dev[i], W_2[i], D * F * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(W_3_dev[i], W_3[i], D * F * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        nvtxRangePop();
        

        nvtxRangePush("Compute");
        // GEMM: gate = X @ W1^T
        gemm_rowmajor_bf16(handle, X_dev, W_1_dev[i], gate, B, D, F);
        // bool ok = verify_bf16_gemm_cpu_sample(X_dev, W_1_dev, gate, B, D, F,
        //                                   /*num_samples=*/10,
        //                                   /*atol=*/5e-2f, /*rtol=*/5e-2f,
        //                                   /*seed=*/1234ULL);
        // if (!ok) {
        //     std::cout << "bf16_gemm failed" << std::endl;
        // }

        // GEMM: up = X @ W3^T
        gemm_rowmajor_bf16(handle, X_dev, W_3_dev[i], up, B, D, F);
        // ok = verify_bf16_gemm_cpu_sample(X_dev, W_3_dev, up, B, D, F,
        //                                   /*num_samples=*/10,
        //                                   /*atol=*/5e-2f, /*rtol=*/5e-2f,
        //                                   /*seed=*/1235ULL);
        // if (!ok) {
        //     std::cout << "bf16_gemm failed" << std::endl;
        // }


        // h = (SiLU(gate) ⊙ up) → [B, d_ff_new]
        launch_silu_mul_bf16(gate, up, h, B, F);


        // out = h @ W2ᵀ → [B, d_model]
        gemm_rowmajor_bf16(handle, h, W_2_dev[i], out, B, F, D);
        // ok = verify_bf16_gemm_cpu_sample(h, W_2_dev, out, B, F, D,
        //                                   /*num_samples=*/10,
        //                                   /*atol=*/5e-2f, /*rtol=*/5e-2f,
        //                                   /*seed=*/1235ULL);
        // if (!ok) {
        //     std::cout << "bf16_gemm failed" << std::endl;
        // }
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePop();

        if (i < 5)
        {
            /* code */
        }
        
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
    float latency_per_iter = elapsed.count() / iter;
    std::cout << "Average time per iteration: " << latency_per_iter << " us" << " (" << iter << "iters)" << std::endl;
    
    

    free(X);
    cudaFree(X_dev);        
    for (int i = 0; i < iter+warmup_iter; i++)
    {
        free(W_1[i]);
        free(W_2[i]);
        free(W_3[i]);

        cudaFree(W_1_dev[i]);
        cudaFree(W_2_dev[i]);
        cudaFree(W_3_dev[i]);
    }
    
    
    cudaFree(gate);
    cudaFree(up);
    cudaFree(h);
    cudaFree(out);

    CUBLAS_CHECK(cublasDestroy(handle));




    /* FP32 VERSION */


    // float* X = (float*)malloc(B * D * sizeof(float));
    // random_float(X, B * D);
    // float* X_dev;
    // cudaMalloc((void**)&X_dev, B * D * sizeof(float));
    // cudaMemcpy(X_dev, X, B * D * sizeof(float), cudaMemcpyHostToDevice);




    // float* W_1 = (float*)malloc(D * F * sizeof(float));
    // random_float(W_1, D * F);
    // float* W_1_dev;
    // cudaMalloc((void**)&W_1_dev, D * F * sizeof(float));
    // cudaMemcpy(W_1_dev, W_1, D * F * sizeof(float), cudaMemcpyHostToDevice);

    // float* W_2 = (float*)malloc(D * F * sizeof(float));
    // random_float(W_2, D * F);
    // float* W_2_dev;
    // cudaMalloc((void**)&W_2_dev, D * F * sizeof(float));
    // cudaMemcpy(W_2_dev, W_2, D * F * sizeof(float), cudaMemcpyHostToDevice);

    // float* W_3 = (float*)malloc(D * F * sizeof(float));
    // random_float(W_3, D * F);
    // float* W_3_dev;
    // cudaMalloc((void**)&W_3_dev, D * F * sizeof(float));
    // cudaMemcpy(W_3_dev, W_3, D * F * sizeof(float), cudaMemcpyHostToDevice);


    // float* gate;
    // cudaMalloc((void**)&gate, B * F * sizeof(float));

    // float* up;
    // cudaMalloc((void**)&up, B * F * sizeof(float));
    
    // float* h;
    // cudaMalloc((void**)&h, B * F * sizeof(float));

    // float* out;
    // cudaMalloc((void**)&out, B * D * sizeof(float));



    // cublasHandle_t handle = nullptr;
    // CUBLAS_CHECK(cublasCreate(&handle));


    // // gate = x @ W1ᵀ → [B, d_ff_new]
    // gemm_rowmajor_fp32(handle, X_dev, W_1_dev, gate, B, D, F);
    // // bool ok = verify_gemm_fp32(X_dev, W_1_dev, gate, B, D, F);
    // // if (!ok) {
    // //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // // } 

    // // up = x @ W3ᵀ → [B, d_ff_new]
    // gemm_rowmajor_fp32(handle, X_dev, W_3_dev, up, B, D, F);
    // // ok = verify_gemm_fp32(X_dev, W_3_dev, up, B, D, F);
    // // if (!ok) {
    // //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // // } 

    // // h = (SiLU(gate) ⊙ up) → [B, d_ff_new]
    // launch_silu_mul_fp32(gate, up, h, B, F);


    // // out = h @ W2ᵀ → [B, d_model]
    // gemm_rowmajor_fp32(handle, h, W_2_dev, out, B, F, D);
    // // ok = verify_gemm_fp32(h, W_2_dev, out, B, F, D);
    // // if (!ok) {
    // //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // // } 

    // cudaDeviceSynchronize();


    // free(X);
    // free(W_1);
    // free(W_2);
    // free(W_3);
    // cudaFree(X_dev);
    // cudaFree(W_1_dev);
    // cudaFree(W_2_dev);
    // cudaFree(W_3_dev);
    // cudaFree(gate);
    // cudaFree(up);
    // cudaFree(h);
    // cudaFree(out);
    // CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}