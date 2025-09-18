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


    float* X = (float*)malloc(B * D * sizeof(float));
    random_float(X, B * D);
    float* X_dev;
    cudaMalloc((void**)&X_dev, B * D * sizeof(float));
    cudaMemcpy(X_dev, X, B * D * sizeof(float), cudaMemcpyHostToDevice);




    float* W_1 = (float*)malloc(D * F * sizeof(float));
    random_float(W_1, D * F);
    float* W_1_dev;
    cudaMalloc((void**)&W_1_dev, D * F * sizeof(float));
    cudaMemcpy(W_1_dev, W_1, D * F * sizeof(float), cudaMemcpyHostToDevice);

    float* W_2 = (float*)malloc(D * F * sizeof(float));
    random_float(W_2, D * F);
    float* W_2_dev;
    cudaMalloc((void**)&W_2_dev, D * F * sizeof(float));
    cudaMemcpy(W_2_dev, W_2, D * F * sizeof(float), cudaMemcpyHostToDevice);

    float* W_3 = (float*)malloc(D * F * sizeof(float));
    random_float(W_3, D * F);
    float* W_3_dev;
    cudaMalloc((void**)&W_3_dev, D * F * sizeof(float));
    cudaMemcpy(W_3_dev, W_3, D * F * sizeof(float), cudaMemcpyHostToDevice);


    float* gate;
    cudaMalloc((void**)&gate, B * F * sizeof(float));

    float* up;
    cudaMalloc((void**)&up, B * F * sizeof(float));
    
    float* h;
    cudaMalloc((void**)&h, B * F * sizeof(float));

    float* out;
    cudaMalloc((void**)&out, B * D * sizeof(float));



    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));


    // gate = x @ W1ᵀ → [B, d_ff_new]
    gemm_rowmajor_fp32(handle, X_dev, W_1_dev, gate, B, D, F);
    // bool ok = verify_gemm_fp32(X_dev, W_1_dev, gate, B, D, F);
    // if (!ok) {
    //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // } 

    // up = x @ W3ᵀ → [B, d_ff_new]
    gemm_rowmajor_fp32(handle, X_dev, W_3_dev, up, B, D, F);
    // ok = verify_gemm_fp32(X_dev, W_3_dev, up, B, D, F);
    // if (!ok) {
    //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // } 

    // h = (SiLU(gate) ⊙ up) → [B, d_ff_new]
    launch_silu_mul_fp32(gate, up, h, B, F);


    // out = h @ W2ᵀ → [B, d_model]
    gemm_rowmajor_fp32(handle, h, W_2_dev, out, B, F, D);
    // ok = verify_gemm_fp32(h, W_2_dev, out, B, F, D);
    // if (!ok) {
    //     std::cout << "gate_gemm_fp32 failed" << std::endl;
    // } 

    cudaDeviceSynchronize();


    free(X);
    free(W_1);
    free(W_2);
    free(W_3);
    cudaFree(X_dev);
    cudaFree(W_1_dev);
    cudaFree(W_2_dev);
    cudaFree(W_3_dev);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(h);
    cudaFree(out);
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}