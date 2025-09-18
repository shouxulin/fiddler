#include <cublas_v2.h>
#include <cuda_bf16.h>   // for __nv_bfloat16 (CUDA 11+)
#include <stdexcept>

#include <cstring>
#include <algorithm>
#include <omp.h>


#include <cstdio>
#include <cstdlib>


// random initialize a buffer of float
void random_float(float* buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

enum AllocationType {
    HOST,
    DEVICE,
    SYSTEM
};

void myAlloc(void** ptr, size_t size, AllocationType type) {
    if (type == HOST) {
        cudaMallocHost(ptr, size);
    } else if (type == DEVICE) {
        cudaMalloc(ptr, size);
    } else if (type == SYSTEM) {
        *ptr = malloc(size);
    }
}

void myFree(void* ptr, AllocationType type) {
    if (type == HOST) {
        cudaFreeHost(ptr);
    } else if (type == DEVICE) {
        cudaFree(ptr);
    } else if (type == SYSTEM) {
        free(ptr);
    }
}



#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)


inline void CUBLAS_CHECK(cublasStatus_t st) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error");
}

// Compute: Y = X @ W1^T
// X:  [B, D] row-major
// W1: [F, D] row-major
// Y:  [B, F] row-major
// All pointers are device pointers.
void gemm_fp32(cublasHandle_t handle,
                    const float* X, const float* W1, float* Y,
                    int B, int D, int F)
{
    // Column-major GEMM parameters to realize Y = X * W1^T
    // We compute C_col (F x B) = A_col (D x F)^T * B_col (D x B),
    // Mapping row-major -> column-major:
    //   W1_rm [F x D] -> W1_cm [D x F] with lda = D
    //   X_rm  [B x D] -> X_cm  [D x B] with ldb = D
    //   Y_rm  [B x F] -> C_cm  [F x B] with ldc = F  (same memory!)
    const int m = F;       // rows of C (in column-major)
    const int n = B;       // cols of C
    const int k = D;       // inner dimension

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // A = W1_cm with op(A)=T  (so effective F x D)
    const int lda = D;
    // B = X_cm with op(B)=N   (D x B)
    const int ldb = D;
    // C = Y^T as column-major (F x B)
    const int ldc = F;

    // Enable TF32 on Ampere+ if you want more speed (optional):
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,            // op(A): W1_cm^T -> (F x D)
        CUBLAS_OP_N,            // op(B): X_cm    -> (D x B)
        m, n, k,
        &alpha,
        W1, CUDA_R_32F, lda,    // A
        X,  CUDA_R_32F, ldb,    // B
        &beta,
        Y,  CUDA_R_32F, ldc,    // C (writes row-major [B x F])
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// FP32 -> BF16 cast (GPU)
__global__ void k_f32_to_bf16(const float* __restrict__ in,
                              __nv_bfloat16* __restrict__ out,
                              size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}


// Row-major BF16 GEMM: Y = X @ W^T
// X_rm: [B, in_dim], W_rm: [out_dim, in_dim], Y_rm: [B, out_dim]
// Internally: interpret as column-major by swapping axes.
inline void gemm_rowmajor_bf16(cublasHandle_t handle,
                               const __nv_bfloat16* dX,   // [B, in_dim] (rm)
                               const __nv_bfloat16* dW,   // [out_dim, in_dim] (rm)
                               __nv_bfloat16* dY,         // [B, out_dim] (rm)
                               int B, int in_dim, int out_dim,
                               float alpha = 1.0f, float beta = 0.0f)
{
    // Map row-major to column-major:
    //   X_rm[B x in]  -> X_cm[in x B]   (ldb = in_dim)
    //   W_rm[out x in]-> W_cm[in x out] (lda = in_dim)
    //   Y_rm[B x out] -> C_cm[out x B]  (ldc = out_dim)
    const int m   = out_dim;     // rows of C (cm)
    const int n   = B;           // cols of C
    const int k   = in_dim;      // inner dim
    const int lda = in_dim;      // leading dim A
    const int ldb = in_dim;      // leading dim B
    const int ldc = out_dim;     // leading dim C

    // Inputs/Output: BF16, Accumulation: FP32
    // Note: alpha/beta pointers must be of the compute type (float here).
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                  // A = W_cm^T -> [out x in]
        CUBLAS_OP_N,                  // B = X_cm    -> [in  x B ]
        m, n, k,
        &alpha,
        dW, CUDA_R_16BF, lda,         // A
        dX, CUDA_R_16BF, ldb,         // B
        &beta,
        dY, CUDA_R_16BF, ldc,         // C
        CUBLAS_COMPUTE_32F,           // accumulate in FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP // use tensor cores
    ));
}


// Computes: Y = X @ W^T
// Row-major shapes:
//   X  : [B, in_dim]
//   W  : [out_dim, in_dim]
//   Y  : [B, out_dim]
inline void gemm_rowmajor_fp32(cublasHandle_t handle,
                                    const float* dX,           // [B, in_dim]
                                    const float* dW,           // [out_dim, in_dim]
                                    float* dY,                 // [B, out_dim]
                                    int B, int in_dim, int out_dim,
                                    float alpha = 1.0f, float beta = 0.0f)
{
    // Interpret row-major buffers as column-major by swapping axes:
    //   X_rm [B x in]   -> X_cm [in x B]   with ldb = in_dim
    //   W_rm [out x in] -> W_cm [in x out] with lda = in_dim
    //   Y_rm [B x out]  -> C_cm [out x B]  with ldc = out_dim
    const int m = out_dim;     // rows of C (col-major)
    const int n = B;           // cols of C
    const int k = in_dim;      // inner dimension
    const int lda = in_dim;    // leading dim of A (W_cm)
    const int ldb = in_dim;    // leading dim of B (X_cm)
    const int ldc = out_dim;   // leading dim of C (Y_cm)

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,                  // op(A): W_cm^T -> (out_dim x in_dim)
        CUBLAS_OP_N,                  // op(B): X_cm    -> (in_dim x B)
        m, n, k,
        &alpha,
        dW, CUDA_R_32F, lda,          // A = W
        dX, CUDA_R_32F, ldb,          // B = X
        &beta,
        dY, CUDA_R_32F, ldc,          // C = Y (row-major [B x out_dim])
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


// // Row-major shapes:
// // X:  [B, D]
// // W1: [F, D]
// // Y:  [B, F]
// bool verify_gemm_fp32(const float* dX,
//                            const float* dW1,
//                            const float* dY,
//                            int B, int D, int F,
//                            float atol = 1e-4f,
//                            float rtol = 1e-5f)
// {
//     // Pull device buffers to host
//     std::vector<float> X(B * D), W1(F * D), Y(B * F);
//     CUDA_CHECK(cudaMemcpy(X.data(),  dX,  sizeof(float) * B * D, cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(W1.data(), dW1, sizeof(float) * F * D, cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(Y.data(),  dY,  sizeof(float) * B * F, cudaMemcpyDeviceToHost));

//     // Build reference: Y_ref = X @ W1^T
//     std::vector<float> Yref(B * F, 0.0f);
//     for (int i = 0; i < B; ++i) {
//         const float* Xi = &X[i * D];
//         for (int f = 0; f < F; ++f) {
//             const float* W1f = &W1[f * D];              // row f of W1 (len D)
//             double acc = 0.0;                           // use double for a tiny bit more stability
//             for (int d = 0; d < D; ++d) {
//                 acc += static_cast<double>(Xi[d]) * static_cast<double>(W1f[d]);
//             }
//             Yref[i * F + f] = static_cast<float>(acc);
//         }
//     }

//     // Compare
//     double max_abs_err = 0.0, sum_sq = 0.0, sum_sq_ref = 0.0;
//     size_t N = static_cast<size_t>(B) * F;
//     for (size_t k = 0; k < N; ++k) {
//         double ref = static_cast<double>(Yref[k]);
//         double got = static_cast<double>(Y[k]);
//         double abs_err = std::fabs(ref - got);
//         max_abs_err = std::max(max_abs_err, abs_err);
//         sum_sq += abs_err * abs_err;
//         sum_sq_ref += ref * ref;
//         // Optional per-element tolerance check (short-circuit)
//         double thr = static_cast<double>(atol) + static_cast<double>(rtol) * std::fabs(ref);
//         if (abs_err > thr) {
//             // Print first failing element to help debug
//             std::fprintf(stderr,
//                          "Mismatch at k=%zu: ref=%.7g got=%.7g abs_err=%.3g (thr=%.3g)\n",
//                          k, ref, got, abs_err, thr);
//             // continue scanning to report full stats
//         }
//     }
//     double rms_err = std::sqrt(sum_sq / std::max<size_t>(1, N));
//     double rel_rms = rms_err / (std::sqrt(sum_sq_ref / std::max<size_t>(1, N)) + 1e-30);

//     // std::printf("[verify fp32] B=%d D=%d F=%d  max_abs=%.3g  rms=%.3g  rel_rms=%.3g\n",
//     //             B, D, F, max_abs_err, rms_err, rel_rms);

//     // Pass criterion: elementwise within atol + rtol*|ref|
//     // (If you prefer a pure norm-based check, use rel_rms + max_abs.)
//     bool ok = true;
//     for (size_t k = 0; k < N; ++k) {
//         double ref = static_cast<double>(Yref[k]);
//         double got = static_cast<double>(Y[k]);
//         double abs_err = std::fabs(ref - got);
//         double thr = static_cast<double>(atol) + static_cast<double>(rtol) * std::fabs(ref);
//         if (abs_err > thr) { ok = false; break; }
//     }
//     return ok;
// }


bool verify_gemm_fp32(const float* dX,
                      const float* dW,
                      const float* dY,
                      int B, int in_dim, int out_dim,
                      float tol = 1e-4f)
{
    // Copy device data back
    size_t nX = (size_t)B * in_dim;
    size_t nW = (size_t)out_dim * in_dim;
    size_t nY = (size_t)B * out_dim;

    std::vector<float> X(nX), W(nW), Y(nY);
    cudaMemcpy(X.data(), dX, nX*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W.data(), dW, nW*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y.data(), dY, nY*sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference
    for (int b = 0; b < B; b++) {
        for (int o = 0; o < out_dim; o++) {
            double acc = 0.0;
            for (int k = 0; k < in_dim; k++) {
                acc += (double)X[b*in_dim + k] * (double)W[o*in_dim + k];
            }
            float ref = (float)acc;
            float got = Y[b*out_dim + o];
            if (fabs(ref - got) > tol * std::max(1.0f, fabs(ref))) {
                printf("Mismatch at (b=%d,o=%d): ref=%f got=%f\n",
                       b, o, ref, got);
                return false;
            }
        }
    }
    printf("verify_gemm_fp32 passed (B=%d in=%d out=%d)\n", B, in_dim, out_dim);
    return true;
}



__device__ inline float sigmoidf_fast(float x) {
    // stable + fast: sigmoid(x) = 0.5 * (tanh(0.5*x) + 1)
    return 0.5f * (tanhf(0.5f * x) + 1.0f);
}

// gate, up, h are all length N = B*F (row-major [B,F] flattened)
__global__ void silu_mul_fp32_kernel(const float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     float* __restrict__ h,
                                     int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    float g = gate[idx];
    float u = up[idx];
    float out = (g * sigmoidf_fast(g)) * u;   // SiLU(g) * up
    h[idx] = out;
}


inline void launch_silu_mul_fp32(const float* d_gate,
                                 const float* d_up,
                                 float* d_h,
                                 int B, int F,
                                 cudaStream_t stream=0)
{
    const int N = B * F;
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    silu_mul_fp32_kernel<<<grid, block, 0, stream>>>(d_gate, d_up, d_h, N);
}