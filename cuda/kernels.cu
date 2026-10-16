// ============================================================
// trba-rust — CUDA backend
//
// Every symbol the Rust side links against is `extern "C"`.  The CPU
// fallback (cpu_backend.cpp) implements exactly the same set, so the
// whole pipeline can be built and unit-tested on a machine with no GPU.
//
// Why this file looks the way it does
// -----------------------------------
//  * No kernel materialises an im2col buffer.  For a 3x3 conv on
//    [B=256, C=512, H=4, W=16] the col matrix is
//    B*Ho*Wo*C*kH*kW*4B = 38.6 GB.  The previous version asked
//    cudaMalloc for it anyway; the allocation failed, CHECK_CUDA only
//    *printed* the error, and the next kernel dereferenced a null device
//    pointer -> "an illegal memory access was encountered" (reported at
//    the following cudaMalloc, kernels.cu:112).
//    grad_W, grad_bias and grad_input are now three fused kernels that
//    read the activation / grad_output tensors directly.
//  * Every launch uses a 1-D grid-stride loop.  gridDim.y/z are capped at
//    65535, and the old gemm_fp32 used dim3 grid((N+15)/16,(M+15)/16),
//    which silently fails to launch as soon as M > 1 048 560.
//  * CHECK_CUDA is fatal.  A CUDA error is sticky: continuing after one
//    only produces nonsense and a crash somewhere unrelated.
//  * Host pointers are never handed to a kernel (the old maxpool backward
//    did exactly that).
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "\n[CUDA FATAL] %s:%d: %s\n              call: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err__), #call);     \
            fflush(stderr);                                                    \
            abort();                                                           \
        }                                                                      \
    } while (0)

// Launch errors are asynchronous — catch them right after the launch so the
// message names the offending kernel instead of some later unrelated call.
#define CHECK_LAUNCH() CHECK_CUDA(cudaGetLastError())

// Upper bound for a 1-D grid-stride launch.
static const int kMaxGrid = 262144;

static inline int grid_for(long long work, int threads) {
    long long g = (work + threads - 1) / threads;
    if (g < 1) g = 1;
    if (g > kMaxGrid) g = kMaxGrid;
    return (int)g;
}

// ------------------------------------------------------------
// Device properties (cached)
// ------------------------------------------------------------
static const cudaDeviceProp &dev_prop() {
    static cudaDeviceProp prop;
    static bool init = false;
    if (!init) {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
        init = true;
    }
    return prop;
}

// ============================================================
// INT8 GEMM   C[M,N](int32) = A[M,K](int8) * Bt[N,K](int8)^T
// Bt is the weight matrix already transposed to [N,K] row-major, which is
// exactly what wmma::col_major expects.
//
// Operand layout contract (the Rust side pads to this):
//   A  : [align16(M), align16(K)] row-major, zero padded
//   Bt : [align16(N), align16(K)] row-major, zero padded
//   C  : [M, N]                   row-major, *not* padded
// M, N and K themselves may be arbitrary — the kernel works on the padded
// extents and clips on store.  The previous version stored whole 16x16
// accumulator tiles with leading dimension N, which ran off the end of C
// whenever N % 16 != 0.
//
// NOTE on guards: `#if defined(__CUDA_ARCH__)` must never be used around code
// the *host* side also has to see — nvcc runs a host pass with __CUDA_ARCH__
// undefined, so a host launcher wrapped in it silently compiles the other
// branch.  Availability is decided here by the toolkit version (compile time)
// and the dispatch by the device compute capability (run time).
// ============================================================
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 10)
#define TRBA_WMMA_AVAILABLE 1
#include <mma.h>
using namespace nvcuda;
#endif

#ifdef TRBA_WMMA_AVAILABLE
__global__ void gemm_int8_wmma_kernel(const int8_t *__restrict__ A,
                                      const int8_t *__restrict__ Bt,
                                      int32_t *__restrict__ C, int M, int N,
                                      int K, int Mp, int Np, int Kp) {
    constexpr int BM = 64, BN = 64, BK = 32;
    // 4 warps as 2x2 -> each warp produces a 32x32 output patch.
    const int warp_id = threadIdx.x >> 5;
    const int warp_m = (warp_id & 1) * 32;
    const int warp_n = (warp_id >> 1) * 32;

    __shared__ int8_t smem_a[BM][BK];
    __shared__ int8_t smem_b[BN][BK];
    __shared__ int32_t stage[BM][BN];  // clip-on-store buffer

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag[4];

    const int tiles_x = Np / BN;
    const int total_tiles = tiles_x * (Mp / BM);

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int tile_y = tile / tiles_x;
        const int tile_x = tile - tile_y * tiles_x;
        const int row0 = tile_y * BM;
        const int col0 = tile_x * BN;

#pragma unroll
        for (int i = 0; i < 4; ++i) wmma::fill_fragment(c_frag[i], 0);

        for (int k0 = 0; k0 < Kp; k0 += BK) {
            // 64 x 32 int8 = 2048 B = 512 ints; 128 threads x 4 ints each.
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int idx = threadIdx.x + i * 128;
                int flat = idx * 4;
                int r = flat / BK, c = flat - r * BK;
                int v = 0;
                if (k0 + c < Kp) {
                    v = *reinterpret_cast<const int *>(
                        A + (long long)(row0 + r) * Kp + k0 + c);
                }
                reinterpret_cast<int *>(smem_a)[idx] = v;
                int v2 = 0;
                if (k0 + c < Kp) {
                    v2 = *reinterpret_cast<const int *>(
                        Bt + (long long)(col0 + r) * Kp + k0 + c);
                }
                reinterpret_cast<int *>(smem_b)[idx] = v2;
            }
            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; k += 16) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    wmma::load_matrix_sync(a_frag, &smem_a[warp_m + mi * 16][k], BK);
#pragma unroll
                    for (int ni = 0; ni < 2; ++ni) {
                        wmma::load_matrix_sync(b_frag, &smem_b[warp_n + ni * 16][k], BK);
                        wmma::mma_sync(c_frag[mi * 2 + ni], a_frag, b_frag,
                                       c_frag[mi * 2 + ni]);
                    }
                }
            }
            __syncthreads();
        }

        // Stage the whole 64x64 tile in shared memory, then copy only the
        // in-bounds part out to C.  Each warp writes its own 32x32 quadrant,
        // so no extra synchronisation is needed before the copy loop.
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                wmma::store_matrix_sync(&stage[warp_m + mi * 16][warp_n + ni * 16],
                                        c_frag[mi * 2 + ni], BN,
                                        wmma::mem_row_major);
            }
        __syncthreads();

        // 128 threads copy 64x64 = 4096 ints -> 32 each, 4 rows x 8 cols.
        for (int i = threadIdx.x; i < BM * BN / 4; i += 128) {
            int r = i / (BN / 4);
            int c = (i - r * (BN / 4)) * 4;
            int gm = row0 + r, gn = col0 + c;
            if (gm < M && gn < N) C[(long long)gm * N + gn] = stage[r][c];
            if (gm < M && gn + 1 < N) C[(long long)gm * N + gn + 1] = stage[r][c + 1];
            if (gm < M && gn + 2 < N) C[(long long)gm * N + gn + 2] = stage[r][c + 2];
            if (gm < M && gn + 3 < N) C[(long long)gm * N + gn + 3] = stage[r][c + 3];
        }
        __syncthreads();
    }
}
#endif  // TRBA_WMMA_AVAILABLE (int8)

// DP4A path: keeps the file compilable (and correct, just slower) for
// pre-Turing targets where s8s8s32 wmma does not exist.  1-D grid-stride, so
// it has no gridDim.y limit either.
__global__ void gemm_int8_dp4a_kernel(const int8_t *__restrict__ A,
                                      const int8_t *__restrict__ Bt,
                                      int32_t *__restrict__ C, int M, int N,
                                      int K, int Kp) {
    const long long total = (long long)M * N;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += (long long)gridDim.x * blockDim.x) {
        const int m = (int)(t / N);
        const int n = (int)(t - (long long)m * N);
        const int8_t *a = A + (long long)m * Kp;
        const int8_t *b = Bt + (long long)n * Kp;
        int32_t acc = 0;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            int av, bv;
            memcpy(&av, a + k, 4);
            memcpy(&bv, b + k, 4);
            acc = __dp4a(av, bv, acc);
        }
        for (; k < K; ++k) acc += (int32_t)a[k] * (int32_t)b[k];
        C[t] = acc;
    }
}

extern "C" void launch_gemm_int8(const int8_t *A, const int8_t *B, int32_t *C,
                                 int M, int N, int K, void *stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int Mp = align16(M), Np = align16(N), Kp = align16(K);
    const cudaDeviceProp &p = dev_prop();
    const bool tensor_s8 = (p.major > 7) || (p.major == 7 && p.minor >= 2);
#ifdef TRBA_WMMA_AVAILABLE
    if (tensor_s8) {
        // At least one block: M or N below 64 gives zero whole tiles, and a
        // zero-block launch would silently leave C untouched.  The kernel walks
        // the padded extents and clips on store, so partial tiles are fine.
        long long tiles = std::max<long long>((long long)(Mp / 64) * (Np / 64), 1);
        int blocks = (int)std::min<long long>(tiles, 8192);
        gemm_int8_wmma_kernel<<<blocks, 128, 0, (cudaStream_t)stream>>>(
            A, B, C, M, N, K, Mp, Np, Kp);
        CHECK_LAUNCH();
        return;
    }
#endif
    long long work = (long long)M * N;
    gemm_int8_dp4a_kernel<<<grid_for(work, 256), 256, 0, (cudaStream_t)stream>>>(
        A, B, C, M, N, K, Kp);
    CHECK_LAUNCH();
}

// ============================================================
// FP16 GEMM   C[M,N](f32) = A[M,K](f16) * Bt[N,K](f16)^T
// Same operand contract as the INT8 GEMM: A is [align16(M), align16(K)],
// Bt is [align16(N), align16(K)], C is exactly [M, N].
// ============================================================
#ifdef TRBA_WMMA_AVAILABLE

__global__ void gemm_fp16_wmma_kernel(const half *__restrict__ A,
                                      const half *__restrict__ Bt,
                                      float *__restrict__ C, int M, int N,
                                      int K, int Mp, int Np, int Kp) {
    constexpr int BM = 64, BN = 64, BK = 32;
    const int warp_id = threadIdx.x >> 5;
    const int warp_m = (warp_id & 1) * 32;
    const int warp_n = (warp_id >> 1) * 32;

    __shared__ half smem_a[BM][BK];
    __shared__ half smem_b[BN][BK];
    __shared__ float stage[BM][BN];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4];

    const int tiles_x = Np / BN;
    const int total_tiles = tiles_x * (Mp / BM);

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int tile_y = tile / tiles_x;
        const int tile_x = tile - tile_y * tiles_x;
        const int row0 = tile_y * BM;
        const int col0 = tile_x * BN;

#pragma unroll
        for (int i = 0; i < 4; ++i) wmma::fill_fragment(c_frag[i], 0.0f);

        for (int k0 = 0; k0 < Kp; k0 += BK) {
            // 64 x 32 halves = 4096 B = 256 float4; 128 threads x 2 each.
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int idx = threadIdx.x + i * 128;
                int flat = idx * 8;                 // in halves
                int r = flat / BK, c = flat - r * BK;
                reinterpret_cast<float4 *>(smem_a)[idx] = *reinterpret_cast<const float4 *>(
                    A + (long long)(row0 + r) * Kp + k0 + c);
                reinterpret_cast<float4 *>(smem_b)[idx] = *reinterpret_cast<const float4 *>(
                    Bt + (long long)(col0 + r) * Kp + k0 + c);
            }
            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; k += 16) {
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    wmma::load_matrix_sync(a_frag, &smem_a[warp_m + mi * 16][k], BK);
#pragma unroll
                    for (int ni = 0; ni < 2; ++ni) {
                        wmma::load_matrix_sync(b_frag, &smem_b[warp_n + ni * 16][k], BK);
                        wmma::mma_sync(c_frag[mi * 2 + ni], a_frag, b_frag,
                                       c_frag[mi * 2 + ni]);
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
                wmma::store_matrix_sync(&stage[warp_m + mi * 16][warp_n + ni * 16],
                                        c_frag[mi * 2 + ni], BN, wmma::mem_row_major);
        __syncthreads();

        for (int i = threadIdx.x; i < BM * BN / 4; i += 128) {
            int r = i / (BN / 4);
            int c = (i - r * (BN / 4)) * 4;
            int gm = row0 + r, gn = col0 + c;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (gm < M && gn + j < N)
                    C[(long long)gm * N + gn + j] = stage[r][c + j];
        }
        __syncthreads();
    }
}
#endif  // TRBA_WMMA_AVAILABLE (fp16)

extern "C" void launch_gemm_fp16(const void *A, const void *B, float *C, int M,
                                 int N, int K, void *stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int Mp = align16(M), Np = align16(N), Kp = align16(K);
#ifdef TRBA_WMMA_AVAILABLE
    const cudaDeviceProp &p = dev_prop();
    if (p.major >= 6 || (p.major == 5 && p.minor >= 3)) {
        // At least one block: M or N below 64 gives zero whole tiles, and a
        // zero-block launch would silently leave C untouched.  The kernel walks
        // the padded extents and clips on store, so partial tiles are fine.
        long long tiles = std::max<long long>((long long)(Mp / 64) * (Np / 64), 1);
        int blocks = (int)std::min<long long>(tiles, 8192);
        gemm_fp16_wmma_kernel<<<blocks, 128, 0, (cudaStream_t)stream>>>(
            reinterpret_cast<const half *>(A), reinterpret_cast<const half *>(B), C,
            M, N, K, Mp, Np, Kp);
        CHECK_LAUNCH();
        return;
    }
#endif
    fprintf(stderr, "[CUDA FATAL] launch_gemm_fp16: this device has no fp16 tensor cores\n");
    abort();
}

// ============================================================
// FP32 GEMM   C[M,N] = A[M,K] * B[K,N]
// Shared-memory tiled.  1-D grid-stride over 32x32 output tiles, so any
// M/N/K works (the old dim3 grid blew past the 65535 gridDim.y limit).
// Block = (32,8) = 256 threads; each thread owns 4 rows x 1 column.
// ============================================================
#define FT 32
__global__ void gemm_fp32_tiled(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, int M, int N, int K) {
    __shared__ float As[FT][FT];
    __shared__ float Bs[FT][FT];

    const int tx = threadIdx.x;  // 0..31 -> column inside the tile
    const int ty = threadIdx.y;  // 0..7  -> row group (4 rows each)

    const int tiles_x = (N + FT - 1) / FT;
    const int tiles_y = (M + FT - 1) / FT;
    const int total = tiles_x * tiles_y;

    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int t_y = tile / tiles_x;
        const int t_x = tile - t_y * tiles_x;
        const int row0 = t_y * FT;
        const int col0 = t_x * FT;

        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        for (int k0 = 0; k0 < K; k0 += FT) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int ar = ty * 4 + i;
                int ga = row0 + ar, ka = k0 + tx;
                As[ar][tx] = (ga < M && ka < K) ? A[(long long)ga * K + ka] : 0.f;
                int kb = k0 + ar, nb = col0 + tx;
                Bs[ar][tx] = (kb < K && nb < N) ? B[(long long)kb * N + nb] : 0.f;
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a0 = As[ty * 4 + i][tx];
#pragma unroll 8
                for (int k = 0; k < FT; ++k) acc[i] += a0 * Bs[k][tx];
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int gm = row0 + ty * 4 + i;
            int gn = col0 + tx;
            if (gm < M && gn < N) C[(long long)gm * N + gn] = acc[i];
        }
    }
}
#undef FT

extern "C" void launch_gemm_fp32(const float *A, const float *B, float *C,
                                 int M, int N, int K, void *stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    long long tiles = (long long)((M + 31) / 32) * ((N + 31) / 32);
    dim3 thr(32, 8);
    gemm_fp32_tiled<<<grid_for(tiles, 1), thr, 0, (cudaStream_t)stream>>>(A, B, C, M, N, K);
    CHECK_LAUNCH();
}

// ============================================================
// Conv epilogue: int32 GEMM result -> dequant -> +bias -> ReLU -> requant int8
//   out[b,oc,oh,ow] = clamp(round(max(acc[row, oc]/acc_div + bias[oc], 0) * out_mul))
// acc is [B*Ho*Wo, acc_ld]; the useful columns are the first `C`.
// ============================================================
__global__ void conv_epilogue_kernel(const int32_t *__restrict__ acc,
                                     const float *__restrict__ bias,
                                     int8_t *__restrict__ out, int acc_ld,
                                     float acc_div, float out_mul, int C, int Ho,
                                     int Wo, long long total) {
    // out is [B, C, Ho, Wo] (NCHW); acc is [B*Ho*Wo, C] row-major with
    // leading dimension acc_ld >= C.  Deriving oc from i % C would treat the
    // output as NHWC and read the accumulator with the wrong strides.
    const long long hw = (long long)Ho * Wo;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        // out is [B, C, Ho, Wo] (NCHW); acc is [B*Ho*Wo, C] row-major with
        // leading dimension acc_ld >= C.
        long long b = i / ((long long)C * hw);
        int oc = (int)((i / hw) % C);
        long long row = b * hw + (i % hw);  // == ((b*Ho)+oh)*Wo + ow
        float v = (float)acc[row * (long long)acc_ld + oc] / acc_div + bias[oc];
        v = fmaxf(v, 0.0f) * out_mul;
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        out[i] = (int8_t)lrintf(v);
    }
}

extern "C" void launch_conv_epilogue(const int32_t *acc, const float *bias,
                                     int8_t *out, int acc_ld, float acc_div,
                                     float out_mul, int B, int C, int Ho, int Wo,
                                     void *stream) {
    long long total = (long long)B * C * Ho * Wo;
    if (total <= 0) return;
    conv_epilogue_kernel<<<grid_for(total, 256), 256, 0, (cudaStream_t)stream>>>(
        acc, bias, out, acc_ld, acc_div, out_mul, C, Ho, Wo, total);
    CHECK_LAUNCH();
}

// ============================================================
// im2col for the FORWARD pass: int8 activation -> int8 col matrix, written
// directly with a padded leading dimension so the tensor-core GEMM can consume
// it with no second pass.  Staying in int8 keeps the stem-layer col matrix at
// 33 MB instead of 130 MB and matches the quantisation the GEMM expects.
// (The *backward* path never forms a col matrix at all — see the fused
// grad_W / grad_input kernels below.)
// ============================================================
__global__ void im2col_pad_kernel(const int8_t *__restrict__ input,
                                  int8_t *__restrict__ col, int C, int H, int W,
                                  int kH, int kW, int sH, int sW, int pH, int pW,
                                  int Ho, int Wo, int col_cols, int col_ld,
                                  long long total4) {
    const int hw_out = Ho * Wo;
    const int kk = kH * kW;
    // One thread produces 4 consecutive int8 columns = one aligned int32 store.
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total4; t += (long long)gridDim.x * blockDim.x) {
        const int groups = col_ld >> 2;
        const long long row = t / groups;
        const int c0 = (int)(t - row * groups) * 4;

        int32_t packed = 0;
        if (c0 < col_cols) {
            const int b = (int)(row / hw_out);
            const int rem = (int)(row - (long long)b * hw_out);
            const int oh = rem / Wo, ow = rem - oh * Wo;
            const int8_t *in_b = input + (long long)b * C * H * W;

            for (int j = 0; j < 4; ++j) {
                const int c = c0 + j;
                int8_t v = 0;
                if (c < col_cols) {
                    const int c_idx = c / kk;
                    const int ki = c - c_idx * kk;
                    const int khi = ki / kW, kwi = ki - khi * kW;
                    const int ih = oh * sH + khi - pH;
                    const int iw = ow * sW + kwi - pW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        v = in_b[(long long)c_idx * H * W + ih * W + iw];
                }
                packed |= ((int32_t)(uint8_t)v) << (8 * j);
            }
        }
        reinterpret_cast<int32_t *>(col)[t] = packed;
    }
}

extern "C" void launch_im2col_pad(const int8_t *input, int8_t *col, int batch,
                                  int C, int H, int W, int kH, int kW, int sH,
                                  int sW, int pH, int pW, int Ho, int Wo,
                                  int col_ld, void *stream) {
    const int col_cols = C * kH * kW;
    const long long rows = (long long)batch * Ho * Wo;
    if (col_ld < col_cols || (col_ld & 15) || rows <= 0) return;
    const long long total4 = rows * (col_ld >> 2);

    // The tensor-core kernel walks whole 16x16 tiles over the *padded* extents,
    // so rows in [rows, align16(rows)) are read even though their result is
    // clipped on store.  Uninitialised arena memory there can hold a bit
    // pattern that is Inf/NaN in int32, and lrintf of a NaN accumulator is
    // undefined — cheap insurance is to zero the tail before the kernel runs.
    const long long rows_p = ((rows + 15) / 16) * 16;
    if (rows_p > rows) {
        CHECK_CUDA(cudaMemsetAsync(col + rows * col_ld, 0,
                                   (size_t)(rows_p - rows) * col_ld,
                                   (cudaStream_t)stream));
    }

    im2col_pad_kernel<<<grid_for(total4, 256), 256, 0, (cudaStream_t)stream>>>(
        input, col, C, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo, col_cols, col_ld,
        total4);
    CHECK_LAUNCH();
}

// Compact (unpadded) im2col — kept for the diagnose tool / unit tests.
__global__ void im2col_gpu_kernel(const int8_t *__restrict__ input,
                                  float *__restrict__ col, int C, int H, int W,
                                  int kH, int kW, int sH, int sW, int pH, int pW,
                                  int Ho, int Wo, long long col_total) {
    const int col_cols = C * kH * kW;
    const int hw_out = Ho * Wo;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < col_total; idx += (long long)gridDim.x * blockDim.x) {
        long long row = idx / col_cols;
        int ci = (int)(idx - row * col_cols);
        int b = (int)(row / hw_out);
        int rem = (int)(row - (long long)b * hw_out);
        int oh = rem / Wo, ow = rem - oh * Wo;
        int c_idx = ci / (kH * kW);
        int ki = ci - c_idx * (kH * kW);
        int khi = ki / kW, kwi = ki - khi * kW;
        int ih = oh * sH + khi - pH;
        int iw = ow * sW + kwi - pW;
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            val = (float)input[((long long)(b * C + c_idx) * H + ih) * W + iw];
        col[idx] = val;
    }
}

extern "C" void launch_im2col_gpu(const int8_t *input, float *col, int batch,
                                  int C, int H, int W, int kH, int kW, int sH,
                                  int sW, int pH, int pW, int Ho, int Wo,
                                  void *stream) {
    long long col_total = (long long)batch * Ho * Wo * C * kH * kW;
    if (col_total <= 0) return;
    im2col_gpu_kernel<<<grid_for(col_total, 256), 256, 0, (cudaStream_t)stream>>>(
        input, col, C, H, W, kH, kW, sH, sW, pH, pW, Ho, Wo, col_total);
    CHECK_LAUNCH();
}

// ============================================================
// FUSED conv backward (1/3): grad_W
//   grad_W[oc, ci] = sum_{b,oh,ow} go[b,oc,oh,ow] *
//                    input[b, c, oh*sH+khi-pH, ow*sW+kwi-pW]
// Output layout is [Cout, Cin*kH*kW] — identical to weight_f32, so the
// optimizer consumes it with no reshaping.  Peak extra memory: Cout*col_cols
// floats (1 MB for 512x512x3x3) instead of a 38 GB col matrix.
// One block per output channel; threads sweep the col columns.
// ============================================================
#define GW_SMEM 512
__global__ void grad_weight_fused_kernel(const int8_t *__restrict__ input,
                                         const float *__restrict__ go,
                                         float *__restrict__ grad_W, int B,
                                         int Cin, int H, int W, int Cout, int kH,
                                         int kW, int sH, int sW, int pH, int pW,
                                         int Ho, int Wo) {
    const int oc = blockIdx.x;
    if (oc >= Cout) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int hw_out = Ho * Wo;
    const int col_cols = Cin * kH * kW;

    __shared__ float sgo[GW_SMEM];

    for (int ci = tid; ci < col_cols; ci += nthreads) {
        const int c_idx = ci / (kH * kW);
        const int kk = ci - c_idx * (kH * kW);
        const int khi = kk / kW, kwi = kk - khi * kW;

        float acc = 0.0f;
        for (int b = 0; b < B; ++b) {
            const int8_t *in_b = input + (long long)(b * Cin + c_idx) * H * W;
            const float *go_b = go + (long long)(b * Cout + oc) * hw_out;
            for (int p0 = 0; p0 < hw_out; p0 += GW_SMEM) {
                const int cnt = min(GW_SMEM, hw_out - p0);
                for (int j = tid; j < cnt; j += nthreads) sgo[j] = go_b[p0 + j];
                __syncthreads();
                for (int j = 0; j < cnt; ++j) {
                    const int p = p0 + j;
                    const int oh = p / Wo, ow = p - oh * Wo;
                    const int ih = oh * sH + khi - pH;
                    const int iw = ow * sW + kwi - pW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        acc += sgo[j] * (float)in_b[ih * W + iw];
                }
                __syncthreads();
            }
        }
        grad_W[(long long)oc * col_cols + ci] = acc;
    }
}
#undef GW_SMEM

extern "C" void launch_conv_backward_grad_weight(const int8_t *input,
                                                 const float *grad_out,
                                                 float *grad_W, int B, int Cin,
                                                 int H, int W, int Cout, int kH,
                                                 int kW, int sH, int sW, int pH,
                                                 int pW, int Ho, int Wo,
                                                 void *stream) {
    if (B <= 0 || Cout <= 0 || Cin <= 0) return;
    grad_weight_fused_kernel<<<Cout, 256, 0, (cudaStream_t)stream>>>(
        input, grad_out, grad_W, B, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, Ho, Wo);
    CHECK_LAUNCH();
}

// ============================================================
// FUSED conv backward (2/3): grad_bias[oc] = sum_{b,oh,ow} go[b,oc,oh,ow]
// grid = (Cout, B); results accumulate with atomicAdd.
// ============================================================
__global__ void grad_bias_kernel(const float *__restrict__ go,
                                 float *__restrict__ grad_bias, long long hw_out,
                                 int Cout) {
    const int oc = blockIdx.x;
    if (oc >= Cout) return;
    const float *g = go + ((long long)blockIdx.y * Cout + oc) * hw_out;

    float acc = 0.0f;
    for (long long i = threadIdx.x; i < hw_out; i += blockDim.x) acc += g[i];

    __shared__ float red[256];
    const int t = threadIdx.x;
    red[t] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    if (t == 0) atomicAdd(&grad_bias[oc], red[0]);
}

extern "C" void launch_conv_backward_grad_bias(const float *grad_out,
                                               float *grad_bias, int B, int Cout,
                                               int Ho, int Wo, void *stream) {
    if (B <= 0 || Cout <= 0) return;
    long long hw = (long long)Ho * Wo;
    dim3 grid((unsigned)Cout, (unsigned)B);
    int threads = (int)std::min<long long>(hw, 256);
    if (threads < 1) threads = 1;
    // blockDim must be a power of two for the shared-memory reduction above.
    int pow2 = 1;
    while (pow2 * 2 <= threads) pow2 *= 2;
    cudaMemsetAsync(grad_bias, 0, (size_t)Cout * sizeof(float), (cudaStream_t)stream);
    grad_bias_kernel<<<grid, pow2, 0, (cudaStream_t)stream>>>(grad_out, grad_bias,
                                                              hw, Cout);
    CHECK_LAUNCH();
}

// ============================================================
// FUSED conv backward (3/3): grad_input  ==  col2im(go @ W), never formed.
//
//   grad_input[b,ci,ih,iw] = sum_{khi,kwi} sum_{oc}
//        W[oc, (ci*kH+khi)*kW+kwi] * go[b, oc, oh, ow]
//   oh = (ih + pH - khi)/sH,  ow = (iw + pW - kwi)/sW   (skipped if not exact)
//
// Gather form: one thread == one element of grad_input, indexed by the INPUT
// position.  That means no atomics, no pre-zeroing and exactly one writer per
// output element (indexing by the output position instead would leave most of
// grad_input untouched and make the rest race).
//
// threadIdx.x is the input channel, so a whole warp shares the same
// (b, ih, iw) and therefore the same grad_output addresses (pure broadcast),
// while each lane reads a different, contiguous column of W.
// ============================================================
__global__ void grad_input_fused_kernel(const float *__restrict__ go,
                                        const float *__restrict__ W,
                                        float *__restrict__ grad_input, int B,
                                        int Cin, int H, int W, int Cout, int kH,
                                        int kW, int sH, int sW, int pH, int pW,
                                        int Ho, int Wo) {
    const int hw_out = Ho * Wo;
    const int col_cols = Cin * kH * kW;
    const int hw_in = H * W;
    const long long total = (long long)B * Cin * hw_in;

    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += (long long)gridDim.x * blockDim.x) {
        const int ci = (int)(t % Cin);
        const long long r = t / Cin;
        const int hw = (int)(r % hw_in);
        const int b = (int)(r / hw_in);
        const int ih = hw / W, iw = hw - ih * W;

        const float *go_batch = go + (long long)b * Cout * hw_out;
        float acc = 0.0f;

        for (int khi = 0; khi < kH; ++khi) {
            const int numh = ih + pH - khi;
            if (numh < 0 || numh % sH != 0) continue;
            const int oh = numh / sH;
            if (oh >= Ho) continue;
            for (int kwi = 0; kwi < kW; ++kwi) {
                const int numw = iw + pW - kwi;
                if (numw < 0 || numw % sW != 0) continue;
                const int ow = numw / sW;
                if (ow >= Wo) continue;

                const float *go_p = go_batch + oh * Wo + ow;   // stride hw_out
                const float *wp = W + (long long)(ci * kH + khi) * kW + kwi;
#pragma unroll 4
                for (int oc = 0; oc < Cout; ++oc)
                    acc += go_p[(long long)oc * hw_out] *
                           wp[(long long)oc * col_cols];
            }
        }

        grad_input[t] = acc;
    }
}

extern "C" void launch_conv_backward_grad_input(const float *grad_out,
                                                const float *weight,
                                                float *grad_input, int B, int Cin,
                                                int H, int W, int Cout, int kH,
                                                int kW, int sH, int sW, int pH,
                                                int pW, int Ho, int Wo,
                                                void *stream) {
    if (B <= 0 || Cin <= 0 || H <= 0 || W <= 0) return;
    long long work = (long long)B * Cin * H * W;
    grad_input_fused_kernel<<<grid_for(work, 256), 256, 0, (cudaStream_t)stream>>>(
        grad_out, weight, grad_input, B, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW,
        Ho, Wo);
    CHECK_LAUNCH();
}

// ============================================================
// Maxpool 2x2 backward — DEVICE pointers only.
// The old code passed a host Vec<f32> straight into a kernel, which is an
// instant illegal access.  The training loop uses the _host variant below
// because max_indices is produced by the CPU maxpool.
// ============================================================
__global__ void maxpool2x2_backward_kernel(const float *__restrict__ grad_output,
                                           const long long *__restrict__ max_indices,
                                           float *__restrict__ grad_input,
                                           long long output_size) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < output_size; i += (long long)gridDim.x * blockDim.x)
        atomicAdd(&grad_input[max_indices[i]], grad_output[i]);
}

extern "C" void launch_maxpool2x2_backward(const float *grad_output,
                                           const long long *max_indices,
                                           float *grad_input,
                                           long long output_size,
                                           long long input_size, void *stream) {
    cudaMemsetAsync(grad_input, 0, (size_t)input_size * sizeof(float),
                    (cudaStream_t)stream);
    maxpool2x2_backward_kernel<<<grid_for(output_size, 256), 256, 0,
                                 (cudaStream_t)stream>>>(
        grad_output, max_indices, grad_input, output_size);
    CHECK_LAUNCH();
}

extern "C" void maxpool2x2_backward_host(const float *grad_output,
                                         const unsigned long long *max_indices,
                                         float *grad_input,
                                         unsigned long long output_size,
                                         unsigned long long input_size) {
    std::memset(grad_input, 0, (size_t)input_size * sizeof(float));
    for (unsigned long long i = 0; i < output_size; ++i)
        grad_input[(size_t)max_indices[i]] += grad_output[i];
}

// ============================================================
// Small utilities
// ============================================================
__global__ void float_to_half_kernel(const float *src, __half *dst, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        dst[i] = __float2half(src[i]);
}

extern "C" void cuda_float_to_half(const float *src, void *dst, long long n,
                                   void *stream) {
    if (n <= 0) return;
    float_to_half_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(
        src, (__half *)dst, n);
    CHECK_LAUNCH();
}

__global__ void scale_vector_kernel(float *data, float scale, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        data[i] *= scale;
}

extern "C" void cuda_scale_vector(float *data, float scale, long long n,
                                  void *stream) {
    if (n <= 0) return;
    scale_vector_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(
        data, scale, n);
    CHECK_LAUNCH();
}

// dst[j,i] = src[i,j];  src is [rows,cols], dst is [cols,rows]
__global__ void transpose_f32_kernel(const float *__restrict__ src,
                                     float *__restrict__ dst, int rows, int cols) {
    __shared__ float tile[32][33];
    const int tiles_x = (cols + 31) / 32;
    const int total = tiles_x * ((rows + 31) / 32);
    for (int t = blockIdx.x; t < total; t += gridDim.x) {
        const int ty = t / tiles_x, tx = t - ty * tiles_x;
        const int x = tx * 32 + threadIdx.x;
        const int y = ty * 32 + threadIdx.y;
        for (int j = 0; j < 32; j += 8)
            if (x < cols && (y + j) < rows)
                tile[threadIdx.y + j][threadIdx.x] = src[(long long)(y + j) * cols + x];
        __syncthreads();
        const int x2 = ty * 32 + threadIdx.x;
        const int y2 = tx * 32 + threadIdx.y;
        for (int j = 0; j < 32; j += 8)
            if (x2 < rows && (y2 + j) < cols)
                dst[(long long)(y2 + j) * rows + x2] = tile[threadIdx.x][threadIdx.y + j];
        __syncthreads();
    }
}

extern "C" void cuda_transpose_f32(const float *src, float *dst, int rows,
                                   int cols, void *stream) {
    dim3 thr(32, 8);
    long long tiles = (long long)((rows + 31) / 32) * ((cols + 31) / 32);
    transpose_f32_kernel<<<grid_for(tiles, 1), thr, 0, (cudaStream_t)stream>>>(
        src, dst, rows, cols);
    CHECK_LAUNCH();
}

// ============================================================
// Runtime plumbing
// ============================================================
extern "C" void cuda_malloc(void **ptr, size_t size) {
    *ptr = nullptr;
    if (size == 0) return;
    CHECK_CUDA(cudaMalloc(ptr, size));
}

extern "C" void cuda_free(void *ptr) {
    if (ptr) CHECK_CUDA(cudaFree(ptr));
}

extern "C" void cuda_memcpy_dtod(void *dst, const void *src, size_t size, void *stream) {
    CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

extern "C" void cuda_memcpy_h2d(void *dst, const void *src, size_t size, void *stream) {
    if (size == 0) return;
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

extern "C" void cuda_memcpy_d2h(void *dst, const void *src, size_t size, void *stream) {
    if (size == 0) return;
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

extern "C" void cuda_sync(void *stream) { CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)stream)); }

extern "C" void cuda_stream_create(void **stream) {
    CHECK_CUDA(cudaStreamCreate((cudaStream_t *)stream));
}

extern "C" void cuda_stream_destroy(void *stream) {
    if (stream) CHECK_CUDA(cudaStreamDestroy((cudaStream_t)stream));
}

extern "C" void cuda_get_memory_info(size_t *free_mem, size_t *total_mem) {
    CHECK_CUDA(cudaMemGetInfo(free_mem, total_mem));
}

// ---- Persistent workspace -------------------------------------------------
// Allocating and freeing hundreds of MB per layer per batch is slow and a
// fragmentation magnet.  One grow-only arena serves the whole run; callers
// carve it into non-overlapping regions themselves.
static void *g_workspace = nullptr;
static size_t g_workspace_size = 0;

extern "C" void *trba_workspace_acquire(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > g_workspace_size) {
        if (g_workspace) { CHECK_CUDA(cudaFree(g_workspace)); g_workspace = nullptr; }
        size_t want = bytes + bytes / 8 + (16u << 20u);
        if (cudaMalloc(&g_workspace, want) != cudaSuccess) {
            g_workspace = nullptr;
            CHECK_CUDA(cudaMalloc(&g_workspace, bytes));  // exact, or die loudly
            want = bytes;
        }
        g_workspace_size = want;
    }
    return g_workspace;
}

extern "C" size_t trba_workspace_size(void) { return g_workspace_size; }

extern "C" void trba_workspace_release(void) {
    if (g_workspace) { CHECK_CUDA(cudaFree(g_workspace)); g_workspace = nullptr; g_workspace_size = 0; }
}

extern "C" const char *trba_backend_name(void) { return "cuda"; }
extern "C" int trba_device_cc_major(void) { return dev_prop().major; }
extern "C" int trba_device_cc_minor(void) { return dev_prop().minor; }
extern "C" const char *trba_device_name(void) { return dev_prop().name; }


// ============================================================
// BatchNorm + dynamic-quantisation helpers (QAT pipeline)
// ============================================================
__global__ void bn_stats_kernel(const float *__restrict__ x, int C, int HW,
                                long long n, float *__restrict__ sum,
                                float *__restrict__ sumsq) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)((i / HW) % C);
        float v = x[i];
        atomicAdd(&sum[c], v);
        atomicAdd(&sumsq[c], v * v);
    }
}
__global__ void bn_finalise_kernel(float *sum, float *sumsq, long long count, int C) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float m = sum[c] / (float)count;
        float v = sumsq[c] / (float)count - m * m;
        sum[c] = m;
        sumsq[c] = v < 0.0f ? 0.0f : v;
    }
}
extern "C" void launch_bn_stats(const float *x, int B, int C, int HW, float *mean,
                                float *var, void *stream) {
    long long n = (long long)B * C * HW;
    if (n <= 0) return;
    CHECK_CUDA(cudaMemsetAsync(mean, 0, sizeof(float) * C, (cudaStream_t)stream));
    CHECK_CUDA(cudaMemsetAsync(var, 0, sizeof(float) * C, (cudaStream_t)stream));
    bn_stats_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, C, HW, n, mean, var);
    bn_finalise_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(mean, var, (long long)B * HW, C);
    CHECK_LAUNCH();
}

__global__ void bn_apply_kernel(const float *__restrict__ x, const float *__restrict__ mean,
                                const float *__restrict__ var, const float *__restrict__ gamma,
                                const float *__restrict__ beta, float eps, float *__restrict__ y,
                                int C, int HW, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)((i / HW) % C);
        float r = rsqrtf(var[c] + eps);
        y[i] = (x[i] - mean[c]) * r * gamma[c] + beta[c];
    }
}
extern "C" void launch_bn_apply(const float *x, const float *mean, const float *var,
                                const float *gamma, const float *beta, float eps, float *y,
                                int C, int HW, long long n, void *stream) {
    if (n <= 0) return;
    bn_apply_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, mean, var, gamma, beta, eps, y, C, HW, n);
    CHECK_LAUNCH();
}

__global__ void bn_bwd_sums_kernel(const float *__restrict__ dy, const float *__restrict__ x,
                                   const float *__restrict__ mean, const float *__restrict__ var,
                                   float eps, int C, int HW, long long n,
                                   float *__restrict__ sum_dy, float *__restrict__ sum_dy_xhat) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)((i / HW) % C);
        float xhat = (x[i] - mean[c]) * rsqrtf(var[c] + eps);
        atomicAdd(&sum_dy[c], dy[i]);
        atomicAdd(&sum_dy_xhat[c], dy[i] * xhat);
    }
}
extern "C" void launch_bn_bwd_sums(const float *dy, const float *x, const float *mean,
                                   const float *var, float eps, int C, int HW, long long n,
                                   float *sum_dy, float *sum_dy_xhat, void *stream) {
    if (n <= 0) return;
    CHECK_CUDA(cudaMemsetAsync(sum_dy, 0, sizeof(float) * C, (cudaStream_t)stream));
    CHECK_CUDA(cudaMemsetAsync(sum_dy_xhat, 0, sizeof(float) * C, (cudaStream_t)stream));
    bn_bwd_sums_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(dy, x, mean, var, eps, C, HW, n, sum_dy, sum_dy_xhat);
    CHECK_LAUNCH();
}

__global__ void bn_bwd_dx_kernel(const float *__restrict__ dy, const float *__restrict__ x,
                                 const float *__restrict__ mean, const float *__restrict__ var,
                                 const float *__restrict__ gamma, const float *__restrict__ sum_dy,
                                 const float *__restrict__ sum_dy_xhat, float eps, int C, int HW,
                                 long long n, long long count, float *__restrict__ dx) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)((i / HW) % C);
        float r = rsqrtf(var[c] + eps);
        float xhat = (x[i] - mean[c]) * r;
        dx[i] = gamma[c] * r / (float)count *
                ((float)count * dy[i] - sum_dy[c] - xhat * sum_dy_xhat[c]);
    }
}
extern "C" void launch_bn_bwd_dx(const float *dy, const float *x, const float *mean,
                                 const float *var, const float *gamma, const float *sum_dy,
                                 const float *sum_dy_xhat, float eps, int C, int HW, long long n,
                                 long long count, float *dx, void *stream) {
    if (n <= 0) return;
    bn_bwd_dx_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(dy, x, mean, var, gamma, sum_dy, sum_dy_xhat, eps, C, HW, n, count, dx);
    CHECK_LAUNCH();
}

__global__ void reduce_max_abs_kernel(const float *__restrict__ x, long long n, float *__restrict__ out) {
    __shared__ float sm[256];
    float m = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float v = fabsf(x[i]);
        if (v > m) m = v;
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((int *)out, __float_as_int(sm[0]));
}
extern "C" void launch_reduce_max_abs(const float *x, long long n, float *out, void *stream) {
    CHECK_CUDA(cudaMemsetAsync(out, 0, sizeof(float), (cudaStream_t)stream));
    if (n <= 0) return;
    reduce_max_abs_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, n, out);
    CHECK_LAUNCH();
}

__global__ void quant_f32_i8_kernel(const float *__restrict__ x, long long n, float inv_scale,
                                    int8_t *__restrict__ q) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float v = x[i] * inv_scale;
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        q[i] = (int8_t)lrintf(v);
    }
}
extern "C" void launch_quant_f32_i8(const float *x, long long n, float scale, int8_t *q, void *stream) {
    if (n <= 0) return;
    quant_f32_i8_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, n, 1.0f / scale, q);
    CHECK_LAUNCH();
}

__global__ void relu_quant_kernel(const float *__restrict__ y, long long n, float inv_scale,
                                  int8_t *__restrict__ q) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float v = fmaxf(y[i], 0.0f) * inv_scale;
        v = fminf(v, 127.0f);
        q[i] = (int8_t)lrintf(v);
    }
}
extern "C" void launch_relu_quant(const float *y, long long n, float scale, int8_t *q, void *stream) {
    if (n <= 0) return;
    relu_quant_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(y, n, 1.0f / scale, q);
    CHECK_LAUNCH();
}

__global__ void dequant_i8_f32_kernel(const int8_t *__restrict__ q, long long n, float scale,
                                      float *__restrict__ x) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) x[i] = (float)q[i] * scale;
}
extern "C" void launch_dequant_i8_f32(const int8_t *q, long long n, float scale, float *x, void *stream) {
    if (n <= 0) return;
    dequant_i8_f32_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(q, n, scale, x);
    CHECK_LAUNCH();
}

__global__ void add_scaled_i8_kernel(const int8_t *__restrict__ a, float sa,
                                     const int8_t *__restrict__ b, float sb, long long n,
                                     float *__restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        out[i] = (float)a[i] * sa + (float)b[i] * sb;
}
extern "C" void launch_add_scaled_i8(const int8_t *a, float sa, const int8_t *b, float sb,
                                     long long n, float *out, void *stream) {
    if (n <= 0) return;
    add_scaled_i8_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(a, sa, b, sb, n, out);
    CHECK_LAUNCH();
}

// acc [rows, C] (ld = acc_ld) -> NCHW f32, x = acc * scale
__global__ void epilogue_f32_kernel(const int32_t *__restrict__ acc, int acc_ld, float scale,
                                    float *__restrict__ out, int C, int Ho, int Wo, long long total) {
    const long long hw = (long long)Ho * Wo;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        long long b = i / ((long long)C * hw);
        int oc = (int)((i / hw) % C);
        long long row = b * hw + (i % hw);
        out[i] = (float)acc[row * (long long)acc_ld + oc] * scale;
    }
}
extern "C" void launch_epilogue_f32(const int32_t *acc, int acc_ld, float scale, float *out,
                                    int C, int Ho, int Wo, long long total, void *stream) {
    if (total <= 0) return;
    epilogue_f32_kernel<<<grid_for(total, 256), 256, 0, (cudaStream_t)stream>>>(acc, acc_ld, scale, out, C, Ho, Wo, total);
    CHECK_LAUNCH();
}

__global__ void avgpool_hw_kernel(const float *__restrict__ x, int C, int HW, long long bc,
                                  float *__restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bc;
         i += (long long)gridDim.x * blockDim.x) {
        const float *p = x + i * HW;
        float s = 0.0f;
        for (int j = 0; j < HW; ++j) s += p[j];
        out[i] = s / (float)HW;
    }
}
extern "C" void launch_avgpool_hw(const float *x, int C, int HW, long long B, float *out, void *stream) {
    long long bc = B * C;
    if (bc <= 0) return;
    avgpool_hw_kernel<<<grid_for(bc, 256), 256, 0, (cudaStream_t)stream>>>(x, C, HW, bc, out);
    CHECK_LAUNCH();
}

__global__ void avgpool_hw_bwd_kernel(const float *__restrict__ dy, int C, int HW, long long bc,
                                      float *__restrict__ dx) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bc;
         i += (long long)gridDim.x * blockDim.x) {
        float v = dy[i] / (float)HW;
        float *p = dx + i * HW;
        for (int j = 0; j < HW; ++j) p[j] = v;
    }
}
extern "C" void launch_avgpool_hw_backward(const float *dy, int C, int HW, long long B, float *dx, void *stream) {
    long long bc = B * C;
    if (bc <= 0) return;
    avgpool_hw_bwd_kernel<<<grid_for(bc, 256), 256, 0, (cudaStream_t)stream>>>(dy, C, HW, bc, dx);
    CHECK_LAUNCH();
}

__global__ void mul_gate_kernel(const float *__restrict__ x, const float *__restrict__ gate,
                                int C, int HW, long long n, float *__restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)((i / HW) % C);
        long long b = i / ((long long)C * HW);
        out[i] = x[i] * gate[b * C + c];
    }
}
extern "C" void launch_mul_gate(const float *x, const float *gate, int C, int HW, long long n,
                                float *out, void *stream) {
    if (n <= 0) return;
    mul_gate_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(x, gate, C, HW, n, out);
    CHECK_LAUNCH();
}

__global__ void mask_relu_bwd_kernel(const float *__restrict__ y, const float *__restrict__ dy,
                                     long long n, float *__restrict__ dx) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        dx[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}
extern "C" void launch_mask_relu_bwd(const float *y, const float *dy, long long n, float *dx, void *stream) {
    if (n <= 0) return;
    mask_relu_bwd_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(y, dy, n, dx);
    CHECK_LAUNCH();
}

__global__ void zero_f32_kernel(float *__restrict__ p, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) p[i] = 0.0f;
}
extern "C" void launch_zero_f32(float *p, long long n, void *stream) {
    if (n <= 0) return;
    zero_f32_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(p, n);
    CHECK_LAUNCH();
}


__global__ void transpose_i8_kernel(const int8_t *__restrict__ src, int rows, int cols,
                                    int8_t *__restrict__ dst) {
    const long long n = (long long)rows * cols;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int r = (int)(i / cols), c = (int)(i - (long long)r * cols);
        dst[(long long)c * rows + r] = src[i];
    }
}
extern "C" void launch_transpose_i8(const int8_t *src, int rows, int cols, int8_t *dst, void *stream) {
    long long n = (long long)rows * cols;
    if (n <= 0) return;
    transpose_i8_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(src, rows, cols, dst);
    CHECK_LAUNCH();
}


__global__ void add_f32_kernel(const float *__restrict__ a, const float *__restrict__ b, long long n, float *__restrict__ o) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) o[i] = a[i] + b[i];
}
extern "C" void launch_add_f32(const float *a, const float *b, long long n, float *o, void *stream) {
    if (n <= 0) return;
    add_f32_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(a, b, n, o);
    CHECK_LAUNCH();
}

__global__ void mul_f32_kernel(const float *__restrict__ a, const float *__restrict__ b, long long n, float *__restrict__ o) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) o[i] = a[i] * b[i];
}
extern "C" void launch_mul_f32(const float *a, const float *b, long long n, float *o, void *stream) {
    if (n <= 0) return;
    mul_f32_kernel<<<grid_for(n, 256), 256, 0, (cudaStream_t)stream>>>(a, b, n, o);
    CHECK_LAUNCH();
}

// x [B, C, H, W] -> out [B, W, C] = mean over H (the TRBA encoder pooling)
__global__ void meanpool_h_kernel(const float *__restrict__ x, int C, int H, int W,
                                  long long total, float *__restrict__ out) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C);
        int w = (int)((i / C) % W);
        long long b = i / ((long long)C * W);
        float s = 0.0f;
        const float *p = x + ((b * C + c) * H) * W + w;
        for (int h = 0; h < H; ++h) s += p[(long long)h * W];
        out[i] = s / (float)H;
    }
}
extern "C" void launch_meanpool_h(const float *x, int C, int H, int W, long long B, float *out, void *stream) {
    long long total = B * W * C;
    if (total <= 0) return;
    meanpool_h_kernel<<<grid_for(total, 256), 256, 0, (cudaStream_t)stream>>>(x, C, H, W, total, out);
    CHECK_LAUNCH();
}

__global__ void meanpool_h_bwd_kernel(const float *__restrict__ dy, int C, int H, int W,
                                      long long total, float *__restrict__ dx) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int c = (int)(i % C);
        int w = (int)((i / C) % W);
        long long b = i / ((long long)C * W);
        float v = dy[i] / (float)H;
        float *p = dx + ((b * C + c) * H) * W + w;
        for (int h = 0; h < H; ++h) p[(long long)h * W] = v;
    }
}
extern "C" void launch_meanpool_h_backward(const float *dy, int C, int H, int W, long long B, float *dx, void *stream) {
    long long total = B * W * C;
    if (total <= 0) return;
    meanpool_h_bwd_kernel<<<grid_for(total, 256), 256, 0, (cudaStream_t)stream>>>(dy, C, H, W, total, dx);
    CHECK_LAUNCH();
}

// ---- Static model-weight blobs (kept for the memory report) ---------------
static void *g_cnn_weights = nullptr;
static void *g_bilstm_weights = nullptr;
static void *g_attention_weights = nullptr;
static void *g_ctc_weights = nullptr;

extern "C" void allocate_model_weights(size_t cnn_size, size_t bilstm_size,
                                       size_t attn_size, size_t ctc_size) {
    if (cnn_size > 0) CHECK_CUDA(cudaMalloc(&g_cnn_weights, cnn_size));
    if (bilstm_size > 0) CHECK_CUDA(cudaMalloc(&g_bilstm_weights, bilstm_size));
    if (attn_size > 0) CHECK_CUDA(cudaMalloc(&g_attention_weights, attn_size));
    if (ctc_size > 0) CHECK_CUDA(cudaMalloc(&g_ctc_weights, ctc_size));
}

extern "C" void upload_cnn_weights(const void *d, size_t s) {
    if (g_cnn_weights && d && s) CHECK_CUDA(cudaMemcpy(g_cnn_weights, d, s, cudaMemcpyHostToDevice));
}
extern "C" void upload_bilstm_weights(const void *d, size_t s) {
    if (g_bilstm_weights && d && s) CHECK_CUDA(cudaMemcpy(g_bilstm_weights, d, s, cudaMemcpyHostToDevice));
}
extern "C" void upload_attention_weights(const void *d, size_t s) {
    if (g_attention_weights && d && s) CHECK_CUDA(cudaMemcpy(g_attention_weights, d, s, cudaMemcpyHostToDevice));
}
extern "C" void upload_ctc_weights(const void *d, size_t s) {
    if (g_ctc_weights && d && s) CHECK_CUDA(cudaMemcpy(g_ctc_weights, d, s, cudaMemcpyHostToDevice));
}
extern "C" void *get_cnn_weights_ptr(void) { return g_cnn_weights; }
extern "C" void *get_bilstm_weights_ptr(void) { return g_bilstm_weights; }
extern "C" void *get_attention_weights_ptr(void) { return g_attention_weights; }
extern "C" void *get_ctc_weights_ptr(void) { return g_ctc_weights; }

extern "C" void free_model_weights(void) {
    if (g_cnn_weights) { cudaFree(g_cnn_weights); g_cnn_weights = nullptr; }
    if (g_bilstm_weights) { cudaFree(g_bilstm_weights); g_bilstm_weights = nullptr; }
    if (g_attention_weights) { cudaFree(g_attention_weights); g_attention_weights = nullptr; }
    if (g_ctc_weights) { cudaFree(g_ctc_weights); g_ctc_weights = nullptr; }
}
