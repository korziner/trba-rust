// ============================================================
// trba-rust — CPU fallback backend
//
// Implements exactly the same `extern "C"` surface as cuda/kernels.cu, so the
// binary builds and runs on a machine with no CUDA toolkit and no GPU.  Used
// by `cargo test` and by `cargo run --bin diagnose` to check that the fused
// backward kernels compute the same gradients as a straightforward reference
// implementation.
//
// "Device" pointers are just malloc'd host pointers, streams are no-ops and
// every call is synchronous.
//
// Build with -fopenmp when available; everything also works single-threaded.
// ============================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline void *xmalloc(size_t n) {
    void *p = std::malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "\n[CPU FATAL] out of memory allocating %zu bytes\n", n);
        std::abort();
    }
    return p;
}

// ---- IEEE-754 half <-> float (no F16C dependency) ----
inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = bits & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;                 // flush denormals
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);    // inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

inline int align16(int x) { return (x + 15) & ~15; }

void *g_workspace = nullptr;
size_t g_workspace_size = 0;

void *g_cnn_weights = nullptr;
void *g_bilstm_weights = nullptr;
void *g_attention_weights = nullptr;
void *g_ctc_weights = nullptr;

}  // namespace

// ============================================================
// GEMMs
// ============================================================

// C[M,N](int32) = A[M,K](int8) * Bt[N,K](int8)^T
// Operands are zero-padded to a multiple of 16 in every dimension; C is not.
extern "C" void launch_gemm_int8(const int8_t *A, const int8_t *B, int32_t *C,
                                 int M, int N, int K, void *) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int Kp = align16(K);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        const int8_t *a = A + (long long)m * Kp;
        int32_t *c = C + (long long)m * N;
        for (int n = 0; n < N; ++n) {
            const int8_t *b = B + (long long)n * Kp;
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) acc += (int32_t)a[k] * (int32_t)b[k];
            c[n] = acc;
        }
    }
}

// C[M,N](f32) = A[M,K](f16) * Bt[N,K](f16)^T   (same padding contract)
extern "C" void launch_gemm_fp16(const void *A, const void *B, float *C, int M,
                                 int N, int K, void *) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int Kp = align16(K);
    const uint16_t *a = static_cast<const uint16_t *>(A);
    const uint16_t *b = static_cast<const uint16_t *>(B);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += half_to_float(a[(long long)m * Kp + k]) *
                       half_to_float(b[(long long)n * Kp + k]);
            C[(long long)m * N + n] = acc;
        }
    }
}

// C[M,N] = A[M,K] * B[K,N]
extern "C" void launch_gemm_fp32(const float *A, const float *B, float *C,
                                 int M, int N, int K, void *) {
    if (M <= 0 || N <= 0 || K <= 0) return;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        const float *a = A + (long long)m * K;
        float *c = C + (long long)m * N;
        for (int n = 0; n < N; ++n) c[n] = 0.f;
        for (int k = 0; k < K; ++k) {
            const float av = a[k];
            if (av == 0.f) continue;
            const float *b = B + (long long)k * N;
            for (int n = 0; n < N; ++n) c[n] += av * b[n];
        }
    }
}

// ============================================================
// Conv epilogue
// ============================================================
extern "C" void launch_conv_epilogue(const int32_t *acc, const float *bias,
                                     int8_t *out, int acc_ld, float acc_div,
                                     float out_mul, int B, int C, int Ho, int Wo,
                                     void *) {
    long long total = (long long)B * C * Ho * Wo;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    const long long hw = (long long)Ho * Wo;
    for (long long i = 0; i < total; ++i) {
        // out is [B, C, Ho, Wo]; acc is [B*Ho*Wo, C] row-major with leading
        // dimension acc_ld (>= C).  The old `i % C` / `i / C` mapping treated
        // the output as NHWC and read the accumulator with the wrong strides,
        // so every activation after the first conv was garbage.  Note that for
        // B > 1 the batch index has to come out of the row, not of oc.
        long long b = i / ((long long)C * hw);
        int oc = (int)((i / hw) % C);
        long long row = b * hw + (i % hw);
        float v = (float)acc[row * (long long)acc_ld + oc] / acc_div + bias[oc];
        v = std::max(v, 0.0f) * out_mul;
        v = std::min(std::max(v, -127.0f), 127.0f);
        out[i] = (int8_t)std::lrint(v);
    }
}

// ============================================================
// im2col
// ============================================================
extern "C" void launch_im2col_pad(const int8_t *input, int8_t *col, int batch,
                                  int C, int H, int W, int kH, int kW, int sH,
                                  int sW, int pH, int pW, int Ho, int Wo,
                                  int col_ld, void *) {
    const int col_cols = C * kH * kW;
    const long long rows = (long long)batch * Ho * Wo;
    if (col_ld < col_cols || (col_ld & 15) || rows <= 0) return;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long row = 0; row < rows; ++row) {
        int b = (int)(row / ((long long)Ho * Wo));
        int rem = (int)(row - (long long)b * Ho * Wo);
        int oh = rem / Wo, ow = rem - oh * Wo;
        int8_t *dst = col + row * col_ld;
        for (int c = col_cols; c < col_ld; ++c) dst[c] = 0;
        int p = 0;
        for (int ci = 0; ci < C; ++ci) {
            const int8_t *in_b = input + (long long)(b * C + ci) * H * W;
            for (int khi = 0; khi < kH; ++khi) {
                int ih = oh * sH + khi - pH;
                for (int kwi = 0; kwi < kW; ++kwi) {
                    int iw = ow * sW + kwi - pW;
                    dst[p++] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                   ? in_b[ih * W + iw]
                                   : (int8_t)0;
                }
            }
        }
    }
    // Rows in [rows, align16(rows)) are padding the GEMM may read; zero them.
    const long long rows_p = ((rows + 15) / 16) * 16;
    if (rows_p > rows)
        std::memset(col + rows * col_ld, 0, (size_t)(rows_p - rows) * col_ld);
}

extern "C" void launch_im2col_gpu(const int8_t *input, float *col, int batch,
                                  int C, int H, int W, int kH, int kW, int sH,
                                  int sW, int pH, int pW, int Ho, int Wo,
                                  void *) {
    const int col_cols = C * kH * kW;
    const int hw_out = Ho * Wo;
    const long long rows = (long long)batch * hw_out;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long row = 0; row < rows; ++row) {
        int b = (int)(row / hw_out);
        int rem = (int)(row - (long long)b * hw_out);
        int oh = rem / Wo, ow = rem - oh * Wo;
        float *dst = col + row * col_cols;
        int p = 0;
        for (int c = 0; c < C; ++c) {
            const int8_t *in_b = input + (long long)(b * C + c) * H * W;
            for (int khi = 0; khi < kH; ++khi) {
                int ih = oh * sH + khi - pH;
                for (int kwi = 0; kwi < kW; ++kwi) {
                    int iw = ow * sW + kwi - pW;
                    dst[p++] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                   ? (float)in_b[ih * W + iw]
                                   : 0.0f;
                }
            }
        }
    }
}

// ============================================================
// Fused conv backward
// ============================================================
extern "C" void launch_conv_backward_grad_weight(const int8_t *input,
                                                 const float *go, float *grad_W,
                                                 int B, int Cin, int H, int W,
                                                 int Cout, int kH, int kW, int sH,
                                                 int sW, int pH, int pW, int Ho,
                                                 int Wo, void *) {
    if (B <= 0 || Cout <= 0 || Cin <= 0) return;
    const int col_cols = Cin * kH * kW;
    const int hw_out = Ho * Wo;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int oc = 0; oc < Cout; ++oc) {
        for (int ci = 0; ci < col_cols; ++ci) {
            const int c_idx = ci / (kH * kW);
            const int kk = ci - c_idx * (kH * kW);
            const int khi = kk / kW, kwi = kk - khi * kW;
            float acc = 0.0f;
            for (int b = 0; b < B; ++b) {
                const int8_t *in_b = input + (long long)(b * Cin + c_idx) * H * W;
                const float *go_b = go + (long long)(b * Cout + oc) * hw_out;
                for (int p = 0; p < hw_out; ++p) {
                    int oh = p / Wo, ow = p - oh * Wo;
                    int ih = oh * sH + khi - pH;
                    int iw = ow * sW + kwi - pW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        acc += go_b[p] * (float)in_b[ih * W + iw];
                }
            }
            grad_W[(long long)oc * col_cols + ci] = acc;
        }
    }
}

extern "C" void launch_conv_backward_grad_bias(const float *go, float *grad_bias,
                                               int B, int Cout, int Ho, int Wo,
                                               void *) {
    if (B <= 0 || Cout <= 0) return;
    const long long hw = (long long)Ho * Wo;
    std::memset(grad_bias, 0, (size_t)Cout * sizeof(float));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < Cout; ++oc) {
        float acc = 0.0f;
        for (int b = 0; b < B; ++b) {
            const float *g = go + ((long long)b * Cout + oc) * hw;
            for (long long i = 0; i < hw; ++i) acc += g[i];
        }
        grad_bias[oc] = acc;
    }
}

extern "C" void launch_conv_backward_grad_input(const float *go, const float *Wt,
                                                float *grad_input, int B, int Cin,
                                                int H, int W, int Cout, int kH,
                                                int kW, int sH, int sW, int pH,
                                                int pW, int Ho, int Wo, void *) {
    if (B <= 0 || Cin <= 0 || H <= 0 || W <= 0) return;
    const int hw_out = Ho * Wo;
    const int col_cols = Cin * kH * kW;
    const int hw_in = H * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < B; ++b) {
        const float *go_b = go + (long long)b * Cout * hw_out;
        for (int ci = 0; ci < Cin; ++ci) {
            float *gi = grad_input + (long long)(b * Cin + ci) * hw_in;
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    float acc = 0.0f;
                    for (int khi = 0; khi < kH; ++khi) {
                        int numh = ih + pH - khi;
                        if (numh < 0 || numh % sH != 0) continue;
                        int oh = numh / sH;
                        if (oh >= Ho) continue;
                        for (int kwi = 0; kwi < kW; ++kwi) {
                            int numw = iw + pW - kwi;
                            if (numw < 0 || numw % sW != 0) continue;
                            int ow = numw / sW;
                            if (ow >= Wo) continue;
                            const float *wp = Wt + (long long)(ci * kH + khi) * kW + kwi;
                            const float *gp = go_b + oh * Wo + ow;
                            for (int oc = 0; oc < Cout; ++oc)
                                acc += gp[(long long)oc * hw_out] *
                                       wp[(long long)oc * col_cols];
                        }
                    }
                    gi[ih * W + iw] = acc;
                }
            }
        }
    }
}

// ============================================================
// Maxpool 2x2 backward
// ============================================================
extern "C" void launch_maxpool2x2_backward(const float *grad_output,
                                           const long long *max_indices,
                                           float *grad_input,
                                           long long output_size,
                                           long long input_size, void *) {
    std::memset(grad_input, 0, (size_t)input_size * sizeof(float));
    for (long long i = 0; i < output_size; ++i)
        grad_input[max_indices[i]] += grad_output[i];
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
extern "C" void cuda_float_to_half(const float *src, void *dst, long long n, void *) {
    uint16_t *d = static_cast<uint16_t *>(dst);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < n; ++i) d[i] = float_to_half(src[i]);
}

extern "C" void cuda_scale_vector(float *data, float scale, long long n, void *) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < n; ++i) data[i] *= scale;
}

// dst[j,i] = src[i,j]
extern "C" void cuda_transpose_f32(const float *src, float *dst, int rows,
                                   int cols, void *) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[(long long)j * rows + i] = src[(long long)i * cols + j];
}

// ============================================================
// Runtime plumbing
// ============================================================
extern "C" void cuda_malloc(void **ptr, size_t size) {
    *ptr = size ? xmalloc(size) : nullptr;
    if (size) std::memset(*ptr, 0, size);
}

extern "C" void cuda_free(void *ptr) { std::free(ptr); }

extern "C" void cuda_memcpy_dtod(void *dst, const void *src, size_t size, void *) {
    std::memcpy(dst, src, size);
}

extern "C" void cuda_memcpy_h2d(void *dst, const void *src, size_t size, void *) {
    if (size) std::memcpy(dst, src, size);
}

extern "C" void cuda_memcpy_d2h(void *dst, const void *src, size_t size, void *) {
    if (size) std::memcpy(dst, src, size);
}

extern "C" void cuda_sync(void *) {}
extern "C" void cuda_stream_create(void **stream) { *stream = nullptr; }
extern "C" void cuda_stream_destroy(void *) {}

extern "C" void cuda_get_memory_info(size_t *free_mem, size_t *total_mem) {
    size_t total = 0, avail = 0;
    FILE *f = std::fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long v = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1) total = (size_t)v * 1024;
            else if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) avail = (size_t)v * 1024;
        }
        std::fclose(f);
    }
    if (total_mem) *total_mem = total;
    if (free_mem) *free_mem = avail;
}

extern "C" void *trba_workspace_acquire(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > g_workspace_size) {
        std::free(g_workspace);
        size_t want = bytes + bytes / 8 + (16u << 20u);
        g_workspace = xmalloc(want);
        g_workspace_size = want;
    }
    return g_workspace;
}

extern "C" size_t trba_workspace_size(void) { return g_workspace_size; }

extern "C" void trba_workspace_release(void) {
    std::free(g_workspace);
    g_workspace = nullptr;
    g_workspace_size = 0;
}

extern "C" const char *trba_backend_name(void) { return "cpu"; }
extern "C" int trba_device_cc_major(void) { return 0; }
extern "C" int trba_device_cc_minor(void) { return 0; }
extern "C" const char *trba_device_name(void) { return "CPU (no CUDA)"; }

extern "C" void allocate_model_weights(size_t cnn_size, size_t bilstm_size,
                                       size_t attn_size, size_t ctc_size) {
    if (cnn_size > 0) g_cnn_weights = xmalloc(cnn_size);
    if (bilstm_size > 0) g_bilstm_weights = xmalloc(bilstm_size);
    if (attn_size > 0) g_attention_weights = xmalloc(attn_size);
    if (ctc_size > 0) g_ctc_weights = xmalloc(ctc_size);
}


// ============================================================
// BatchNorm + dynamic-quantisation helpers (CPU mirror)
// ============================================================
extern "C" void launch_bn_stats(const float *x, int B, int C, int HW, float *mean,
                                float *var, void *) {
    long long n = (long long)B * C * HW;
    if (n <= 0) return;
    std::fill(mean, mean + C, 0.0f);
    std::fill(var, var + C, 0.0f);
    const long long count = (long long)B * HW;
#pragma omp parallel
    {
        std::vector<double> s(C, 0.0), sq(C, 0.0);
#pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            int c = (int)((i / HW) % C);
            double v = x[i];
            s[c] += v;
            sq[c] += v * v;
        }
#pragma omp critical
        for (int c = 0; c < C; ++c) { mean[c] += (float)s[c]; var[c] += (float)sq[c]; }
    }
    for (int c = 0; c < C; ++c) {
        float m = mean[c] / (float)count;
        float v = var[c] / (float)count - m * m;
        mean[c] = m;
        var[c] = v < 0.0f ? 0.0f : v;
    }
}

extern "C" void launch_bn_apply(const float *x, const float *mean, const float *var,
                                const float *gamma, const float *beta, float eps, float *y,
                                int C, int HW, long long n, void *) {
    std::vector<float> r(C);
    for (int c = 0; c < C; ++c) r[c] = 1.0f / std::sqrt(var[c] + eps);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        int c = (int)((i / HW) % C);
        y[i] = (x[i] - mean[c]) * r[c] * gamma[c] + beta[c];
    }
}

extern "C" void launch_bn_bwd_sums(const float *dy, const float *x, const float *mean,
                                   const float *var, float eps, int C, int HW, long long n,
                                   float *sum_dy, float *sum_dy_xhat, void *) {
    std::fill(sum_dy, sum_dy + C, 0.0f);
    std::fill(sum_dy_xhat, sum_dy_xhat + C, 0.0f);
    std::vector<float> r(C);
    for (int c = 0; c < C; ++c) r[c] = 1.0f / std::sqrt(var[c] + eps);
#pragma omp parallel
    {
        std::vector<double> s(C, 0.0), sx(C, 0.0);
#pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            int c = (int)((i / HW) % C);
            double xhat = (x[i] - mean[c]) * r[c];
            s[c] += dy[i];
            sx[c] += dy[i] * xhat;
        }
#pragma omp critical
        for (int c = 0; c < C; ++c) { sum_dy[c] += (float)s[c]; sum_dy_xhat[c] += (float)sx[c]; }
    }
}

extern "C" void launch_bn_bwd_dx(const float *dy, const float *x, const float *mean,
                                 const float *var, const float *gamma, const float *sum_dy,
                                 const float *sum_dy_xhat, float eps, int C, int HW, long long n,
                                 long long count, float *dx, void *) {
    std::vector<float> r(C);
    for (int c = 0; c < C; ++c) r[c] = 1.0f / std::sqrt(var[c] + eps);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        int c = (int)((i / HW) % C);
        float xhat = (x[i] - mean[c]) * r[c];
        dx[i] = gamma[c] * r[c] / (float)count *
                ((float)count * dy[i] - sum_dy[c] - xhat * sum_dy_xhat[c]);
    }
}

extern "C" void launch_reduce_max_abs(const float *x, long long n, float *out, void *) {
    float m = 0.0f;
#pragma omp parallel for reduction(max : m) schedule(static)
    for (long long i = 0; i < n; ++i) {
        float v = std::fabs(x[i]);
        if (v > m) m = v;
    }
    *out = m;
}

extern "C" void launch_quant_f32_i8(const float *x, long long n, float scale, int8_t *q, void *) {
    float inv = 1.0f / scale;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        float v = std::min(std::max(x[i] * inv, -127.0f), 127.0f);
        q[i] = (int8_t)std::lrint(v);
    }
}

extern "C" void launch_relu_quant(const float *y, long long n, float scale, int8_t *q, void *) {
    float inv = 1.0f / scale;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        float v = std::min(std::max(y[i], 0.0f) * inv, 127.0f);
        q[i] = (int8_t)std::lrint(v);
    }
}

extern "C" void launch_dequant_i8_f32(const int8_t *q, long long n, float scale, float *x, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) x[i] = (float)q[i] * scale;
}

extern "C" void launch_add_scaled_i8(const int8_t *a, float sa, const int8_t *b, float sb,
                                     long long n, float *out, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) out[i] = (float)a[i] * sa + (float)b[i] * sb;
}

extern "C" void launch_epilogue_f32(const int32_t *acc, int acc_ld, float scale, float *out,
                                    int C, int Ho, int Wo, long long total, void *) {
    const long long hw = (long long)Ho * Wo;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < total; ++i) {
        long long b = i / ((long long)C * hw);
        int oc = (int)((i / hw) % C);
        long long row = b * hw + (i % hw);
        out[i] = (float)acc[row * (long long)acc_ld + oc] * scale;
    }
}

extern "C" void launch_avgpool_hw(const float *x, int C, int HW, long long B, float *out, void *) {
    long long bc = B * C;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < bc; ++i) {
        const float *p = x + i * HW;
        double s = 0.0;
        for (int j = 0; j < HW; ++j) s += p[j];
        out[i] = (float)(s / HW);
    }
}

extern "C" void launch_avgpool_hw_backward(const float *dy, int C, int HW, long long B, float *dx, void *) {
    long long bc = B * C;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < bc; ++i) {
        float v = dy[i] / (float)HW;
        float *p = dx + i * HW;
        for (int j = 0; j < HW; ++j) p[j] = v;
    }
}

extern "C" void launch_mul_gate(const float *x, const float *gate, int C, int HW, long long n,
                                float *out, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        int c = (int)((i / HW) % C);
        long long b = i / ((long long)C * HW);
        out[i] = x[i] * gate[b * C + c];
    }
}

extern "C" void launch_mask_relu_bwd(const float *y, const float *dy, long long n, float *dx, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) dx[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}

extern "C" void launch_zero_f32(float *p, long long n, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) p[i] = 0.0f;
}


extern "C" void launch_transpose_i8(const int8_t *src, int rows, int cols, int8_t *dst, void *) {
#pragma omp parallel for schedule(static) collapse(2)
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) dst[(long long)c * rows + r] = src[(long long)r * cols + c];
}


extern "C" void launch_add_f32(const float *a, const float *b, long long n, float *o, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) o[i] = a[i] + b[i];
}
extern "C" void launch_mul_f32(const float *a, const float *b, long long n, float *o, void *) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) o[i] = a[i] * b[i];
}
extern "C" void launch_meanpool_h(const float *x, int C, int H, int W, long long B, float *out, void *) {
    long long total = B * W * C;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < total; ++i) {
        int c = (int)(i % C);
        int w = (int)((i / C) % W);
        long long b = i / ((long long)C * W);
        float s = 0.0f;
        const float *p = x + ((b * C + c) * H) * W + w;
        for (int h = 0; h < H; ++h) s += p[(long long)h * W];
        out[i] = s / (float)H;
    }
}
extern "C" void launch_meanpool_h_backward(const float *dy, int C, int H, int W, long long B, float *dx, void *) {
    long long total = B * W * C;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < total; ++i) {
        int c = (int)(i % C);
        int w = (int)((i / C) % W);
        long long b = i / ((long long)C * W);
        float v = dy[i] / (float)H;
        float *p = dx + ((b * C + c) * H) * W + w;
        for (int h = 0; h < H; ++h) p[(long long)h * W] = v;
    }
}

extern "C" void upload_cnn_weights(const void *d, size_t s) {
    if (g_cnn_weights && d && s) std::memcpy(g_cnn_weights, d, s);
}
extern "C" void upload_bilstm_weights(const void *d, size_t s) {
    if (g_bilstm_weights && d && s) std::memcpy(g_bilstm_weights, d, s);
}
extern "C" void upload_attention_weights(const void *d, size_t s) {
    if (g_attention_weights && d && s) std::memcpy(g_attention_weights, d, s);
}
extern "C" void upload_ctc_weights(const void *d, size_t s) {
    if (g_ctc_weights && d && s) std::memcpy(g_ctc_weights, d, s);
}
extern "C" void *get_cnn_weights_ptr(void) { return g_cnn_weights; }
extern "C" void *get_bilstm_weights_ptr(void) { return g_bilstm_weights; }
extern "C" void *get_attention_weights_ptr(void) { return g_attention_weights; }
extern "C" void *get_ctc_weights_ptr(void) { return g_ctc_weights; }

extern "C" void free_model_weights(void) {
    std::free(g_cnn_weights); g_cnn_weights = nullptr;
    std::free(g_bilstm_weights); g_bilstm_weights = nullptr;
    std::free(g_attention_weights); g_attention_weights = nullptr;
    std::free(g_ctc_weights); g_ctc_weights = nullptr;
}
