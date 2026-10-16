//! Correctness tests for the native backend and the quantised layers.
//!
//! These run against whichever backend build.rs linked: with CUDA they check
//! the real kernels on the GPU, without it they check the CPU reference
//! implementation that mirrors them line for line.  Either way they catch the
//! class of bug that produced "an illegal memory access was encountered" —
//! wrong buffer sizes, wrong leading dimensions, wrong index arithmetic.

use crate::backend::*;
use crate::conv2d::{maxpool2x2_with_indices, Conv2dInt8, ACC_DIVISOR, OUT_MULTIPLIER};
use crate::linear::LinearInt8;

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn as_bytes_mut_i32(v: &mut [i32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}

fn i32_as_bytes(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random in [-1, 1] (no dependency on rand's RNG state).
fn det(i: usize, seed: usize) -> f32 {
    let x = ((i.wrapping_mul(2654435761)) ^ seed) as u64;
    let x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((x >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn naive_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ------------------------------------------------------------
// GEMM tests
// ------------------------------------------------------------
#[test]
fn int8_gemm_matches_naive_including_unpadded_shapes() {
    let stream = CudaStream::new();
    // Deliberately not multiples of 16: the kernel must pad internally and
    // still write an exactly [M, N] result.
    for &(m, n, k) in &[(16usize, 16usize, 16usize), (17, 5, 3), (100, 33, 27), (256, 192, 512)] {
        let a: Vec<i8> = (0..m * k).map(|i| (det(i, 1) * 100.0) as i8).collect();
        let b: Vec<i8> = (0..n * k).map(|i| (det(i, 2) * 100.0) as i8).collect();

        let (a_pad, _, _) = pad_to_multiple_i8(&a, m, k, 16);
        let (b_pad, _, _) = pad_to_multiple_i8(&b, n, k, 16);
        let ws = GpuWorkspace::new(&[a_pad.len(), b_pad.len(), m * n * 4]);
        ws.upload_i8(0, &a_pad, &stream);
        ws.upload_i8(1, &b_pad, &stream);

        let mut got = vec![0i32; m * n];
        unsafe {
            launch_gemm_int8(ws.as_i8(0), ws.as_i8(1), ws.as_i32_mut(2), m as i32, n as i32, k as i32, stream.0);
        }
        ws.download(2, as_bytes_mut_i32(&mut got), &stream);
        stream.sync();

        let mut want = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i32;
                for p in 0..k {
                    s += a[i * k + p] as i32 * b[j * k + p] as i32;
                }
                want[i * n + j] = s;
            }
        }
        assert_eq!(got, want, "int8 gemm mismatch at M={m} N={n} K={k}");
    }
}

#[test]
fn fp32_gemm_matches_naive() {
    let stream = CudaStream::new();
    for &(m, n, k) in &[(1usize, 1usize, 1usize), (33, 5, 17), (64, 64, 64), (200, 40, 7)] {
        let a: Vec<f32> = (0..m * k).map(|i| det(i, 3)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| det(i, 4)).collect();
        let ws = GpuWorkspace::new(&[m * k * 4, k * n * 4, m * n * 4]);
        ws.upload(0, &f32_bytes(&a), &stream);
        ws.upload(1, &f32_bytes(&b), &stream);

        let mut got = vec![0.0f32; m * n];
        unsafe {
            launch_gemm_fp32(ws.as_f32(0), ws.as_f32(1), ws.as_f32_mut(2), m as i32, n as i32, k as i32, stream.0);
        }
        ws.download_f32(2, &mut got, &stream);
        stream.sync();

        let want = naive_gemm(&a, &b, m, n, k);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-3 + 1e-3 * want.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                "fp32 gemm mismatch at M={m} N={n} K={k}: {d}");
    }
}

#[test]
fn fp32_gemm_handles_huge_k_without_grid_overflow() {
    // The old kernel used dim3 grid((N+15)/16, (M+15)/16); gridDim.y is capped
    // at 65535, so any M above ~1M silently launched nothing and left the
    // output buffer untouched.  This checks a shape that would have hit it.
    let stream = CudaStream::new();
    let (m, n, k) = (70000usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
    let ws = GpuWorkspace::new(&[m * k * 4, k * n * 4, m * n * 4]);
    ws.upload(0, &f32_bytes(&a), &stream);
    ws.upload(1, &f32_bytes(&b), &stream);

    let mut got = vec![-1.0f32; m * n];
    unsafe {
        launch_gemm_fp32(ws.as_f32(0), ws.as_f32(1), ws.as_f32_mut(2), m as i32, n as i32, k as i32, stream.0);
    }
    ws.download_f32(2, &mut got, &stream);
    stream.sync();

    let want = naive_gemm(&a, &b, m, n, k);
    assert_eq!(max_abs_diff(&got, &want), 0.0, "large-M fp32 gemm produced wrong values");
    assert!(got.iter().all(|x| x.is_finite()));
}

// ------------------------------------------------------------
// im2col / epilogue
// ------------------------------------------------------------
fn ref_im2col(input: &[i8], b: usize, c: usize, h: usize, w: usize,
              kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> Vec<i8> {
    let ho = (h + 2 * ph - kh) / sh + 1;
    let wo = (w + 2 * pw - kw) / sw + 1;
    let cols = c * kh * kw;
    let mut out = vec![0i8; b * ho * wo * cols];
    for bi in 0..b {
        for oh in 0..ho {
            for ow in 0..wo {
                let row = (bi * ho + oh) * wo + ow;
                let mut p = 0;
                for ci in 0..c {
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let ih = oh * sh + khi;
                            let iw = ow * sw + kwi;
                            let v = if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                                input[((bi * c + ci) * h + (ih - ph)) * w + (iw - pw)]
                            } else {
                                0
                            };
                            out[row * cols + p] = v;
                            p += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn im2col_pad_matches_reference() {
    let stream = CudaStream::new();
    let (b, c, h, w, kh, sh, ph) = (3usize, 5usize, 9usize, 11usize, 3usize, 2usize, 1usize);
    let ho = (h + 2 * ph - kh) / sh + 1;
    let wo = (w + 2 * ph - kh) / sh + 1;
    let cols = c * kh * kh;
    let kp = align16(cols);
    let rows = b * ho * wo;
    let mp = align16(rows);

    let input: Vec<i8> = (0..b * c * h * w).map(|i| (det(i, 7) * 100.0) as i8).collect();
    let want = ref_im2col(&input, b, c, h, w, kh, kh, sh, sh, ph, ph);
    let ws = GpuWorkspace::new(&[input.len(), mp * kp]);
    ws.upload_i8(0, &input, &stream);
    unsafe {
        launch_im2col_pad(ws.as_i8(0), ws.as_i8_mut(1), b as i32, c as i32, h as i32, w as i32,
                          kh as i32, kh as i32, sh as i32, sh as i32, ph as i32, ph as i32,
                          ho as i32, wo as i32, kp as i32, stream.0);
    }
    stream.sync();

    let mut got = vec![0i8; mp * kp];
    ws.download_i8(1, &mut got, &stream);
    stream.sync();

    for r in 0..rows {
        for cc in 0..cols {
            assert_eq!(got[r * kp + cc], want[r * cols + cc], "im2col mismatch at row {r} col {cc}");
        }
        for cc in cols..kp {
            assert_eq!(got[r * kp + cc], 0, "im2col padding not zero at row {r} col {cc}");
        }
    }
    for r in rows..mp {
        for cc in 0..kp {
            assert_eq!(got[r * kp + cc], 0, "im2col row padding not zero at row {r}");
        }
    }
}

#[test]
fn conv_epilogue_dequant_bias_relu_requant() {
    let stream = CudaStream::new();
    let (b, c, ho, wo) = (2usize, 5usize, 3usize, 4usize);
    // launch_gemm_int8 produces a compact [rows, C] matrix, so the leading
    // dimension of the accumulator is C itself (no 16-byte padding).
    let acc_ld = c;
    let acc: Vec<i32> = (0..b * ho * wo * acc_ld).map(|i| (det(i, 9) * 3000.0) as i32).collect();
    let bias: Vec<f32> = (0..c).map(|i| det(i, 11)).collect();
    let out_len = b * c * ho * wo;
    let ws = GpuWorkspace::new(&[acc.len() * 4, c * 4, out_len]);
    ws.upload(0, i32_as_bytes(&acc), &stream);
    ws.upload(1, &f32_bytes(&bias), &stream);
    unsafe {
        launch_conv_epilogue(ws.ptr(0) as *const i32, ws.as_f32(1), ws.as_i8_mut(2),
                             acc_ld as i32, ACC_DIVISOR, OUT_MULTIPLIER,
                             b as i32, c as i32, ho as i32, wo as i32, stream.0);
    }
    stream.sync();

    let mut got = vec![0i8; out_len];
    ws.download_i8(2, &mut got, &stream);
    stream.sync();

    let hw = ho * wo;
    for i in 0..out_len {
        // out is [B, C, Ho, Wo]; the accumulator rows are [B*Ho*Wo] x [C]
        let bi = i / (c * hw);
        let oc = (i / hw) % c;
        let row = bi * hw + (i % hw);
        let v = acc[row * acc_ld + oc] as f32 / ACC_DIVISOR + bias[oc];
        let want = (v.max(0.0) * OUT_MULTIPLIER).clamp(-127.0, 127.0).round() as i8;
        assert_eq!(got[i], want, "epilogue mismatch at {i}");
    }
}

// ------------------------------------------------------------
// Fused conv backward
// ------------------------------------------------------------
struct ConvShape {
    b: usize, cin: usize, h: usize, w: usize, cout: usize,
    kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize,
    ho: usize, wo: usize,
}

fn shape(b: usize, cin: usize, h: usize, w: usize, cout: usize, k: usize, s: usize, p: usize) -> ConvShape {
    ConvShape {
        b, cin, h, w, cout,
        kh: k, kw: k, sh: s, sw: s, ph: p, pw: p,
        ho: (h + 2 * p - k) / s + 1,
        wo: (w + 2 * p - k) / s + 1,
    }
}

/// grad_W reference: build the col matrix, then grad_W = go_mat^T @ col.
fn ref_grad_w(s: &ConvShape, input: &[i8], go: &[f32]) -> Vec<f32> {
    let cols = s.cin * s.kh * s.kw;
    let hw_out = s.ho * s.wo;
    let rows = s.b * hw_out;
    let mut col = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let b = r / hw_out;
        let rem = r % hw_out;
        let oh = rem / s.wo;
        let ow = rem % s.wo;
        for ci in 0..s.cin {
            for khi in 0..s.kh {
                for kwi in 0..s.kw {
                    let ih = oh * s.sh + khi;
                    let iw = ow * s.sw + kwi;
                    let v = if ih >= s.ph && ih < s.h + s.ph && iw >= s.pw && iw < s.w + s.pw {
                        input[((b * s.cin + ci) * s.h + (ih - s.ph)) * s.w + (iw - s.pw)] as f32
                    } else {
                        0.0
                    };
                    col[r * cols + (ci * s.kh + khi) * s.kw + kwi] = v;
                }
            }
        }
    }
    // go is [b, cout, ho, wo]; reshape to [rows, cout]
    let mut go_mat = vec![0.0f32; rows * s.cout];
    for b in 0..s.b {
        for oc in 0..s.cout {
            for p in 0..hw_out {
                go_mat[(b * hw_out + p) * s.cout + oc] = go[(b * s.cout + oc) * hw_out + p];
            }
        }
    }
    let mut gw = vec![0.0f32; s.cout * cols];
    for oc in 0..s.cout {
        for ci in 0..cols {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += go_mat[r * s.cout + oc] * col[r * cols + ci];
            }
            gw[oc * cols + ci] = acc;
        }
    }
    gw
}

fn ref_grad_input(s: &ConvShape, go: &[f32], weight: &[f32]) -> Vec<f32> {
    let cols = s.cin * s.kh * s.kw;
    let hw_out = s.ho * s.wo;
    let mut gi = vec![0.0f32; s.b * s.cin * s.h * s.w];
    for b in 0..s.b {
        for ci in 0..s.cin {
            for ih in 0..s.h {
                for iw in 0..s.w {
                    let mut acc = 0.0f32;
                    for khi in 0..s.kh {
                        let nh = ih + s.ph;
                        if nh < khi { continue; }
                        let numh = nh - khi;
                        if numh % s.sh != 0 { continue; }
                        let oh = numh / s.sh;
                        if oh >= s.ho { continue; }
                        for kwi in 0..s.kw {
                            let nw = iw + s.pw;
                            if nw < kwi { continue; }
                            let numw = nw - kwi;
                            if numw % s.sw != 0 { continue; }
                            let ow = numw / s.sw;
                            if ow >= s.wo { continue; }
                            let wcol = (ci * s.kh + khi) * s.kw + kwi;
                            for oc in 0..s.cout {
                                acc += go[(b * s.cout + oc) * hw_out + oh * s.wo + ow]
                                    * weight[oc * cols + wcol];
                            }
                        }
                    }
                    gi[(b * s.cin + ci) * s.h * s.w + ih * s.w + iw] = acc;
                }
            }
        }
    }
    gi
}

#[test]
fn fused_backward_grad_w_matches_im2col_reference() {
    let stream = CudaStream::new();
    for s in [shape(2, 3, 8, 7, 5, 3, 1, 1), shape(1, 4, 6, 6, 8, 3, 2, 1), shape(3, 2, 5, 5, 3, 1, 1, 0)] {
        let input: Vec<i8> = (0..s.b * s.cin * s.h * s.w).map(|i| (det(i, 13) * 100.0) as i8).collect();
        let go: Vec<f32> = (0..s.b * s.cout * s.ho * s.wo).map(|i| det(i, 17)).collect();
        let want = ref_grad_w(&s, &input, &go);
        let ws = GpuWorkspace::new(&[input.len(), go.len() * 4, want.len() * 4]);
        ws.upload_i8(0, &input, &stream);
        ws.upload(1, &f32_bytes(&go), &stream);
        unsafe {
            launch_conv_backward_grad_weight(
                ws.as_i8(0), ws.as_f32(1), ws.as_f32_mut(2),
                s.b as i32, s.cin as i32, s.h as i32, s.w as i32, s.cout as i32,
                s.kh as i32, s.kw as i32, s.sh as i32, s.sw as i32, s.ph as i32, s.pw as i32,
                s.ho as i32, s.wo as i32, stream.0);
        }
        stream.sync();
        let mut got = vec![0u8; want.len() * 4];
        ws.download(2, &mut got, &stream);
        stream.sync();
        let got = bytes_f32(&got);

        let scale = want.iter().map(|x| x.abs()).fold(1.0f32, f32::max);
        let d = max_abs_diff(&got, &want);
        assert!(d <= 1e-3 * scale, "grad_W mismatch (max diff {d}, scale {scale})");
    }
}

#[test]
fn fused_backward_grad_bias_is_column_sum() {
    let stream = CudaStream::new();
    let s = shape(4, 3, 6, 5, 7, 3, 1, 1);
    let go: Vec<f32> = (0..s.b * s.cout * s.ho * s.wo).map(|i| det(i, 19)).collect();

    let mut want = vec![0.0f32; s.cout];
    let hw = s.ho * s.wo;
    for b in 0..s.b {
        for oc in 0..s.cout {
            for p in 0..hw {
                want[oc] += go[(b * s.cout + oc) * hw + p];
            }
        }
    }
    let ws = GpuWorkspace::new(&[go.len() * 4, s.cout * 4]);
    ws.upload(0, &f32_bytes(&go), &stream);
    unsafe {
        launch_conv_backward_grad_bias(ws.as_f32(0), ws.as_f32_mut(1),
                                       s.b as i32, s.cout as i32, s.ho as i32, s.wo as i32, stream.0);
    }
    stream.sync();
    let mut got = vec![0u8; s.cout * 4];
    ws.download(1, &mut got, &stream);
    stream.sync();
    let got = bytes_f32(&got);

    let scale = want.iter().map(|x| x.abs()).fold(1.0f32, f32::max);
    let d = max_abs_diff(&got, &want);
    assert!(d <= 1e-3 * scale, "grad_bias mismatch (max diff {d}, scale {scale})");
}

#[test]
fn fused_backward_grad_input_matches_col2im_reference() {
    let stream = CudaStream::new();
    for s in [shape(2, 3, 7, 6, 4, 3, 1, 1), shape(1, 2, 8, 8, 3, 3, 2, 1), shape(2, 2, 4, 9, 2, 1, 1, 0)] {
        let go: Vec<f32> = (0..s.b * s.cout * s.ho * s.wo).map(|i| det(i, 23)).collect();
        let cols = s.cin * s.kh * s.kw;
        let weight: Vec<f32> = (0..s.cout * cols).map(|i| det(i, 29)).collect();
        let want = ref_grad_input(&s, &go, &weight);
        let ws = GpuWorkspace::new(&[go.len() * 4, weight.len() * 4, want.len() * 4]);
        ws.upload(0, &f32_bytes(&go), &stream);
        ws.upload(1, &f32_bytes(&weight), &stream);
        unsafe {
            launch_conv_backward_grad_input(
                ws.as_f32(0), ws.as_f32(1), ws.as_f32_mut(2),
                s.b as i32, s.cin as i32, s.h as i32, s.w as i32, s.cout as i32,
                s.kh as i32, s.kw as i32, s.sh as i32, s.sw as i32, s.ph as i32, s.pw as i32,
                s.ho as i32, s.wo as i32, stream.0);
        }
        stream.sync();
        let mut got = vec![0u8; want.len() * 4];
        ws.download(2, &mut got, &stream);
        stream.sync();
        let got = bytes_f32(&got);

        let scale = want.iter().map(|x| x.abs()).fold(1.0f32, f32::max);
        let d = max_abs_diff(&got, &want);
        assert!(d <= 1e-3 * scale + 1e-3, "grad_input mismatch (max diff {d}, scale {scale})");
    }
}

// ------------------------------------------------------------
// Conv layer
// ------------------------------------------------------------
#[test]
fn conv_forward_matches_quantised_reference() {
    let stream = CudaStream::new();
    let mut conv = Conv2dInt8::new(3, 8, 3, 1, 1);
    conv.bias = (0..8).map(|i| det(i, 31) * 0.1).collect();

    let (b, h, w) = (2usize, 9usize, 11usize);
    let input: Vec<i8> = (0..b * 3 * h * w).map(|i| (det(i, 37) * 100.0) as i8).collect();
    let (out, ho, wo) = conv.forward_batched(&input, b, h, w, &stream);

    // reference: im2col in int8, exact integer dot product with weight_int8,
    // then the same dequant/bias/ReLU/requant epilogue.
    // signature: (input, b, c, h, w, kh, kw, sh, sw, ph, pw)
    let col = ref_im2col(&input, b, 3, h, w, 3, 3, 1, 1, 1, 1);
    let cols = 3 * 3 * 3;
    assert_eq!(out.len(), b * 8 * ho * wo);
    for bi in 0..b {
        for p in 0..ho * wo {
            let row = bi * ho * wo + p;
            for oc in 0..8 {
                let mut acc = 0i32;
                for cc in 0..cols {
                    // weight_int8 is [out_channels][col_cols]
                    acc += col[row * cols + cc] as i32 * conv.weight_int8[oc * cols + cc] as i32;
                }
                let v = acc as f32 / ACC_DIVISOR + conv.bias[oc];
                let want = (v.max(0.0) * OUT_MULTIPLIER).clamp(-127.0, 127.0).round() as i8;
                let got = out[(bi * 8 + oc) * ho * wo + p];
                assert_eq!(got, want, "conv forward mismatch b={bi} p={p} oc={oc}");
            }
        }
    }
}

#[test]
fn conv_backward_shapes_and_finite() {
    let stream = CudaStream::new();
    let mut conv = Conv2dInt8::new(4, 6, 3, 1, 1);
    let (b, h, w) = (2usize, 7usize, 5usize);
    let (ho, wo) = conv.out_hw(h, w);

    let input: Vec<i8> = (0..b * 4 * h * w).map(|i| (det(i, 41) * 100.0) as i8).collect();
    let go: Vec<f32> = (0..b * 6 * ho * wo).map(|i| det(i, 43) * 0.01).collect();

    let w_before = conv.weight_f32.clone();
    let gi = conv.backward_qat_gpu(&input, &go, b, h, w, 1e-3, 0.9, &stream, false);

    assert_eq!(gi.len(), b * 4 * h * w, "grad_input length");
    assert!(gi.iter().all(|x| x.is_finite()), "grad_input has NaN/Inf");
    assert!(conv.weight_f32.iter().all(|x| x.is_finite()), "weights have NaN/Inf");
    assert!(conv.bias.iter().all(|x| x.is_finite()), "bias has NaN/Inf");
    // The optimizer must actually move the weights.
    let moved = conv.weight_f32.iter().zip(w_before.iter()).filter(|(a, b2)| (*a - *b2).abs() > 0.0).count();
    assert!(moved > 0, "backward did not update any weight");
    assert_eq!(conv.weight_int8.len(), conv.weight_f32.len());
}

// ------------------------------------------------------------
// Maxpool
// ------------------------------------------------------------
#[test]
fn maxpool_forward_backward_roundtrip() {
    let (b, c, h, w) = (2usize, 3usize, 6usize, 8usize);
    let input: Vec<i8> = (0..b * c * h * w).map(|i| (det(i, 47) * 100.0) as i8).collect();
    let (pooled, idx) = maxpool2x2_with_indices(&input, b, c, h, w);

    let (ho, wo) = (h / 2, w / 2);
    assert_eq!(pooled.len(), b * c * ho * wo);
    for (i, &v) in pooled.iter().enumerate() {
        assert_eq!(input[idx[i] as usize], v, "maxpool index does not point at the maximum");
    }

    // backward: every output gradient lands on exactly its argmax input
    let go: Vec<f32> = (0..b * c * ho * wo).map(|i| det(i, 53)).collect();
    let mut gi = vec![0.0f32; b * c * h * w];
    unsafe {
        maxpool2x2_backward_host(go.as_ptr(), idx.as_ptr(), gi.as_mut_ptr(),
                                 go.len() as u64, gi.len() as u64);
    }
    let mut expected = vec![0.0f32; b * c * h * w];
    for (i, &g) in go.iter().enumerate() {
        expected[idx[i] as usize] += g;
    }
    assert_eq!(gi, expected);
}

// ------------------------------------------------------------
// Linear
// ------------------------------------------------------------
#[test]
fn linear_forward_matches_int8_reference() {
    let stream = CudaStream::new();
    let lin = LinearInt8::new(20, 7);
    let rows = 5;
    let inp: Vec<i8> = (0..rows * 20).map(|i| (det(i, 59) * 100.0) as i8).collect();
    let out = lin.forward(&inp, &stream);
    assert_eq!(out.len(), rows * 7);

    for r in 0..rows {
        for o in 0..7 {
            let mut acc = 0i32;
            for i in 0..20 {
                acc += inp[r * 20 + i] as i32 * lin.weight_int8[o * 20 + i] as i32;
            }
            assert_eq!(out[r * 7 + o], acc, "linear mismatch r={r} o={o}");
        }
    }
}

#[test]
fn linear_backward_gradient_matches_finite_differences() {
    let mut lin = LinearInt8::new(6, 4);
    lin.weight_f32 = (0..4 * 6).map(|i| det(i, 61) * 0.5).collect();
    lin.requantize();

    let rows = 3;
    let inp: Vec<f32> = (0..rows * 6).map(|i| det(i, 67)).collect();
    // Arbitrary scalar function of the layer output: sum of squares.
    let out: Vec<f32> = (0..rows * 4)
        .map(|i| {
            let r = i / 4;
            let o = i % 4;
            (0..6).map(|k| inp[r * 6 + k] * lin.weight_f32[o * 6 + k]).sum::<f32>()
        })
        .collect();
    let go: Vec<f32> = out.iter().map(|x| 2.0 * x / rows as f32).collect();

    let lr = 1.0f32;
    let before = lin.weight_f32.clone();
    let clip = lin.backward(&inp, &go, lr, 0.0);

    // LinearInt8::backward averages the outer product over the batch, so
    // grad_W[o,i] = (1/rows) * sum_r go[r,o] * inp[r,i].
    for o in 0..4 {
        for i in 0..6 {
            let mut g = 0.0f32;
            for r in 0..rows {
                g += go[r * 4 + o] * inp[r * 6 + i];
            }
            let g = g / rows as f32;
            let want = before[o * 6 + i] - lr * clip * g;
            let got = lin.weight_f32[o * 6 + i];
            // The kernel accumulates in f32 over the batch, the reference in
            // a different order; allow the corresponding rounding slack.
            let tol = 1e-5 * want.abs().max(1.0) + 1e-4;
            assert!((got - want).abs() < tol, "linear grad mismatch at o={o} i={i}: {got} vs {want}");
        }
    }
}

// ------------------------------------------------------------
// Backbone
// ------------------------------------------------------------
#[test]
fn backbone_forward_shape_and_backward_chain() {
    let stream = CudaStream::new();
    let mut cnn = crate::seresnet::SEResNet31::new();
    // Keep the test cheap: shrink every layer to a couple of channels would
    // change the architecture, so instead use a single small image.
    let b = 1usize;
    let input = vec![7i8; b * 3 * 64 * 256];

    let r = cnn.forward_save(&input, b, &stream);
    // 5 maxpools: 64x256 -> 2x8
    assert_eq!(r.out_h, 2, "output height should be 64/32");
    assert_eq!(r.out_w, 8, "output width should be 256/32");
    assert_eq!(r.output.len(), b * 512 * 2 * 8);
    // 19 conv + 5 maxpool activations
    assert_eq!(r.activations.len(), 24, "activation count");

    let grad = vec![0.001f32; b * 512 * 2 * 8];
    let last_before = cnn.conv_out.weight_f32.clone();
    let first_before = cnn.conv0_0.weight_f32.clone();

    // stage 98 == train only the last 4 conv layers
    let _ = cnn.backward_chain(&r, &grad, 1e-4, 0.9, &stream, 98, false);

    let last_moved = cnn.conv_out.weight_f32.iter().zip(last_before.iter())
        .any(|(a, b2)| (*a - *b2).abs() > 0.0);
    let first_moved = cnn.conv0_0.weight_f32.iter().zip(first_before.iter())
        .any(|(a, b2)| (*a - *b2).abs() > 0.0);
    assert!(last_moved, "the last conv layer should have been trained");
    assert!(!first_moved, "a frozen stem layer must not be touched at stage 98");
}

#[test]
fn workspace_regions_do_not_overlap() {
    let ws = GpuWorkspace::new(&[1000, 1, 4096, 7]);
    let p0 = ws.ptr(0) as usize;
    let p1 = ws.ptr(1) as usize;
    let p2 = ws.ptr(2) as usize;
    let p3 = ws.ptr(3) as usize;
    assert!(p1 >= p0 + 1000);
    assert!(p2 >= p1 + 1);
    assert!(p3 >= p2 + 4096);
    assert_eq!(p0 % 16, 0, "regions must be at least 16-byte aligned");
    assert_eq!(p1 % 16, 0);
}
