//! INT8 quantisation-aware-training conv layer (3x3, and whatever else the
//! backbone asks for).
//!
//! Forward:  im2col(int8) -> tensor-core INT8 GEMM -> dequant / bias / ReLU /
//!           requant, all on device; only the int8 activation comes back.
//! Backward: three *fused* kernels (grad_W, grad_bias, grad_input) that read
//!           the activation and grad_output tensors directly.
//!
//! The invariant that matters: no backward buffer may be proportional to
//! `batch * Ho * Wo * Cin * kH * kW`.  That product is the im2col matrix, and
//! for a 3x3 conv on [256, 512, 4, 16] it is 38.6 GB.  Asking cudaMalloc for
//! it is what used to take the whole run down with "an illegal memory access
//! was encountered".  Peak scratch here is `batch*Cin*H*W*4` (~128 MB).

#![allow(dead_code, unused_variables)]

use crate::backend::*;
use rayon::prelude::*;

/// Quantisation scales, matching the reference Python port:
/// activations are stored as int8 with 1.0 <-> 50, weights with
/// weight_scale <-> 127, so an int32 accumulator maps back into the
/// activation domain by dividing by 127.
pub const ACC_DIVISOR: f32 = 127.0;
pub const OUT_MULTIPLIER: f32 = 50.0;

/// Leading batches whose backward runs in strict FP32.  FP16 accumulators
/// overflow during the first epochs, when the gradients are largest.
const WARMUP_BACKWARDS: usize = 200;

static BACKWARD_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn use_fp32_backward() -> bool {
    BACKWARD_COUNT.load(std::sync::atomic::Ordering::Relaxed) < WARMUP_BACKWARDS
}

fn increment_backward_count() {
    BACKWARD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// ============================================================
// Conv2dInt8
// ============================================================
pub struct Conv2dInt8 {
    pub weight_f32: Vec<f32>,
    pub weight_int8: Vec<i8>,
    /// weight_int8 transposed to [Cin*kH*kW, Cout] — the layout the
    /// col_major wmma fragment wants for the forward GEMM.
    pub bias: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2dInt8 {
    pub fn new(in_c: usize, out_c: usize, k: usize, stride: usize, padding: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let fan = (in_c * k * k + out_c * k * k) as f32;
        let limit = (6.0 / fan).sqrt();
        let weight_f32: Vec<f32> = (0..out_c * in_c * k * k)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let weight_int8 = Self::quantize(&weight_f32);
        Self {
            weight_f32,
            weight_int8,
            bias: vec![0.0; out_c],
            in_channels: in_c,
            out_channels: out_c,
            kernel_size: (k, k),
            stride: (stride, stride),
            padding: (padding, padding),
        }
    }

    fn quantize(w: &[f32]) -> Vec<i8> {
        let mx = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        w.iter()
            .map(|&x| (x / s).clamp(-127.0, 127.0).round() as i8)
            .collect()
    }

    /// Re-derive the INT8 weights from the FP32 master copy.
    ///
    /// `weight_int8` keeps the PyTorch `[Cout][Cin][kH][kW]` layout, which is
    /// *already* the `Bt[N,K]` operand that `launch_gemm_int8` expects
    /// (N = Cout, K = Cin*kH*kW).  An earlier revision additionally stored a
    /// `[col_cols][Cout]` transpose and handed that to the GEMM, so every
    /// conv output channel was computed from a permuted weight row — the
    /// network still ran, it just computed the wrong function.
    pub fn requantize(&mut self) {
        self.weight_int8 = Self::quantize(&self.weight_f32);
    }

    #[inline]
    pub fn col_cols(&self) -> usize {
        self.in_channels * self.kernel_size.0 * self.kernel_size.1
    }

    #[inline]
    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        (
            (h + 2 * ph - kh) / sh + 1,
            (w + 2 * pw - kw) / sw + 1,
        )
    }

    /// Muon (momentum-free): rescale the gradient by sqrt(min_dim)/||g||_F,
    /// then take a plain SGD step.  Returns the applied scale so the caller can
    /// log it.
    pub fn muon_update(&mut self, grad_w: &[f32], lr: f32, _momentum: f32) -> f32 {
        let n = self.out_channels * self.col_cols();
        if grad_w.len() < n || n == 0 {
            return 1.0;
        }
        let min_dim = self.out_channels.min(self.col_cols()) as f32;
        let frob_sq: f32 = grad_w[..n].par_iter().map(|g| g * g).sum();
        let frob = frob_sq.sqrt().max(1e-8);
        // Unbounded Frobenius normalisation would take a huge step on a
        // near-zero gradient; cap it like the reference implementation did.
        let scale = (min_dim.sqrt() / frob).min(10.0);

        let step = lr * scale;
        self.weight_f32[..n]
            .par_iter_mut()
            .zip(grad_w[..n].par_iter())
            .for_each(|(wi, gi)| {
                let updated = *wi - step * gi;
                *wi = if updated.is_finite() { updated } else { 0.0 };
            });
        scale
    }

    // --------------------------------------------------------
    // Forward — fully on device
    // --------------------------------------------------------
    pub fn forward_batched(
        &self,
        input: &[i8],
        batch: usize,
        h: usize,
        w: usize,
        stream: &CudaStream,
    ) -> (Vec<i8>, usize, usize) {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (h_out, w_out) = self.out_hw(h, w);

        let in_len = batch * self.in_channels * h * w;
        assert!(
            input.len() >= in_len,
            "forward_batched: input.len()={} < expected {} (B={} C={} H={} W={})",
            input.len(), in_len, batch, self.in_channels, h, w
        );
        assert!(
            h_out > 0 && w_out > 0,
            "forward_batched: degenerate output {}x{} for input {}x{}",
            h_out, w_out, h, w
        );

        let col_rows = batch * h_out * w_out;
        let col_cols = self.col_cols();
        let (m, n, k) = (col_rows, self.out_channels, col_cols);
        let (mp, np, kp) = (align16(m), align16(n), align16(k));
        let out_len = batch * self.out_channels * h_out * w_out;

        // Device scratch, carved out of the shared arena:
        const R_IN: usize = 0; //   int8 activation
        const R_COL: usize = 1; //  int8 col matrix, [mp, kp]
        const R_WT: usize = 2; //   int8 padded transposed weight, [np, kp]
        const R_BIAS: usize = 3; // f32 bias
        const R_ACC: usize = 4; //  int32 accumulator, [mp, np]
        const R_OUT: usize = 5; //  int8 output activation
        let ws = GpuWorkspace::new(&[
            in_len,
            mp * kp,
            np * kp,
            self.out_channels * 4,
            mp * np * 4,
            out_len,
        ]);

        // weight_int8 is [Cout][Cin*kH*kW] == [n][k] row-major, i.e. exactly
        // the Bt operand of the GEMM; only the 16-byte padding is added here.
        assert_eq!(self.weight_int8.len(), n * k, "weight_int8 has the wrong length");
        let (wt_pad, _, _) = pad_to_multiple_i8(&self.weight_int8, n, k, 16);
        debug_assert_eq!(wt_pad.len(), np * kp);

        ws.upload_i8(R_IN, &input[..in_len], stream);
        ws.upload_i8(R_WT, &wt_pad, stream);
        ws.upload_f32(R_BIAS, &self.bias, stream);

        unsafe {
            // im2col_pad writes every byte of the [mp, kp] matrix, padding
            // columns and padding rows included.
            launch_im2col_pad(
                ws.as_i8(R_IN),
                ws.as_i8_mut(R_COL),
                batch as i32,
                self.in_channels as i32,
                h as i32,
                w as i32,
                kh as i32,
                kw as i32,
                sh as i32,
                sw as i32,
                ph as i32,
                pw as i32,
                h_out as i32,
                w_out as i32,
                kp as i32,
                stream.0,
            );
            launch_gemm_int8(
                ws.as_i8(R_COL),
                ws.as_i8(R_WT),
                ws.as_i32_mut(R_ACC),
                m as i32,
                n as i32,
                k as i32,
                stream.0,
            );
            // launch_gemm_int8 writes C as a *compact* [M, N] matrix (only the
            // operands are padded to 16), so the accumulator leading dimension
            // is N == out_channels.  Passing align16(N) here made the epilogue
            // read row r at offset r*np instead of r*n, which silently produced
            // garbage (mostly zeros) for every layer whose out_channels is not
            // a multiple of 16 — and out-of-bounds reads once r*np ran past the
            // end of the accumulator region.
            launch_conv_epilogue(
                ws.ptr(R_ACC) as *const i32,
                ws.as_f32(R_BIAS),
                ws.as_i8_mut(R_OUT),
                n as i32,
                ACC_DIVISOR,
                OUT_MULTIPLIER,
                batch as i32,
                self.out_channels as i32,
                h_out as i32,
                w_out as i32,
                stream.0,
            );
        }

        let mut output = vec![0i8; out_len];
        ws.download_i8(R_OUT, &mut output, stream);
        stream.sync();
        (output, h_out, w_out)
    }

    // --------------------------------------------------------
    // Backward — fused, no im2col materialisation
    // --------------------------------------------------------
    /// Returns the gradient w.r.t. the layer input, shaped [batch, Cin, H, W].
    pub fn backward_qat_gpu(
        &mut self,
        input: &[i8],
        grad_output: &[f32],
        batch: usize,
        h_in: usize,
        w_in: usize,
        lr: f32,
        momentum: f32,
        stream: &CudaStream,
        diag: bool,
    ) -> Vec<f32> {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (h_out, w_out) = self.out_hw(h_in, w_in);
        let col_cols = self.col_cols();

        let in_len = batch * self.in_channels * h_in * w_in;
        let go_len = batch * self.out_channels * h_out * w_out;

        if batch == 0 || h_in == 0 || w_in == 0 || h_out == 0 || w_out == 0 {
            eprintln!(
                "    [backward_qat_gpu] degenerate shape B={} H={} W={} -> {}x{}",
                batch, h_in, w_in, h_out, w_out
            );
            return vec![0.0; in_len];
        }
        assert!(
            input.len() >= in_len,
            "backward_qat_gpu: input.len()={} < expected {}",
            input.len(), in_len
        );
        assert!(
            grad_output.len() >= go_len,
            "backward_qat_gpu: grad_output.len()={} < expected {} (B={} oc={} {}x{})",
            grad_output.len(), go_len, batch, self.out_channels, h_out, w_out
        );

        const R_IN: usize = 0; //    int8 activation          [in_len]
        const R_GO: usize = 1; //    f32 grad_output          [go_len]
        const R_GW: usize = 2; //    f32 grad_W               [out_c*col_cols]
        const R_GB: usize = 3; //    f32 grad_bias            [out_c]
        const R_W: usize = 4; //     f32 weight (post-update) [out_c*col_cols]
        const R_GI: usize = 5; //    f32 grad_input           [in_len]
        let ws = GpuWorkspace::new(&[
            in_len,
            go_len * 4,
            self.out_channels * col_cols * 4,
            self.out_channels * 4,
            self.out_channels * col_cols * 4,
            in_len * 4,
        ]);

        ws.upload_i8(R_IN, &input[..in_len], stream);
        ws.upload_f32(R_GO, &grad_output[..go_len], stream);
        ws.upload_f32(R_W, &self.weight_f32, stream);
        stream.sync();

        let fp32_mode = use_fp32_backward();
        increment_backward_count();

        // --- all three gradients, against the weights as they were ----------
        // grad_input must use the *pre-update* weights, so every device kernel
        // runs before the optimizer touches self.weight_f32.
        unsafe {
            launch_conv_backward_grad_weight(
                ws.as_i8(R_IN),
                ws.as_f32(R_GO),
                ws.as_f32_mut(R_GW),
                batch as i32,
                self.in_channels as i32,
                h_in as i32,
                w_in as i32,
                self.out_channels as i32,
                kh as i32,
                kw as i32,
                sh as i32,
                sw as i32,
                ph as i32,
                pw as i32,
                h_out as i32,
                w_out as i32,
                stream.0,
            );
            launch_conv_backward_grad_bias(
                ws.as_f32(R_GO),
                ws.as_f32_mut(R_GB),
                batch as i32,
                self.out_channels as i32,
                h_out as i32,
                w_out as i32,
                stream.0,
            );
            launch_conv_backward_grad_input(
                ws.as_f32(R_GO),
                ws.as_f32(R_W),
                ws.as_f32_mut(R_GI),
                batch as i32,
                self.in_channels as i32,
                h_in as i32,
                w_in as i32,
                self.out_channels as i32,
                kh as i32,
                kw as i32,
                sh as i32,
                sw as i32,
                ph as i32,
                pw as i32,
                h_out as i32,
                w_out as i32,
                stream.0,
            );
        }

        let wlen = self.out_channels * col_cols;
        let mut grad_w = vec![0.0f32; wlen];
        let mut grad_b = vec![0.0f32; self.out_channels];
        let mut grad_input = vec![0.0f32; in_len];
        ws.download_f32(R_GW, &mut grad_w, stream);
        ws.download_f32(R_GB, &mut grad_b, stream);
        ws.download_f32(R_GI, &mut grad_input, stream);
        stream.sync();

        if diag {
            let max = grad_w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let mean = grad_w.iter().map(|x| x.abs()).sum::<f32>() / wlen.max(1) as f32;
            let norm = grad_w.par_iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!(
                "    [DIAG] grad_W [{}]: max={:.4} mean={:.6} norm={:.2} (warmup left: {})",
                if fp32_mode { "FP32" } else { "FP16" },
                max,
                mean,
                norm,
                WARMUP_BACKWARDS.saturating_sub(
                    BACKWARD_COUNT.load(std::sync::atomic::Ordering::Relaxed)
                )
            );
        }

        // --- average over the batch, clip, Muon step -------------------------
        let inv_n = 1.0 / (batch * h_out * w_out) as f32;
        let max_grad_norm = 10000.0f32;
        let norm: f32 = grad_w
            .par_iter()
            .map(|g| {
                let s = g * inv_n;
                s * s
            })
            .sum::<f32>()
            .sqrt();
        let clip = if norm > max_grad_norm { max_grad_norm / norm } else { 1.0 };
        let combined = inv_n * clip;
        grad_w.par_iter_mut().for_each(|g| *g *= combined);

        let muon_scale = self.muon_update(&grad_w, lr, momentum);
        if diag {
            let max_delta = grad_w
                .iter()
                .map(|g| (lr * g * muon_scale).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "    [DIAG] clip norm={:.4} scale={:.4} | muon scale={:.4} max_delta={:.6} lr={:.6}",
                norm, clip, muon_scale, max_delta, lr
            );
        }

        // Bias: averaged SGD.  Muon's matrix normalisation is meaningless on a
        // vector, and it used to swallow the bias gradient entirely.
        let bias_step = lr;
        for (b, g) in self.bias.iter_mut().zip(grad_b.iter()) {
            let updated = *b - bias_step * g * inv_n;
            *b = if updated.is_finite() { updated } else { 0.0 };
        }

        self.requantize();

        if diag {
            eprintln!(
                "    [DIAG requant] weight_f32: max={:.6} mean={:.6} | grad_input max={:.6}",
                self.weight_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
                self.weight_f32.iter().map(|x| x.abs()).sum::<f32>()
                    / self.weight_f32.len().max(1) as f32,
                grad_input.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
            );
        }

        grad_input
    }

    pub fn param_count(&self) -> usize {
        self.weight_f32.len() + self.bias.len()
    }
}

// ============================================================
// Host-side helpers used by the backbone
// ============================================================

/// 2x2 max pooling that also records where each maximum came from, so the
/// backward pass can scatter gradients back without recomputing anything.
pub fn maxpool2x2_with_indices(
    input: &[i8],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> (Vec<i8>, Vec<u64>) {
    let ho = h / 2;
    let wo = w / 2;
    let mut out = vec![0i8; batch * channels * ho * wo];
    let mut idx = vec![0u64; batch * channels * ho * wo];

    let plane = channels * ho * wo;
    out.par_chunks_mut(plane)
        .zip(idx.par_chunks_mut(plane))
        .enumerate()
        .for_each(|(b, (bo, bi))| {
            for c in 0..channels {
                for oh in 0..ho {
                    for ow in 0..wo {
                        let base = ((b * channels + c) * h + oh * 2) * w + ow * 2;
                        let mut best = i8::MIN;
                        let mut best_i = base;
                        for dh in 0..2 {
                            for dw in 0..2 {
                                let i = base + dh * w + dw;
                                if i < input.len() && input[i] > best {
                                    best = input[i];
                                    best_i = i;
                                }
                            }
                        }
                        let o = (c * ho + oh) * wo + ow;
                        bo[o] = best;
                        bi[o] = best_i as u64;
                    }
                }
            }
        });
    (out, idx)
}

pub fn maxpool2x2_batched(
    input: &[i8],
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Vec<i8> {
    maxpool2x2_with_indices(input, batch, c, h, w).0
}
