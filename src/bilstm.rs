//! BiLSTM over the pooled CNN features, INT8 tensor-core GEMMs for the gate
//! projections and FP32 for the recurrent non-linearities.
//!
//! The current OCR head only needs a single time step (the CNN output is mean
//! pooled to one vector per image), but the sequence path is kept intact so a
//! real CTC/attention decoder can be dropped in later.

#![allow(dead_code, unused_variables)]

use crate::backend::*;
use rayon::prelude::*;

pub struct LSTMInt8 {
    pub weight_ih_f32: Vec<f32>,
    pub weight_ih_int8: Vec<i8>,
    pub weight_hh_f32: Vec<f32>,
    pub weight_hh_int8: Vec<i8>,
    pub bias: Vec<f32>,
    pub input_size: usize,
    pub hidden_size: usize,
}

/// The int8 activations are stored with 1.0 <-> ACTIVATION_SCALE, and the
/// hidden state is quantised with the same scale before the h GEMM.
///
/// Anything that feeds the BiLSTM output into another int8 GEMM has to undo
/// *both* int8 scales of the accumulator (activation and weight); see the
/// generator dequantisation in main.rs.
pub const ACTIVATION_SCALE: f32 = 10.0;

impl LSTMInt8 {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let limit_ih = (6.0 / (input_size + 4 * hidden_size) as f32).sqrt();
        let limit_hh = (6.0 / (hidden_size + 4 * hidden_size) as f32).sqrt();
        let weight_ih_f32: Vec<f32> = (0..4 * hidden_size * input_size)
            .map(|_| rng.gen_range(-limit_ih..limit_ih))
            .collect();
        let weight_hh_f32: Vec<f32> = (0..4 * hidden_size * hidden_size)
            .map(|_| rng.gen_range(-limit_hh..limit_hh))
            .collect();
        let bias = vec![0.0; 4 * hidden_size];
        let weight_ih_int8 = Self::quantize(&weight_ih_f32);
        let weight_hh_int8 = Self::quantize(&weight_hh_f32);
        Self {
            weight_ih_f32,
            weight_ih_int8,
            weight_hh_f32,
            weight_hh_int8,
            bias,
            input_size,
            hidden_size,
        }
    }

    fn quantize(w: &[f32]) -> Vec<i8> {
        let mx = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        w.iter()
            .map(|&x| (x / s).clamp(-127.0, 127.0).round() as i8)
            .collect()
    }

    pub fn requantize(&mut self) {
        self.weight_ih_int8 = Self::quantize(&self.weight_ih_f32);
        self.weight_hh_int8 = Self::quantize(&self.weight_hh_f32);
    }

    /// C[batch, 4*hidden] = x[batch, input] @ W_ih^T + h[batch, hidden] @ W_hh^T + bias,
    /// followed by the LSTM cell update.
    pub fn forward_step(
        &self,
        x_int8: &[i8],
        h_prev: &[f32],
        c_prev: &[f32],
        batch_size: usize,
        stream: &CudaStream,
    ) -> (Vec<f32>, Vec<f32>) {
        let hs = self.hidden_size;
        let gates = 4 * hs;

        let gates_i = Self::gemm_int8(x_int8, &self.weight_ih_int8, batch_size, gates, self.input_size, stream);
        let h_int8: Vec<i8> = h_prev
            .iter()
            .map(|&x| (x * ACTIVATION_SCALE).clamp(-127.0, 127.0).round() as i8)
            .collect();
        let gates_h = Self::gemm_int8(&h_int8, &self.weight_hh_int8, batch_size, gates, hs, stream);

        // int32 accumulators from two int8 GEMMs.  Each operand carries a
        // scale of 1/ACTIVATION_SCALE (activations) and 1/127 (weights), so
        // dividing by 127 lands back in the activation domain — same
        // convention as the conv layers.
        let inv = 1.0 / (crate::conv2d::ACC_DIVISOR * ACTIVATION_SCALE);

        let mut h_new = vec![0.0f32; batch_size * hs];
        let mut c_new = vec![0.0f32; batch_size * hs];

        for b in 0..batch_size {
            for i in 0..hs {
                let row = b * gates;
                let pre_i = (gates_i[row + i] as f32 + gates_h[row + i] as f32) * inv + self.bias[i];
                let pre_f = (gates_i[row + hs + i] as f32 + gates_h[row + hs + i] as f32) * inv + self.bias[hs + i];
                let pre_g = (gates_i[row + 2 * hs + i] as f32 + gates_h[row + 2 * hs + i] as f32) * inv + self.bias[2 * hs + i];
                let pre_o = (gates_i[row + 3 * hs + i] as f32 + gates_h[row + 3 * hs + i] as f32) * inv + self.bias[3 * hs + i];

                let ig = sigmoid(pre_i);
                let fg = sigmoid(pre_f);
                let og = sigmoid(pre_o);
                let cg = pre_g.tanh();
                let c = fg * c_prev[b * hs + i] + ig * cg;
                c_new[b * hs + i] = c;
                h_new[b * hs + i] = og * c.tanh();
            }
        }
        (h_new, c_new)
    }

    /// Input gate projection: [batch, input_size] int8 -> [batch, 4*hidden] int32.
    pub fn i2h_forward(&self, x_int8: &[i8], batch: usize, stream: &CudaStream) -> Vec<i32> {
        Self::gemm_int8(x_int8, &self.weight_ih_int8, batch, 4 * self.hidden_size, self.input_size, stream)
    }

    /// Recurrent gate projection: [batch, hidden] int8 -> [batch, 4*hidden] int32.
    pub fn h2h_forward(&self, h_int8: &[i8], batch: usize, stream: &CudaStream) -> Vec<i32> {
        Self::gemm_int8(h_int8, &self.weight_hh_int8, batch, 4 * self.hidden_size, self.hidden_size, stream)
    }

    /// INT8 GEMM with the operand padding the tensor-core kernel requires:
    /// A is [align16(m), align16(k)], B is [align16(n), align16(k)], and the
    /// result is exactly [m, n].
    pub fn gemm_int8(
        a: &[i8],
        b: &[i8],
        m: usize,
        n: usize,
        k: usize,
        stream: &CudaStream,
    ) -> Vec<i32> {
        let (mp, np, kp) = (align16(m), align16(n), align16(k));
        let (a_pad, _, _) = crate::backend::pad_to_multiple_i8(a, m, k, 16);
        let (b_pad, _, _) = crate::backend::pad_to_multiple_i8(b, n, k, 16);
        debug_assert_eq!(a_pad.len(), mp * kp);
        debug_assert_eq!(b_pad.len(), np * kp);

        const R_A: usize = 0;
        const R_B: usize = 1;
        const R_C: usize = 2;
        let ws = GpuWorkspace::new(&[a_pad.len(), b_pad.len(), m * n * 4]);
        ws.upload_i8(R_A, &a_pad, stream);
        ws.upload_i8(R_B, &b_pad, stream);

        let mut out = vec![0i32; m * n];
        unsafe {
            launch_gemm_int8(
                ws.as_i8(R_A),
                ws.as_i8(R_B),
                ws.as_i32_mut(R_C),
                m as i32,
                n as i32,
                k as i32,
                stream.0,
            );
            cuda_memcpy_d2h(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ws.ptr(R_C),
                out.len() * 4,
                stream.0,
            );
        }
        stream.sync();
        out
    }

    pub fn param_count(&self) -> usize {
        self.weight_ih_f32.len() + self.weight_hh_f32.len() + self.bias.len()
    }
}

pub struct BiLSTMInt8 {
    pub forward: LSTMInt8,
    pub backward: LSTMInt8,
    pub hidden_size: usize,
}

impl BiLSTMInt8 {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            forward: LSTMInt8::new(input_size, hidden_size),
            backward: LSTMInt8::new(input_size, hidden_size),
            hidden_size,
        }
    }

    /// `x_seq_int8` is [seq_len, batch, input_size] row-major.  Returns
    /// [batch, 2*hidden] — forward and backward states averaged over time.
    /// `batch` does not have to be a multiple of 16; the GEMM operands are
    /// padded internally and the padding rows are ignored on the way out.
    pub fn forward(
        &self,
        x_seq_int8: &[i8],
        batch_size: usize,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Vec<f32> {
        let inp = self.forward.input_size;
        let hs = self.hidden_size;
        let need = seq_len * batch_size * inp;
        assert!(
            x_seq_int8.len() >= need,
            "BiLSTM::forward: input is {} bytes, need {}",
            x_seq_int8.len(),
            need
        );

        let run = |lstm: &LSTMInt8| -> Vec<Vec<f32>> {
            let mut h = vec![0.0f32; batch_size * hs];
            let mut c = vec![0.0f32; batch_size * hs];
            let mut all = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = &x_seq_int8[t * batch_size * inp..(t + 1) * batch_size * inp];
                let (hn, cn) = lstm.forward_step(x_t, &h, &c, batch_size, stream);
                h = hn;
                c = cn;
                all.push(h.clone());
            }
            all
        };

        let fwd = run(&self.forward);
        let bwd = run(&self.backward);

        let inv_t = 1.0 / seq_len.max(1) as f32;
        let mut output = vec![0.0f32; batch_size * 2 * hs];
        output
            .par_chunks_mut(2 * hs)
            .enumerate()
            .for_each(|(b, row)| {
                for i in 0..hs {
                    let mut sf = 0.0f32;
                    let mut sb = 0.0f32;
                    for t in 0..seq_len {
                        sf += fwd[t][b * hs + i];
                        sb += bwd[t][b * hs + i];
                    }
                    row[i] = sf * inv_t;
                    row[hs + i] = sb * inv_t;
                }
            });
        output
    }

    pub fn param_count(&self) -> usize {
        self.forward.param_count() + self.backward.param_count()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
