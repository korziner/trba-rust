//! INT8 linear layer with a plain-FP32 backward pass.
//!
//! Forward runs on the INT8 tensor-core GEMM; the gradient is accumulated in
//! FP32 on the host.  These layers are small (512x512 and 512x192), so the
//! host-side backward costs microseconds next to the CNN.

#![allow(dead_code, unused_variables)]

use crate::backend::*;
use rayon::prelude::*;

pub struct LinearInt8 {
    pub weight_f32: Vec<f32>,
    pub weight_int8: Vec<i8>,
    pub grad_f32: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Clone for LinearInt8 {
    fn clone(&self) -> Self {
        Self {
            weight_f32: self.weight_f32.clone(),
            weight_int8: self.weight_int8.clone(),
            grad_f32: vec![0.0; self.grad_f32.len()],
            bias: self.bias.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}

impl LinearInt8 {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight_f32: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let weight_int8 = Self::quantize(&weight_f32);
        Self {
            weight_f32,
            weight_int8,
            grad_f32: vec![0.0; out_features * in_features],
            bias: vec![0.0; out_features],
            in_features,
            out_features,
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
        self.weight_int8 = Self::quantize(&self.weight_f32);
    }

    /// `inp` is [rows, in_features] int8; returns [rows, out_features] int32.
    pub fn forward(&self, inp: &[i8], stream: &CudaStream) -> Vec<i32> {
        assert!(
            !inp.is_empty() && inp.len() % self.in_features == 0,
            "LinearInt8::forward: input {} is not a multiple of in_features {}",
            inp.len(),
            self.in_features
        );
        let rows = inp.len() / self.in_features;
        let (mp, np, kp) = (
            align16(rows),
            align16(self.out_features),
            align16(self.in_features),
        );

        let (a_pad, _, _) = pad_to_multiple_i8(inp, rows, self.in_features, 16);
        // weight_int8 is [out_features, in_features] row-major, which is
        // exactly the Bt layout the NT GEMM wants.
        let (b_pad, _, _) =
            pad_to_multiple_i8(&self.weight_int8, self.out_features, self.in_features, 16);
        debug_assert_eq!(a_pad.len(), mp * kp);
        debug_assert_eq!(b_pad.len(), np * kp);

        const R_A: usize = 0;
        const R_B: usize = 1;
        const R_C: usize = 2;
        let ws = GpuWorkspace::new(&[a_pad.len(), b_pad.len(), rows * self.out_features * 4]);
        ws.upload_i8(R_A, &a_pad, stream);
        ws.upload_i8(R_B, &b_pad, stream);

        let mut out = vec![0i32; rows * self.out_features];
        unsafe {
            launch_gemm_int8(
                ws.as_i8(R_A),
                ws.as_i8(R_B),
                ws.as_i32_mut(R_C),
                rows as i32,
                self.out_features as i32,
                self.in_features as i32,
                stream.0,
            );
        }
        {
            let out_bytes = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4)
            };
            ws.download(R_C, out_bytes, stream);
        }
        stream.sync();
        out
    }

    /// FP32 backward + SGD step with global-norm clipping.
    ///
    /// `inp` is the layer input in the *dequantised* domain, `grad_out` the
    /// gradient of the loss w.r.t. the layer output, both [rows, features].
    pub fn backward(&mut self, inp: &[f32], grad_out: &[f32], lr: f32, mom: f32) -> f32 {
        let rows = if self.in_features > 0 {
            inp.len() / self.in_features
        } else {
            0
        };
        if rows == 0 || grad_out.len() < rows * self.out_features {
            return 0.0;
        }
        let inv_rows = 1.0 / rows as f32;

        // grad_W[o, i] = sum_r grad_out[r, o] * inp[r, i]
        self.grad_f32
            .par_chunks_mut(self.in_features)
            .enumerate()
            .for_each(|(o, grow)| {
                for i in 0..self.in_features {
                    let mut acc = 0.0f32;
                    for r in 0..rows {
                        acc += grad_out[r * self.out_features + o] * inp[r * self.in_features + i];
                    }
                    grow[i] = acc * inv_rows;
                }
            });

        let grad_norm: f32 = self
            .grad_f32
            .par_iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let max_grad = 10.0f32;
        let clip = if grad_norm > max_grad { max_grad / grad_norm } else { 1.0 };
        let step = lr * clip;

        self.weight_f32
            .par_iter_mut()
            .zip(self.grad_f32.par_iter())
            .for_each(|(w, g)| {
                let updated = *w - step * g;
                *w = if updated.is_finite() { updated } else { 0.0 };
            });

        for o in 0..self.out_features {
            let mut gb = 0.0f32;
            for r in 0..rows {
                gb += grad_out[r * self.out_features + o];
            }
            let updated = self.bias[o] - step * gb * inv_rows;
            self.bias[o] = if updated.is_finite() { updated } else { 0.0 };
        }

        self.requantize();
        clip
    }

    pub fn param_count(&self) -> usize {
        self.weight_f32.len() + self.bias.len()
    }
}
