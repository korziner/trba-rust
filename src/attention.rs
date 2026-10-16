//! Attention decoder: an attention-weighted context over the BiLSTM outputs is
//! fed to an LSTM cell whose hidden state is projected to class logits by
//! `generator`.
//!
//! The training loop currently drives `generator` directly (single-step,
//! character-level classification); the rest of the decoder is here so a real
//! sequence decode can be switched on without touching the backbone.

#![allow(dead_code, unused_variables)]

use crate::backend::CudaStream;
use crate::bilstm::LSTMInt8;
use crate::linear::LinearInt8;
use rayon::prelude::*;

pub struct AttentionDecoder {
    pub embed_size: usize,
    pub hidden_size: usize,
    pub num_classes: usize,
    /// Splits the input projection (i2h) and the recurrent projection (h2h),
    /// matching the reference PyTorch `LSTMCell`.
    pub cell: CellWeights,
    /// Encoder-state -> scoring-function input.
    pub attention: LinearInt8,
    /// Hidden state -> class logits.
    pub generator: LinearInt8,
}

pub struct CellWeights {
    pub i2h: LSTMInt8,
    pub h2h: LSTMInt8,
}

impl AttentionDecoder {
    pub fn new(embed_size: usize, hidden_size: usize, num_classes: usize) -> Self {
        Self {
            embed_size,
            hidden_size,
            num_classes,
            cell: CellWeights {
                i2h: LSTMInt8::new(embed_size, hidden_size),
                h2h: LSTMInt8::new(hidden_size, hidden_size),
            },
            attention: LinearInt8::new(hidden_size, hidden_size),
            generator: LinearInt8::new(hidden_size, num_classes),
        }
    }

    pub fn requantize(&mut self) {
        self.cell.i2h.requantize();
        self.cell.h2h.requantize();
        self.attention.requantize();
        self.generator.requantize();
    }

    pub fn param_count(&self) -> usize {
        self.cell.i2h.param_count()
            + self.cell.h2h.param_count()
            + self.attention.param_count()
            + self.generator.param_count()
    }

    /// Softmax attention over `encoder_outputs` ([batch, src_len, hidden])
    /// given the previous hidden state ([batch, hidden]).
    /// Returns the context vector [batch, hidden].
    pub fn attend(&self, encoder_outputs: &[f32], prev_hidden: &[f32], batch: usize, src_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let mut scores = vec![0.0f32; batch * src_len];
        for b in 0..batch {
            let hb = &prev_hidden[b * h..(b + 1) * h];
            // dot-product scoring
            let mut mx = f32::NEG_INFINITY;
            for t in 0..src_len {
                let e = &encoder_outputs[(b * src_len + t) * h..(b * src_len + t + 1) * h];
                let s: f32 = e.iter().zip(hb.iter()).map(|(a, b2)| a * b2).sum();
                scores[b * src_len + t] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for t in 0..src_len {
                let v = (scores[b * src_len + t] - mx).exp();
                scores[b * src_len + t] = v;
                sum += v;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for t in 0..src_len {
                scores[b * src_len + t] *= inv;
            }
        }

        let mut context = vec![0.0f32; batch * h];
        context
            .par_chunks_mut(h)
            .enumerate()
            .for_each(|(b, row)| {
                for t in 0..src_len {
                    let w = scores[b * src_len + t];
                    if w == 0.0 {
                        continue;
                    }
                    let e = &encoder_outputs[(b * src_len + t) * h..(b * src_len + t + 1) * h];
                    for (c, v) in row.iter_mut().zip(e.iter()) {
                        *c += w * v;
                    }
                }
            });
        context
    }

    /// One decoder step: attend, run the LSTM cell, project to logits.
    /// Returns (logits [batch, num_classes], new_hidden, new_cell).
    pub fn forward_step(
        &self,
        encoder_outputs: &[f32],
        prev_hidden: &[f32],
        prev_cell: &[f32],
        batch: usize,
        src_len: usize,
        stream: &CudaStream,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let context = self.attend(encoder_outputs, prev_hidden, batch, src_len);

        let scale = crate::conv2d::ACC_DIVISOR;
        let ctx_i8: Vec<i8> = context
            .iter()
            .map(|&x| (x * scale).clamp(-127.0, 127.0).round() as i8)
            .collect();
        let hid_i8: Vec<i8> = prev_hidden
            .iter()
            .map(|&x| (x * scale).clamp(-127.0, 127.0).round() as i8)
            .collect();

        let h = self.hidden_size;
        let gates_i = self.cell.i2h.i2h_forward(&ctx_i8, batch, stream);
        let gates_h = self.cell.h2h.h2h_forward(&hid_i8, batch, stream);

        let mut new_hidden = vec![0.0f32; batch * h];
        let mut new_cell = vec![0.0f32; batch * h];
        for b in 0..batch {
            for i in 0..h {
                let row_i = b * 4 * h;
                let row_h = b * 4 * h;
                let pre = |k: usize| {
                    (gates_i[row_i + k * h + i] as f32 + gates_h[row_h + k * h + i] as f32)
                        / (scale * scale)
                        + self.cell.i2h.bias[k * h + i]
                };
                let ig = 1.0 / (1.0 + (-pre(0)).exp());
                let fg = 1.0 / (1.0 + (-pre(1)).exp());
                let cg = pre(2).tanh();
                let og = 1.0 / (1.0 + (-pre(3)).exp());
                let c = fg * prev_cell[b * h + i] + ig * cg;
                new_cell[b * h + i] = c;
                new_hidden[b * h + i] = og * c.tanh();
            }
        }

        let hid_out_i8: Vec<i8> = new_hidden
            .iter()
            .map(|&x| (x * scale).clamp(-127.0, 127.0).round() as i8)
            .collect();
        let logits_i32 = self.generator.forward(&hid_out_i8, stream);
        let logits: Vec<f32> = logits_i32
            .iter()
            .map(|&x| x as f32 / crate::conv2d::ACC_DIVISOR)
            .collect();
        (logits, new_hidden, new_cell)
    }
}

impl CellWeights {
    pub fn param_count(&self) -> usize {
        self.i2h.param_count() + self.h2h.param_count()
    }
}
