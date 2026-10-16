//! Adam optimiser with PyTorch semantics (`weight_decay` = L2 added to the
//! gradient) plus global gradient-norm clipping, mirroring
//! `torch.nn.utils.clip_grad_norm_(params, 5.0)` from the reference trainer.

use std::collections::HashMap;

/// A (param, grad) pair handed to the optimiser by raw pointer; the trainer
/// builds the list once per step from disjoint struct fields.
pub struct ParamRef {
    pub name: String,
    pub p: *mut f32,
    pub g: *mut f32,
    pub len: usize,
}

// The trainer is single-threaded between optimiser steps.
unsafe impl Send for ParamRef {}

pub struct Adam {
    pub t: usize,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl Adam {
    pub fn new() -> Self {
        Self { t: 0, beta1: 0.9, beta2: 0.999, eps: 1e-8, m: HashMap::new(), v: HashMap::new() }
    }

    pub fn global_norm(params: &[ParamRef]) -> f64 {
        let mut s = 0.0f64;
        for pr in params {
            let g = unsafe { std::slice::from_raw_parts(pr.g, pr.len) };
            for &x in g {
                s += x as f64 * x as f64;
            }
        }
        s.sqrt()
    }

    /// One Adam step over all params.  `clip` is the already-applied scale
    /// factor (trainer clips before calling); `wd` is L2 weight decay.
    pub fn step(&mut self, params: &[ParamRef], lr: f32, wd: f32) {
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32) as f64;
        let bc2 = 1.0 - self.beta2.powi(self.t as i32) as f64;
        for pr in params {
            let p = unsafe { std::slice::from_raw_parts_mut(pr.p, pr.len) };
            let g = unsafe { std::slice::from_raw_parts(pr.g, pr.len) };
            let m = self.m.entry(pr.name.clone()).or_insert_with(|| vec![0.0; pr.len]);
            let v = self.v.entry(pr.name.clone()).or_insert_with(|| vec![0.0; pr.len]);
            for i in 0..pr.len {
                let gi = g[i] as f64 + wd as f64 * p[i] as f64;
                m[i] = (self.beta1 as f64 * m[i] as f64 + (1.0 - self.beta1 as f64) * gi) as f32;
                v[i] = (self.beta2 as f64 * v[i] as f64 + (1.0 - self.beta2 as f64) * gi * gi) as f32;
                let mh = m[i] as f64 / bc1;
                let vh = v[i] as f64 / bc2;
                p[i] -= (lr as f64 * mh / (vh.sqrt() + self.eps as f64)) as f32;
            }
        }
    }
}
