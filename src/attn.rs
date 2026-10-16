//! Attention decoder, mirroring the reference `AttentionCell` /
//! `AttentionDecoder` (teacher-forced training forward + greedy decode).
//!
//! ```text
//! proj_H = i2h(batch_H)                   [B, T, Hs]   (no bias)
//! e      = score(tanh(proj_H + h2h(h)))   [B, T, 1]    (score: no bias)
//! alpha  = softmax(e, dim=1)
//! context= sum_t alpha_t * H_t            [B, C]
//! (h,c)  = LSTMCell([context, onehot], (h,c))
//! logits = generator(h)
//! ```

use crate::qnet::LinearF;

pub struct AttentionDecoder {
    pub i2h: LinearF,
    pub h2h: LinearF,
    pub score: LinearF,
    pub w_ih: Vec<f32>, // [4Hs, C+V]
    pub b_ih: Vec<f32>,
    pub w_hh: Vec<f32>, // [4Hs, Hs]
    pub b_hh: Vec<f32>,
    pub g_w_ih: Vec<f32>,
    pub g_b_ih: Vec<f32>,
    pub g_w_hh: Vec<f32>,
    pub g_b_hh: Vec<f32>,
    pub generator: LinearF,
    pub hidden: usize,
    pub classes: usize,
    pub ctx: usize,
    pub sos: usize,
    pub eos: usize,
    #[allow(dead_code)]
    pub pad: usize,
}

pub struct AttnTrace {
    pub batch_h: Vec<f32>,
    pub proj_h: Vec<f32>,  // [B, T, Hs]
    pub tanh_e: Vec<f32>,  // [steps, B, T, Hs]
    pub alpha: Vec<f32>,   // [B, T] per step? -> [steps, B, T]
    pub onehots: Vec<f32>, // [steps, B, V]
    pub cell_in: Vec<f32>, // [steps, B, C+V]
    pub gates: Vec<f32>,   // [steps, B, 4Hs] post-activation
    pub hs: Vec<f32>,      // [steps, B, Hs]
    pub cs: Vec<f32>,      // [steps, B, Hs]
    pub b: usize,
    pub t: usize,
    pub steps: usize,
}

impl AttentionDecoder {
    pub fn new(ctx: usize, hidden: usize, classes: usize, sos: usize, eos: usize, pad: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let lim_ih = (6.0 / (ctx + classes + 4 * hidden) as f32).sqrt();
        let lim_hh = (6.0 / (hidden + 4 * hidden) as f32).sqrt();
        Self {
            i2h: LinearF::new(ctx, hidden, false),
            h2h: LinearF::new(hidden, hidden, true),
            score: LinearF::new(hidden, 1, false),
            w_ih: (0..4 * hidden * (ctx + classes)).map(|_| rng.gen_range(-lim_ih..lim_ih)).collect(),
            b_ih: vec![0.0; 4 * hidden],
            w_hh: (0..4 * hidden * hidden).map(|_| rng.gen_range(-lim_hh..lim_hh)).collect(),
            b_hh: vec![0.0; 4 * hidden],
            g_w_ih: vec![0.0; 4 * hidden * (ctx + classes)],
            g_b_ih: vec![0.0; 4 * hidden],
            g_w_hh: vec![0.0; 4 * hidden * hidden],
            g_b_hh: vec![0.0; 4 * hidden],
            generator: LinearF::new(hidden, classes, true),
            hidden,
            classes,
            ctx,
            sos,
            eos,
            pad,
        }
    }

    pub fn zero_grad(&mut self) {
        self.i2h.zero_grad();
        self.h2h.zero_grad();
        self.score.zero_grad();
        self.generator.zero_grad();
        for v in [&mut self.g_w_ih, &mut self.g_b_ih, &mut self.g_w_hh, &mut self.g_b_hh] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn cell_gates(&self, ci: &[f32], h: &[f32], b: usize) -> Vec<f32> {
        let (hs, cv) = (self.hidden, self.ctx + self.classes);
        let mut g = vec![0.0f32; b * 4 * hs];
        for bi in 0..b {
            for o in 0..4 * hs {
                let wi = &self.w_ih[o * cv..(o + 1) * cv];
                let mut s = self.b_ih[o] + self.b_hh[o];
                for k in 0..cv {
                    s += ci[bi * cv + k] * wi[k];
                }
                let wh = &self.w_hh[o * hs..(o + 1) * hs];
                for k in 0..hs {
                    s += h[bi * hs + k] * wh[k];
                }
                g[bi * 4 * hs + o] = s;
            }
        }
        g
    }

    /// Teacher-forced forward; `text_in[bi][st]` = token fed at step st.
    /// Returns logits [B, steps, V].
    pub fn forward_train(&self, batch_h: &[f32], text_in: &[Vec<usize>], b: usize, t: usize) -> (Vec<f32>, AttnTrace) {
        let (hs, c, v) = (self.hidden, self.ctx, self.classes);
        let steps = text_in[0].len();
        let proj_h = self.i2h.forward(batch_h, b * t);
        let mut h = vec![0.0f32; b * hs];
        let mut cc = vec![0.0f32; b * hs];
        let mut gates_all = vec![0.0f32; steps * b * 4 * hs];
        let mut hs_all = vec![0.0f32; steps * b * hs];
        let mut cs_all = vec![0.0f32; steps * b * hs];
        let mut alpha_all = vec![0.0f32; steps * b * t];
        let mut tanh_all = vec![0.0f32; steps * b * t * hs];
        let mut onehots = vec![0.0f32; steps * b * v];
        let mut cell_in = vec![0.0f32; steps * b * (c + v)];
        let mut out_hid = vec![0.0f32; steps * b * hs];

        for st in 0..steps {
            let proj_hh = self.h2h.forward(&h, b);
            let mut e = vec![0.0f32; b * t];
            let tn = &mut tanh_all[st * b * t * hs..(st + 1) * b * t * hs];
            for bi in 0..b {
                for tt in 0..t {
                    let r = bi * t + tt;
                    for k in 0..hs {
                        tn[r * hs + k] = (proj_h[r * hs + k] + proj_hh[bi * hs + k]).tanh();
                    }
                    let mut s = 0.0;
                    for k in 0..hs {
                        s += tn[r * hs + k] * self.score.w[k];
                    }
                    e[r] = s;
                }
            }
            let alpha = &mut alpha_all[st * b * t..(st + 1) * b * t];
            for bi in 0..b {
                let row = &e[bi * t..(bi + 1) * t];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = row.iter().map(|&x| (x - mx).exp()).collect();
                let s: f32 = ex.iter().sum();
                alpha[bi * t..(bi + 1) * t].copy_from_slice(&ex.iter().map(|&x| x / s).collect::<Vec<f32>>());
            }
            let mut context = vec![0.0f32; b * c];
            for bi in 0..b {
                for tt in 0..t {
                    let a = alpha[bi * t + tt];
                    if a == 0.0 {
                        continue;
                    }
                    for k in 0..c {
                        context[bi * c + k] += a * batch_h[(bi * t + tt) * c + k];
                    }
                }
            }
            for bi in 0..b {
                onehots[st * b * v + bi * v + text_in[bi][st]] = 1.0;
                cell_in[st * b * (c + v)..st * b * (c + v) + c].copy_from_slice(&context[bi * c..(bi + 1) * c]);
                let co = st * b * (c + v) + bi * (c + v) + c;
                cell_in[co..co + v].copy_from_slice(&onehots[st * b * v + bi * v..st * b * v + (bi + 1) * v]);
            }
            let ci = &cell_in[st * b * (c + v)..(st + 1) * b * (c + v)];
            let pre = self.cell_gates(ci, &h, b);
            let gr = &mut gates_all[st * b * 4 * hs..(st + 1) * b * 4 * hs];
            for bi in 0..b {
                for i in 0..hs {
                    let ig = 1.0 / (1.0 + (-pre[bi * 4 * hs + i]).exp());
                    let fg = 1.0 / (1.0 + (-pre[bi * 4 * hs + hs + i]).exp());
                    let cg = pre[bi * 4 * hs + 2 * hs + i].tanh();
                    let og = 1.0 / (1.0 + (-pre[bi * 4 * hs + 3 * hs + i]).exp());
                    let cnew = fg * cc[bi * hs + i] + ig * cg;
                    gr[bi * 4 * hs + i] = ig;
                    gr[bi * 4 * hs + hs + i] = fg;
                    gr[bi * 4 * hs + 2 * hs + i] = cg;
                    gr[bi * 4 * hs + 3 * hs + i] = og;
                    cc[bi * hs + i] = cnew;
                    h[bi * hs + i] = og * cnew.tanh();
                }
            }
            hs_all[st * b * hs..(st + 1) * b * hs].copy_from_slice(&h);
            cs_all[st * b * hs..(st + 1) * b * hs].copy_from_slice(&cc);
            out_hid[st * b * hs..(st + 1) * b * hs].copy_from_slice(&h);
        }
        let logits_sb = self.generator.forward(&out_hid, steps * b);
        let mut l2 = vec![0.0f32; b * steps * v];
        for st in 0..steps {
            for bi in 0..b {
                l2[(bi * steps + st) * v..(bi * steps + st + 1) * v]
                    .copy_from_slice(&logits_sb[(st * b + bi) * v..(st * b + bi + 1) * v]);
            }
        }
        (
            l2,
            AttnTrace {
                batch_h: batch_h.to_vec(),
                proj_h,
                tanh_e: tanh_all,
                alpha: alpha_all,
                onehots,
                cell_in,
                gates: gates_all,
                hs: hs_all,
                cs: cs_all,
                b,
                t,
                steps,
            },
        )
    }

    /// `d_logits` [B, steps, V] -> d(batch_h) [B, T, C]
    pub fn backward(&mut self, d_logits: &[f32], tr: &AttnTrace) -> Vec<f32> {
        let (b, t, steps) = (tr.b, tr.t, tr.steps);
        let (hs, c, v) = (self.hidden, self.ctx, self.classes);
        let cv = c + v;
        let mut d_out_hid = vec![0.0f32; steps * b * hs];
        for st in 0..steps {
            for bi in 0..b {
                let dyr = &d_logits[(bi * steps + st) * v..(bi * steps + st + 1) * v];
                let xrow = tr.hs[st * b * hs + bi * hs..st * b * hs + (bi + 1) * hs].to_vec();
                let dx = self.generator.backward_row(&xrow, dyr);
                d_out_hid[st * b * hs + bi * hs..st * b * hs + (bi + 1) * hs].copy_from_slice(&dx);
            }
        }
        let mut dh_next = vec![0.0f32; b * hs];
        let mut dc_next = vec![0.0f32; b * hs];
        let mut d_proj_h = vec![0.0f32; b * t * hs];
        let mut d_batch_h = vec![0.0f32; b * t * c];

        for st in (0..steps).rev() {
            let gates = tr.gates[st * b * 4 * hs..(st + 1) * b * 4 * hs].to_vec();
            let ci = tr.cell_in[st * b * cv..(st + 1) * b * cv].to_vec();
            let c_prev: Vec<f32> = if st == 0 { vec![0.0; b * hs] } else { tr.cs[(st - 1) * b * hs..st * b * hs].to_vec() };
            let h_prev: Vec<f32> = if st == 0 { vec![0.0; b * hs] } else { tr.hs[(st - 1) * b * hs..st * b * hs].to_vec() };
            let ccur = tr.cs[st * b * hs..(st + 1) * b * hs].to_vec();
            let mut dgates = vec![0.0f32; b * 4 * hs];
            for bi in 0..b {
                for i in 0..hs {
                    let ig = gates[bi * 4 * hs + i];
                    let fg = gates[bi * 4 * hs + hs + i];
                    let cg = gates[bi * 4 * hs + 2 * hs + i];
                    let og = gates[bi * 4 * hs + 3 * hs + i];
                    let tc = ccur[bi * hs + i].tanh();
                    let dh = d_out_hid[st * b * hs + bi * hs + i] + dh_next[bi * hs + i];
                    let dc = dh * og * (1.0 - tc * tc) + dc_next[bi * hs + i] * fg;
                    dgates[bi * 4 * hs + i] = dc * cg * ig * (1.0 - ig);
                    dgates[bi * 4 * hs + hs + i] = dc * c_prev[bi * hs + i] * fg * (1.0 - fg);
                    dgates[bi * 4 * hs + 2 * hs + i] = dc * ig * (1.0 - cg * cg);
                    dgates[bi * 4 * hs + 3 * hs + i] = dh * tc * og * (1.0 - og);
                    dc_next[bi * hs + i] = dc;
                }
            }
            // weight grads + dh_prev (lstm part) + d cell input
            let mut dh_prev = vec![0.0f32; b * hs];
            let mut d_cellin = vec![0.0f32; b * cv];
            for bi in 0..b {
                let dg = &dgates[bi * 4 * hs..(bi + 1) * 4 * hs];
                for o in 0..4 * hs {
                    let dv = dg[o];
                    if dv == 0.0 {
                        continue;
                    }
                    self.g_b_ih[o] += dv;
                    self.g_b_hh[o] += dv;
                    for k in 0..cv {
                        self.g_w_ih[o * cv + k] += dv * ci[bi * cv + k];
                        d_cellin[bi * cv + k] += dv * self.w_ih[o * cv + k];
                    }
                    for k in 0..hs {
                        self.g_w_hh[o * hs + k] += dv * h_prev[bi * hs + k];
                        dh_prev[bi * hs + k] += dv * self.w_hh[o * hs + k];
                    }
                }
            }
            // context gradients -> alpha, batch_h
            for bi in 0..b {
                let dctx = &d_cellin[bi * c..bi * c + c];
                let alpha = &tr.alpha[st * b * t + bi * t..st * b * t + (bi + 1) * t];
                let mut d_alpha = vec![0.0f32; t];
                for tt in 0..t {
                    let mut s = 0.0;
                    for k in 0..c {
                        s += dctx[k] * tr.batch_h[(bi * t + tt) * c + k];
                    }
                    d_alpha[tt] = s;
                    for k in 0..c {
                        d_batch_h[(bi * t + tt) * c + k] += alpha[tt] * dctx[k];
                    }
                }
                let dot: f32 = (0..t).map(|tt| alpha[tt] * d_alpha[tt]).sum();
                for tt in 0..t {
                    let de = alpha[tt] * (d_alpha[tt] - dot);
                    if de == 0.0 {
                        continue;
                    }
                    let trow = &tr.tanh_e[st * b * t * hs + (bi * t + tt) * hs..st * b * t * hs + (bi * t + tt + 1) * hs];
                    for k in 0..hs {
                        self.score.g_w[k] += de * trow[k];
                        let dz = de * self.score.w[k] * (1.0 - trow[k] * trow[k]);
                        d_proj_h[(bi * t + tt) * hs + k] += dz;
                    }
                }
                // h2h backward for this step (x = h_prev)
                let dprojhh: Vec<f32> = {
                    // d_proj_h contributions from this step go through tanh to BOTH
                    // proj_H (accumulated in d_proj_h) and proj_hh; recompute per bi:
                    let mut v = vec![0.0f32; hs];
                    for tt in 0..t {
                        let _ = tt;
                    }
                    v
                };
                let _ = dprojhh;
            }
            // h2h grad: d proj_hh[bi] = sum over tt of de-path dz (same values added
            // to proj_H and proj_hh since tanh input is their sum)
            for bi in 0..b {
                let mut dprojhh = vec![0.0f32; hs];
                for tt in 0..t {
                    let alpha = &tr.alpha[st * b * t + bi * t..st * b * t + (bi + 1) * t];
                    let mut d_alpha = vec![0.0f32; t];
                    let dctx = &d_cellin[bi * c..bi * c + c];
                    for tt2 in 0..t {
                        let mut s = 0.0;
                        for k in 0..c {
                            s += dctx[k] * tr.batch_h[(bi * t + tt2) * c + k];
                        }
                        d_alpha[tt2] = s;
                    }
                    let dot: f32 = (0..t).map(|x| alpha[x] * d_alpha[x]).sum();
                    let de = alpha[tt] * (d_alpha[tt] - dot);
                    if de == 0.0 {
                        continue;
                    }
                    let trow = &tr.tanh_e[st * b * t * hs + (bi * t + tt) * hs..st * b * t * hs + (bi * t + tt + 1) * hs];
                    for k in 0..hs {
                        dprojhh[k] += de * self.score.w[k] * (1.0 - trow[k] * trow[k]);
                    }
                }
                let dx = self.h2h.backward_row(&h_prev[bi * hs..(bi + 1) * hs], &dprojhh);
                for k in 0..hs {
                    dh_prev[bi * hs + k] += dx[k];
                }
            }
            dh_next = dh_prev;
        }
        // i2h backward over [B*T] rows
        for r in 0..b * t {
            let xrow = tr.batch_h[r * c..(r + 1) * c].to_vec();
            let dyr = d_proj_h[r * hs..(r + 1) * hs].to_vec();
            let dx = self.i2h.backward_row(&xrow, &dyr);
            for k in 0..c {
                d_batch_h[r * c + k] += dx[k];
            }
        }
        d_batch_h
    }

    /// Greedy decode; returns token ids per sample (EOS stripped).
    pub fn greedy_decode(&self, batch_h: &[f32], b: usize, t: usize, max_len: usize) -> Vec<Vec<usize>> {
        let (hs, c, v) = (self.hidden, self.ctx, self.classes);
        let proj_h = self.i2h.forward(batch_h, b * t);
        let mut h = vec![0.0f32; b * hs];
        let mut cc = vec![0.0f32; b * hs];
        let mut targets = vec![self.sos; b];
        let mut out = vec![Vec::new(); b];
        for _ in 0..max_len + 1 {
            let proj_hh = self.h2h.forward(&h, b);
            let mut e = vec![0.0f32; b * t];
            for bi in 0..b {
                for tt in 0..t {
                    let r = bi * t + tt;
                    let mut s = 0.0;
                    for k in 0..hs {
                        let z = (proj_h[r * hs + k] + proj_hh[bi * hs + k]).tanh();
                        s += z * self.score.w[k];
                    }
                    e[r] = s;
                }
            }
            let mut context = vec![0.0f32; b * c];
            for bi in 0..b {
                let row = &e[bi * t..(bi + 1) * t];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = row.iter().map(|&x| (x - mx).exp()).collect();
                let s: f32 = ex.iter().sum();
                for tt in 0..t {
                    let a = ex[tt] / s;
                    if a == 0.0 {
                        continue;
                    }
                    for k in 0..c {
                        context[bi * c + k] += a * batch_h[(bi * t + tt) * c + k];
                    }
                }
            }
            let mut ci = vec![0.0f32; b * (c + v)];
            for bi in 0..b {
                ci[bi * (c + v)..bi * (c + v) + c].copy_from_slice(&context[bi * c..(bi + 1) * c]);
                ci[bi * (c + v) + c + targets[bi]] = 1.0;
            }
            let pre = self.cell_gates(&ci, &h, b);
            for bi in 0..b {
                for i in 0..hs {
                    let ig = 1.0 / (1.0 + (-pre[bi * 4 * hs + i]).exp());
                    let fg = 1.0 / (1.0 + (-pre[bi * 4 * hs + hs + i]).exp());
                    let cg = pre[bi * 4 * hs + 2 * hs + i].tanh();
                    let og = 1.0 / (1.0 + (-pre[bi * 4 * hs + 3 * hs + i]).exp());
                    let cnew = fg * cc[bi * hs + i] + ig * cg;
                    cc[bi * hs + i] = cnew;
                    h[bi * hs + i] = og * cnew.tanh();
                }
            }
            let logits = self.generator.forward(&h, b);
            let mut all_eos = true;
            for bi in 0..b {
                let row = &logits[bi * v..(bi + 1) * v];
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (k, &x) in row.iter().enumerate() {
                    if x > bv {
                        bv = x;
                        best = k;
                    }
                }
                if best == self.eos {
                    all_eos &= true;
                } else {
                    all_eos = false;
                    out[bi].push(best);
                }
                targets[bi] = best;
            }
            if all_eos {
                break;
            }
        }
        out
    }
}
