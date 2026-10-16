//! Host fp32 Bidirectional LSTM stack, mirroring the reference encoder
//! `nn.Sequential(BidirectionalLSTM(512,256,256), BidirectionalLSTM(256,256,256))`
//! where `BidirectionalLSTM = nn.LSTM(bidirectional=True) + nn.Linear(2H, D)`.
//!
//! PyTorch gate order is (input, forget, cell, output).

use rayon::prelude::*;

#[derive(Clone)]
pub struct LstmDir {
    pub w_ih: Vec<f32>, // [4H, D]
    pub b_ih: Vec<f32>,
    pub w_hh: Vec<f32>, // [4H, H]
    pub b_hh: Vec<f32>,
    pub g_w_ih: Vec<f32>,
    pub g_b_ih: Vec<f32>,
    pub g_w_hh: Vec<f32>,
    pub g_b_hh: Vec<f32>,
    pub d: usize,
    pub h: usize,
}

impl LstmDir {
    pub fn new(d: usize, h: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let lim_ih = (6.0 / (d + 4 * h) as f32).sqrt();
        let lim_hh = (6.0 / (h + 4 * h) as f32).sqrt();
        Self {
            w_ih: (0..4 * h * d).map(|_| rng.gen_range(-lim_ih..lim_ih)).collect(),
            b_ih: vec![0.0; 4 * h],
            w_hh: (0..4 * h * h).map(|_| rng.gen_range(-lim_hh..lim_hh)).collect(),
            b_hh: vec![0.0; 4 * h],
            g_w_ih: vec![0.0; 4 * h * d],
            g_b_ih: vec![0.0; 4 * h],
            g_w_hh: vec![0.0; 4 * h * h],
            g_b_hh: vec![0.0; 4 * h],
            d,
            h,
        }
    }
    pub fn zero_grad(&mut self) {
        self.g_w_ih.iter_mut().for_each(|x| *x = 0.0);
        self.g_b_ih.iter_mut().for_each(|x| *x = 0.0);
        self.g_w_hh.iter_mut().for_each(|x| *x = 0.0);
        self.g_b_hh.iter_mut().for_each(|x| *x = 0.0);
    }
}

#[derive(Clone)]
pub struct BiLstmLayer {
    pub fwd: LstmDir,
    pub bwd: LstmDir,
    pub proj_w: Vec<f32>, // [out, 2H]
    pub proj_b: Vec<f32>,
    pub g_proj_w: Vec<f32>,
    pub g_proj_b: Vec<f32>,
    pub h: usize,
    pub out: usize,
}

struct DirTrace {
    gates: Vec<f32>, // [T, B, 4H]
    c: Vec<f32>,     // [T, B, H]
    h: Vec<f32>,     // [T, B, H]
}

pub struct BiLstmTrace {
    x: Vec<f32>,           // [B, T, D]
    f_tr: DirTrace,
    b_tr: DirTrace,
    cat: Vec<f32>,         // [B, T, 2H]
    b: usize,
    t: usize,
}

impl BiLstmLayer {
    pub fn new(d: usize, h: usize, out: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let lim = (6.0 / (2 * h + out) as f32).sqrt();
        Self {
            fwd: LstmDir::new(d, h),
            bwd: LstmDir::new(d, h),
            proj_w: (0..out * 2 * h).map(|_| rng.gen_range(-lim..lim)).collect(),
            proj_b: vec![0.0; out],
            g_proj_w: vec![0.0; out * 2 * h],
            g_proj_b: vec![0.0; out],
            h,
            out,
        }
    }

    pub fn zero_grad(&mut self) {
        self.fwd.zero_grad();
        self.bwd.zero_grad();
        self.g_proj_w.iter_mut().for_each(|x| *x = 0.0);
        self.g_proj_b.iter_mut().for_each(|x| *x = 0.0);
    }

    fn run_dir(l: &LstmDir, x: &[f32], b: usize, t: usize, rev: bool) -> (Vec<f32>, DirTrace) {
        let h = l.h;
        let d = l.d;
        let mut gates = vec![0.0f32; t * b * 4 * h];
        let mut cs = vec![0.0f32; t * b * h];
        let mut hs = vec![0.0f32; t * b * h];
        let mut c_prev = vec![0.0f32; b * h];
        let mut h_prev = vec![0.0f32; b * h];
        for step in 0..t {
            let tt = if rev { t - 1 - step } else { step };
            let xr = &x[tt * b * d..(tt + 1) * b * d];
            let gr = &mut gates[step * b * 4 * h..(step + 1) * b * 4 * h];
            // gates = x @ W_ih^T + b_ih + h_prev @ W_hh^T + b_hh
            gr.par_chunks_mut(4 * h).enumerate().for_each(|(bi, row)| {
                for o in 0..4 * h {
                    let wi = &l.w_ih[o * d..(o + 1) * d];
                    let mut s = l.b_ih[o] + l.b_hh[o];
                    for k in 0..d {
                        s += xr[bi * d + k] * wi[k];
                    }
                    let wh = &l.w_hh[o * h..(o + 1) * h];
                    for k in 0..h {
                        s += h_prev[bi * h + k] * wh[k];
                    }
                    row[o] = s;
                }
            });
            for bi in 0..b {
                for i in 0..h {
                    let gi = gr[bi * 4 * h + i];
                    let gf = gr[bi * 4 * h + h + i];
                    let gg = gr[bi * 4 * h + 2 * h + i];
                    let go = gr[bi * 4 * h + 3 * h + i];
                    let ig = 1.0 / (1.0 + (-gi).exp());
                    let fg = 1.0 / (1.0 + (-gf).exp());
                    let cg = gg.tanh();
                    let og = 1.0 / (1.0 + (-go).exp());
                    let c = fg * c_prev[bi * h + i] + ig * cg;
                    let hh = og * c.tanh();
                    gr[bi * 4 * h + i] = ig;
                    gr[bi * 4 * h + h + i] = fg;
                    gr[bi * 4 * h + 2 * h + i] = cg;
                    gr[bi * 4 * h + 3 * h + i] = og;
                    cs[step * b * h + bi * h + i] = c;
                    hs[step * b * h + bi * h + i] = hh;
                }
            }
            c_prev.copy_from_slice(&cs[step * b * h..(step + 1) * b * h]);
            h_prev.copy_from_slice(&hs[step * b * h..(step + 1) * b * h]);
        }
        (hs.clone(), DirTrace { gates, c: cs, h: hs })
    }

    /// `x` [B, T, D] -> `y` [B, T, out]
    pub fn forward(&self, x: &[f32], b: usize, t: usize) -> (Vec<f32>, BiLstmTrace) {
        let h = self.h;
        let (_hf, f_tr) = Self::run_dir(&self.fwd, x, b, t, false);
        let (_hb, b_tr) = Self::run_dir(&self.bwd, x, b, t, true);
        // hf/hb are [T, B, H]; cat -> [B, T, 2H]
        let mut cat = vec![0.0f32; b * t * 2 * h];
        for tt in 0..t {
            for bi in 0..b {
                cat[(bi * t + tt) * 2 * h..(bi * t + tt) * 2 * h + h]
                    .copy_from_slice(&f_tr.h[tt * b * h + bi * h..tt * b * h + (bi + 1) * h]);
                cat[(bi * t + tt) * 2 * h + h..(bi * t + tt + 1) * 2 * h]
                    .copy_from_slice(&b_tr.h[tt * b * h + bi * h..tt * b * h + (bi + 1) * h]);
            }
        }
        let mut y = vec![0.0f32; b * t * self.out];
        y.par_chunks_mut(self.out).enumerate().for_each(|(r, row)| {
            let cr = &cat[r * 2 * h..(r + 1) * 2 * h];
            for o in 0..self.out {
                let w = &self.proj_w[o * 2 * h..(o + 1) * 2 * h];
                let mut s = self.proj_b[o];
                for k in 0..2 * h {
                    s += cr[k] * w[k];
                }
                row[o] = s;
            }
        });
        (y, BiLstmTrace { x: x.to_vec(), f_tr, b_tr, cat, b, t })
    }

    /// `dy` [B, T, out]; returns dx [B, T, D]
    pub fn backward(&mut self, dy: &[f32], tr: &BiLstmTrace) -> Vec<f32> {
        let (b, t, h) = (tr.b, tr.t, self.h);
        let d = self.fwd.d;
        // projection backward
        let mut dcat = vec![0.0f32; b * t * 2 * h];
        dcat.par_chunks_mut(2 * h).enumerate().for_each(|(r, row)| {
            let dyr = &dy[r * self.out..(r + 1) * self.out];
            let cr = &tr.cat[r * 2 * h..(r + 1) * 2 * h];
            for o in 0..self.out {
                let dv = dyr[o];
                if dv == 0.0 {
                    continue;
                }
                for k in 0..2 * h {
                    row[k] += dv * self.proj_w[o * 2 * h + k];
                }
            }
            let _ = cr;
        });
        // accumulate proj grads serially (cheap enough vs the rest)
        for r in 0..b * t {
            for o in 0..self.out {
                let dv = dy[r * self.out + o];
                if dv == 0.0 {
                    continue;
                }
                self.g_proj_b[o] += dv;
                for k in 0..2 * h {
                    self.g_proj_w[o * 2 * h + k] += dv * tr.cat[r * 2 * h + k];
                }
            }
        }
        // split dcat into per-direction [T, B, H]
        let mut dh_f = vec![0.0f32; t * b * h];
        let mut dh_b = vec![0.0f32; t * b * h];
        for tt in 0..t {
            for bi in 0..b {
                dh_f[tt * b * h + bi * h..tt * b * h + (bi + 1) * h]
                    .copy_from_slice(&dcat[(bi * t + tt) * 2 * h..(bi * t + tt) * 2 * h + h]);
                dh_b[tt * b * h + bi * h..tt * b * h + (bi + 1) * h]
                    .copy_from_slice(&dcat[(bi * t + tt) * 2 * h + h..(bi * t + tt + 1) * 2 * h]);
            }
        }
        let dx1 = Self::bwd_dir(&mut self.fwd, &dh_f, tr, false);
        let dx2 = Self::bwd_dir(&mut self.bwd, &dh_b, tr, true);
        let mut dx = vec![0.0f32; b * t * d];
        for i in 0..dx.len() {
            dx[i] = dx1[i] + dx2[i];
        }
        dx
    }

    fn bwd_dir(l: &mut LstmDir, dh_out: &[f32], tr: &BiLstmTrace, rev: bool) -> Vec<f32> {
        let (b, t, h) = (tr.b, tr.t, l.h);
        let d = l.d;
        let trc = if rev { &tr.b_tr } else { &tr.f_tr };
        let mut dx = vec![0.0f32; t * b * d];
        let mut dc_next = vec![0.0f32; b * h];
        let mut dh_next = vec![0.0f32; b * h];
        for step in (0..t).rev() {
            let tt = if rev { t - 1 - step } else { step };
            let gr = &trc.gates[step * b * 4 * h..(step + 1) * b * 4 * h];
            let cr = &trc.c[step * b * h..(step + 1) * b * h];
            // h_prev / c_prev
            let (h_prev, c_prev): (Vec<f32>, Vec<f32>) = if step == 0 {
                (vec![0.0; b * h], vec![0.0; b * h])
            } else {
                (
                    trc.h[(step - 1) * b * h..step * b * h].to_vec(),
                    trc.c[(step - 1) * b * h..step * b * h].to_vec(),
                )
            };
            let mut dgates = vec![0.0f32; b * 4 * h];
            for bi in 0..b {
                for i in 0..h {
                    let ig = gr[bi * 4 * h + i];
                    let fg = gr[bi * 4 * h + h + i];
                    let cg = gr[bi * 4 * h + 2 * h + i];
                    let og = gr[bi * 4 * h + 3 * h + i];
                    let c = cr[bi * h + i];
                    let tc = c.tanh();
                    let dh = dh_out[step * b * h + bi * h + i] + dh_next[bi * h + i];
                    let dc = dh * og * (1.0 - tc * tc) + dc_next[bi * h + i] * fg;
                    dgates[bi * 4 * h + i] = dc * cg * ig * (1.0 - ig);
                    dgates[bi * 4 * h + h + i] = dc * c_prev[bi * h + i] * fg * (1.0 - fg);
                    dgates[bi * 4 * h + 2 * h + i] = dc * ig * (1.0 - cg * cg);
                    dgates[bi * 4 * h + 3 * h + i] = dh * tc * og * (1.0 - og);
                    dc_next[bi * h + i] = dc;
                    dh_next[bi * h + i] = 0.0; // consumed below via W_hh
                }
            }
            // dx = dgates @ W_ih ; dh_prev = dgates @ W_hh ; weight grads
            let xr = &tr.x[tt * b * d..(tt + 1) * b * d];
            for bi in 0..b {
                let dg = &dgates[bi * 4 * h..(bi + 1) * 4 * h];
                for k in 0..d {
                    let mut s = 0.0;
                    for o in 0..4 * h {
                        s += dg[o] * l.w_ih[o * d + k];
                    }
                    dx[tt * b * d + bi * d + k] = s;
                }
                for k in 0..h {
                    let mut s = 0.0;
                    for o in 0..4 * h {
                        s += dg[o] * l.w_hh[o * h + k];
                    }
                    dh_next[bi * h + k] = s;
                }
                for o in 0..4 * h {
                    let dv = dg[o];
                    if dv == 0.0 {
                        continue;
                    }
                    l.g_b_ih[o] += dv;
                    l.g_b_hh[o] += dv;
                    for k in 0..d {
                        l.g_w_ih[o * d + k] += dv * xr[bi * d + k];
                    }
                    for k in 0..h {
                        l.g_w_hh[o * h + k] += dv * h_prev[bi * h + k];
                    }
                }
            }
        }
        dx
    }
}

pub struct BiLstmStack {
    pub layers: Vec<BiLstmLayer>,
}

impl BiLstmStack {
    /// reference: 2 layers, hidden 256, first input = cnn out (512)
    pub fn new(in_dim: usize, hidden: usize, n_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let d = if i == 0 { in_dim } else { hidden };
            layers.push(BiLstmLayer::new(d, hidden, hidden));
        }
        Self { layers }
    }
    pub fn zero_grad(&mut self) {
        for l in &mut self.layers {
            l.zero_grad();
        }
    }
    pub fn forward(&self, x: &[f32], b: usize, t: usize) -> (Vec<f32>, Vec<BiLstmTrace>) {
        let mut cur = x.to_vec();
        let mut traces = Vec::new();
        for l in &self.layers {
            let (y, tr) = l.forward(&cur, b, t);
            cur = y;
            traces.push(tr);
        }
        (cur, traces)
    }
    pub fn backward(&mut self, dy: &[f32], traces: &[BiLstmTrace]) -> Vec<f32> {
        let mut cur = dy.to_vec();
        for (l, tr) in self.layers.iter_mut().zip(traces.iter()).rev() {
            cur = l.backward(&cur, tr);
        }
        cur
    }
}
