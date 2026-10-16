//! INT8-QAT port of the reference `SEResNet31`
//! (`manuscript-ocr/recognizers/_trba/model/seresnet31.py`).
//!
//! Graph (identical topology to the PyTorch original):
//!
//! ```text
//! stem    : conv3x3(3,64)+BN+ReLU, conv3x3(64,128)+BN+ReLU, maxpool2x2
//! layer1  : SEBasicBlock(128,256, stride 2, 1x1 downsample)
//! layer2  : SEBasicBlock(256,256) x2
//! layer3  : SEBasicBlock(256,512, stride 2, 1x1 downsample) + SEBasicBlock(512,512) x4
//! layer4  : SEBasicBlock(512,512) x3
//! conv_out: conv2x2 s(2,1) p(0,1)(512,512)+BN+ReLU, conv2x2(512,512)+BN+ReLU
//! ```
//!
//! Quantisation discipline (fake-QAT, "Towards Unified INT8 Training" style):
//! * conv GEMMs run on int8 operands with int32 accumulation; the accumulator
//!   is dequantised by `s_act * s_w` and BatchNorm runs in fp32 on top of it;
//! * activations are re-quantised per tensor with a dynamic max-abs scale at
//!   every ReLU output / residual sum (straight-through on the way back);
//! * conv weight gradients use an int8 GEMM (quantised `dX` x int8 `col`),
//!   i.e. gradients travel in int8 too, exactly as in the paper;
//! * `grad_input` uses the fused fp32 gather kernel (no col materialisation).
//!
//! Convention: every int8 tensor `q` represents the value `v = q * s` with a
//! per-tensor scale `s`.  A backward edge therefore hands over `dL/dv`
//! (fp32, device); the consumer multiplies by its own `s` and re-quantises.

use std::sync::Arc;

use crate::backend::*;

type CV = core::ffi::c_void;

pub const BN_EPS: f32 = 1e-5;
pub const BN_MOMENTUM: f32 = 0.1;

// ------------------------------------------------------------ small layers --

#[derive(Clone)]
pub struct BatchNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub run_mean: Vec<f32>,
    pub run_var: Vec<f32>,
    pub g_gamma: Vec<f32>,
    pub g_beta: Vec<f32>,
    pub c: usize,
}

impl BatchNorm {
    pub fn new(c: usize) -> Self {
        Self {
            gamma: vec![1.0; c],
            beta: vec![0.0; c],
            run_mean: vec![0.0; c],
            run_var: vec![1.0; c],
            g_gamma: vec![0.0; c],
            g_beta: vec![0.0; c],
            c,
        }
    }
    pub fn update_running(&mut self, mean: &[f32], var: &[f32]) {
        for i in 0..self.c {
            self.run_mean[i] = (1.0 - BN_MOMENTUM) * self.run_mean[i] + BN_MOMENTUM * mean[i];
            self.run_var[i] = (1.0 - BN_MOMENTUM) * self.run_var[i] + BN_MOMENTUM * var[i];
        }
    }
    pub fn zero_grad(&mut self) {
        self.g_gamma.iter_mut().for_each(|x| *x = 0.0);
        self.g_beta.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Plain fp32 fully-connected layer (SE bottlenecks).  Host side: tiny.
#[derive(Clone)]
pub struct LinearF {
    pub w: Vec<f32>, // [out, in] row-major
    pub b: Vec<f32>,
    pub g_w: Vec<f32>,
    pub g_b: Vec<f32>,
    pub inp: usize,
    pub out: usize,
    pub use_bias: bool,
}

impl LinearF {
    pub fn new(inp: usize, out: usize, use_bias: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (inp + out) as f32).sqrt();
        Self {
            w: (0..inp * out).map(|_| rng.gen_range(-limit..limit)).collect(),
            b: vec![0.0; out],
            g_w: vec![0.0; inp * out],
            g_b: vec![0.0; out],
            inp,
            out,
            use_bias,
        }
    }
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out];
        for r in 0..rows {
            for o in 0..self.out {
                let mut s = if self.use_bias { self.b[o] } else { 0.0 };
                for i in 0..self.inp {
                    s += x[r * self.inp + i] * self.w[o * self.inp + i];
                }
                y[r * self.out + o] = s;
            }
        }
        y
    }
    pub fn backward(&mut self, x: &[f32], dy: &[f32], rows: usize) -> Vec<f32> {
        let mut dx = vec![0.0f32; rows * self.inp];
        for r in 0..rows {
            for o in 0..self.out {
                let d = dy[r * self.out + o];
                if d == 0.0 {
                    continue;
                }
                if self.use_bias {
                    self.g_b[o] += d;
                }
                for i in 0..self.inp {
                    self.g_w[o * self.inp + i] += d * x[r * self.inp + i];
                    dx[r * self.inp + i] += d * self.w[o * self.inp + i];
                }
            }
        }
        dx
    }
    /// single-row backward: accumulates grads, returns dx
    pub fn backward_row(&mut self, x: &[f32], dy: &[f32]) -> Vec<f32> {
        let mut dx = vec![0.0f32; self.inp];
        for o in 0..self.out {
            let d = dy[o];
            if d == 0.0 {
                continue;
            }
            if self.use_bias {
                self.g_b[o] += d;
            }
            for i in 0..self.inp {
                self.g_w[o * self.inp + i] += d * x[i];
                dx[i] += d * self.w[o * self.inp + i];
            }
        }
        dx
    }
    pub fn zero_grad(&mut self) {
        self.g_w.iter_mut().for_each(|x| *x = 0.0);
        self.g_b.iter_mut().for_each(|x| *x = 0.0);
    }
}

// ---------------------------------------------------------------- scratch --

/// Growable pool of device buffers reused across batches/layers.
pub struct Scratch {
    f32s: Vec<DevBuf>,
    i8s: Vec<DevBuf>,
    i32s: Vec<DevBuf>,
}

impl Scratch {
    pub fn new() -> Self {
        Self { f32s: Vec::new(), i8s: Vec::new(), i32s: Vec::new() }
    }
    /// NOTE: the returned pointer stays valid until the same slot is
    /// requested again with a *larger* size.
    pub fn f32(&mut self, slot: usize, elems: usize) -> *mut f32 {
        while self.f32s.len() <= slot {
            self.f32s.push(DevBuf::new(0));
        }
        let need = elems * 4;
        if self.f32s[slot].len < need {
            self.f32s[slot] = DevBuf::new(need);
        }
        self.f32s[slot].ptr as *mut f32
    }
    pub fn i8(&mut self, slot: usize, elems: usize) -> *mut i8 {
        while self.i8s.len() <= slot {
            self.i8s.push(DevBuf::new(0));
        }
        if self.i8s[slot].len < elems {
            self.i8s[slot] = DevBuf::new(elems);
        }
        self.i8s[slot].ptr as *mut i8
    }
    pub fn i32(&mut self, slot: usize, elems: usize) -> *mut i32 {
        while self.i32s.len() <= slot {
            self.i32s.push(DevBuf::new(0));
        }
        let need = elems * 4;
        if self.i32s[slot].len < need {
            self.i32s[slot] = DevBuf::new(need);
        }
        self.i32s[slot].ptr as *mut i32
    }
}

// scratch slots
const S_COL: usize = 0;
const S_ACC: usize = 0;
const S_X: usize = 1;
const S_Y: usize = 2;
const S_T1: usize = 3;
const S_T2: usize = 4;
const S_T3: usize = 5;
const S_T4: usize = 6;
const S_T5: usize = 7;
const S_MEAN: usize = 8;
const S_VAR: usize = 9;
const S_SDY: usize = 10;
const S_SDXH: usize = 11;
const S_MAX: usize = 12;
const S_SMALL: usize = 13; // [B, C] pools / gates
const S_QT: usize = 0;
const S_QTT: usize = 1;

// ------------------------------------------------------------------- conv --

pub struct ConvQat {
    pub name: String,
    pub in_c: usize,
    pub out_c: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub ph: usize,
    pub pw: usize,
    pub w: Vec<f32>, // [out_c, col_cols]
    pub w_q: Vec<i8>,
    pub w_q_pad: Vec<i8>,
    pub w_s: f32,
    pub bn: BatchNorm,
    pub g_w: Vec<f32>,
}

pub struct ConvFwd {
    pub x: DevBuf, // pre-BN, f32 NCHW
    pub y: DevBuf, // post-BN pre-ReLU, f32 NCHW
    pub mean: DevBuf,
    pub var: DevBuf,
    pub mean_h: Vec<f32>,
    pub var_h: Vec<f32>,
    pub h_out: usize,
    pub w_out: usize,
}

pub struct ConvTrace {
    pub b: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub s_in: f32,
    pub x: DevBuf,
    pub mean: DevBuf,
    pub var: DevBuf,
    pub mean_h: Vec<f32>,
    pub var_h: Vec<f32>,
    pub q_in: Arc<DevBuf>,
}

impl ConvQat {
    pub fn new(
        name: &str,
        in_c: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let col = in_c * kh * kw;
        let limit = (6.0 / (col + out_c * kh * kw) as f32).sqrt();
        let w: Vec<f32> = (0..out_c * col).map(|_| rng.gen_range(-limit..limit)).collect();
        let mut c = Self {
            name: name.into(),
            in_c,
            out_c,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            w,
            w_q: Vec::new(),
            w_q_pad: Vec::new(),
            w_s: 1.0,
            bn: BatchNorm::new(out_c),
            g_w: vec![0.0; out_c * col],
        };
        c.requantize();
        c
    }

    pub fn col_cols(&self) -> usize {
        self.in_c * self.kh * self.kw
    }
    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        (
            (h + 2 * self.ph - self.kh) / self.sh + 1,
            (w + 2 * self.pw - self.kw) / self.sw + 1,
        )
    }

    pub fn requantize(&mut self) {
        let mx = self.w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        self.w_s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        self.w_q = self
            .w
            .iter()
            .map(|&v| (v / self.w_s).clamp(-127.0, 127.0).round() as i8)
            .collect();
        let k = self.col_cols();
        let rp = align16(self.out_c);
        let cp = align16(k);
        self.w_q_pad = vec![0i8; rp * cp];
        for r in 0..self.out_c {
            self.w_q_pad[r * cp..r * cp + k].copy_from_slice(&self.w_q[r * k..(r + 1) * k]);
        }
    }

    pub fn zero_grad(&mut self) {
        self.g_w.iter_mut().for_each(|x| *x = 0.0);
        self.bn.zero_grad();
    }

    pub fn forward(
        &self,
        q_in: &DevBuf,
        s_in: f32,
        b: usize,
        h: usize,
        w: usize,
        train: bool,
        stream: &CudaStream,
        sc: &mut Scratch,
    ) -> ConvFwd {
        let (ho, wo) = self.out_hw(h, w);
        let rows = b * ho * wo;
        let k = self.col_cols();
        let kp = align16(k);
        let rp = align16(rows);

        let col = sc.i8(S_COL, rp * kp);
        unsafe {
            launch_im2col_pad(
                q_in.as_i8(),
                col,
                b as i32,
                self.in_c as i32,
                h as i32,
                w as i32,
                self.kh as i32,
                self.kw as i32,
                self.sh as i32,
                self.sw as i32,
                self.ph as i32,
                self.pw as i32,
                ho as i32,
                wo as i32,
                kp as i32,
                stream.0,
            );
        }
        let acc = sc.i32(S_ACC, rows * self.out_c);
        unsafe {
            launch_gemm_int8(
                col as *const i8,
                self.w_q_pad.as_ptr(),
                acc,
                rows as i32,
                self.out_c as i32,
                k as i32,
                stream.0,
            );
        }
        let n = b * self.out_c * ho * wo;
        let x = sc.f32(S_X, n);
        unsafe {
            launch_epilogue_f32(
                acc as *const i32,
                self.out_c as i32,
                s_in * self.w_s,
                x,
                self.out_c as i32,
                ho as i32,
                wo as i32,
                n as i64,
                stream.0,
            );
        }
        let mean = sc.f32(S_MEAN, self.out_c);
        let var = sc.f32(S_VAR, self.out_c);
        if train {
            unsafe {
                launch_bn_stats(
                    x as *const f32,
                    b as i32,
                    self.out_c as i32,
                    (ho * wo) as i32,
                    mean,
                    var,
                    stream.0,
                );
            }
        } else {
            unsafe {
                cuda_memcpy_h2d(mean as *mut CV, self.bn.run_mean.as_ptr() as *const CV, self.out_c * 4, stream.0);
                cuda_memcpy_h2d(var as *mut CV, self.bn.run_var.as_ptr() as *const CV, self.out_c * 4, stream.0);
            }
        }
        let y = sc.f32(S_Y, n);
        unsafe {
            launch_bn_apply(
                x as *const f32,
                mean as *const f32,
                var as *const f32,
                self.bn.gamma.as_ptr(),
                self.bn.beta.as_ptr(),
                BN_EPS,
                y,
                self.out_c as i32,
                (ho * wo) as i32,
                n as i64,
                stream.0,
            );
        }
        let mut mean_h = vec![0.0f32; self.out_c];
        let mut var_h = vec![0.0f32; self.out_c];
        unsafe {
            cuda_memcpy_d2h(mean_h.as_mut_ptr() as *mut CV, mean as *mut CV, self.out_c * 4, stream.0);
            cuda_memcpy_d2h(var_h.as_mut_ptr() as *mut CV, var as *mut CV, self.out_c * 4, stream.0);
        }
        stream.sync();

        let ox = DevBuf::new(n * 4);
        let oy = DevBuf::new(n * 4);
        let om = DevBuf::new(self.out_c * 4);
        let ov = DevBuf::new(self.out_c * 4);
        unsafe {
            cuda_memcpy_dtod(ox.ptr, x as *mut CV, n * 4, stream.0);
            cuda_memcpy_dtod(oy.ptr, y as *mut CV, n * 4, stream.0);
            cuda_memcpy_dtod(om.ptr, mean as *mut CV, self.out_c * 4, stream.0);
            cuda_memcpy_dtod(ov.ptr, var as *mut CV, self.out_c * 4, stream.0);
        }
        ConvFwd { x: ox, y: oy, mean: om, var: ov, mean_h, var_h, h_out: ho, w_out: wo }
    }

    /// `dy` = dL/d(BN output value), device f32.  Returns dL/d(input value).
    pub fn backward(
        &mut self,
        dy: *const f32,
        tr: &ConvTrace,
        stream: &CudaStream,
        sc: &mut Scratch,
    ) -> DevBuf {
        let b = tr.b;
        let (hi, wi) = (tr.h_in, tr.w_in);
        let (ho, wo) = (tr.h_out, tr.w_out);
        let rows = b * ho * wo;
        let n_in = b * self.in_c * hi * wi;
        let n_out = b * self.out_c * ho * wo;
        let hw = ho * wo;
        let k = self.col_cols();
        let kp = align16(k);
        let rp = align16(rows);

        let sdy = sc.f32(S_SDY, self.out_c);
        let sdxh = sc.f32(S_SDXH, self.out_c);
        unsafe {
            launch_bn_bwd_sums(
                dy,
                tr.x.as_f32(),
                tr.mean.as_f32(),
                tr.var.as_f32(),
                BN_EPS,
                self.out_c as i32,
                hw as i32,
                n_out as i64,
                sdy,
                sdxh,
                stream.0,
            );
        }
        let dx = sc.f32(S_T1, n_out);
        unsafe {
            launch_bn_bwd_dx(
                dy,
                tr.x.as_f32(),
                tr.mean.as_f32(),
                tr.var.as_f32(),
                self.bn.gamma.as_ptr(),
                sdy as *const f32,
                sdxh as *const f32,
                BN_EPS,
                self.out_c as i32,
                hw as i32,
                n_out as i64,
                (b * hw) as i64,
                dx,
                stream.0,
            );
        }
        let mut g = vec![0.0f32; self.out_c];
        unsafe { cuda_memcpy_d2h(g.as_mut_ptr() as *mut CV, sdy as *mut CV, self.out_c * 4, stream.0) };
        for i in 0..self.out_c {
            self.bn.g_beta[i] += g[i];
        }
        unsafe { cuda_memcpy_d2h(g.as_mut_ptr() as *mut CV, sdxh as *mut CV, self.out_c * 4, stream.0) };
        for i in 0..self.out_c {
            self.bn.g_gamma[i] += g[i];
        }

        // grad_W: int8 GEMM (quantised dX)^T x col(input)
        let maxb = sc.f32(S_MAX, 1);
        unsafe { launch_reduce_max_abs(dx as *const f32, n_out as i64, maxb, stream.0) };
        let mut mx = [0.0f32];
        unsafe { cuda_memcpy_d2h(mx.as_mut_ptr() as *mut CV, maxb as *mut CV, 4, stream.0) };
        stream.sync();
        let s_dx = (mx[0] / 127.0).max(1e-12);
        let qdx = sc.i8(S_QT, n_out);
        unsafe { launch_quant_f32_i8(dx as *const f32, n_out as i64, s_dx, qdx, stream.0) };
        let op = align16(self.out_c);
        let qdx_t = sc.i8(S_QTT, op * rp);
        unsafe {
            launch_transpose_i8(qdx as *const i8, rows as i32, self.out_c as i32, qdx_t, stream.0);
        }
        let col = sc.i8(S_COL, rp * kp);
        unsafe {
            launch_im2col_pad(
                tr.q_in.as_i8(),
                col,
                b as i32,
                self.in_c as i32,
                hi as i32,
                wi as i32,
                self.kh as i32,
                self.kw as i32,
                self.sh as i32,
                self.sw as i32,
                self.ph as i32,
                self.pw as i32,
                ho as i32,
                wo as i32,
                kp as i32,
                stream.0,
            );
        }
        let gw = sc.i32(S_ACC, self.out_c * k);
        unsafe {
            launch_gemm_int8(
                qdx_t as *const i8,
                col as *const i8,
                gw,
                self.out_c as i32,
                k as i32,
                rows as i32,
                stream.0,
            );
        }
        let mut gw_raw = vec![0i32; self.out_c * k];
        unsafe {
            cuda_memcpy_d2h(gw_raw.as_mut_ptr() as *mut CV, gw as *mut CV, self.out_c * k * 4, stream.0);
        }
        stream.sync();
        let deq = s_dx * tr.s_in;
        for i in 0..self.out_c * k {
            self.g_w[i] += gw_raw[i] as f32 * deq;
        }

        let gi = DevBuf::new(n_in * 4);
        unsafe {
            launch_conv_backward_grad_input(
                dy,
                self.w.as_ptr(),
                gi.as_f32_mut(),
                b as i32,
                self.in_c as i32,
                hi as i32,
                wi as i32,
                self.out_c as i32,
                self.kh as i32,
                self.kw as i32,
                self.sh as i32,
                self.sw as i32,
                self.ph as i32,
                self.pw as i32,
                ho as i32,
                wo as i32,
                stream.0,
            );
        }
        gi
    }
}

// ------------------------------------------------------------- quant utils --

/// dynamic max-abs quantisation of a device f32 tensor into an owned buffer
pub fn quantize_owned(v: *const f32, n: usize, stream: &CudaStream, sc: &mut Scratch) -> (Arc<DevBuf>, f32) {
    let maxb = sc.f32(S_MAX, 1);
    unsafe { launch_reduce_max_abs(v, n as i64, maxb, stream.0) };
    let mut mx = [0.0f32];
    unsafe { cuda_memcpy_d2h(mx.as_mut_ptr() as *mut CV, maxb as *mut CV, 4, stream.0) };
    stream.sync();
    let s = (mx[0] / 127.0).max(1e-12);
    let q = Arc::new(DevBuf::new(n));
    unsafe { launch_quant_f32_i8(v, n as i64, s, q.as_i8_mut(), stream.0) };
    (q, s)
}

/// ReLU + dynamic quantisation (the standard activation boundary)
pub fn relu_quant_owned(y: *const f32, n: usize, stream: &CudaStream, sc: &mut Scratch) -> (Arc<DevBuf>, f32) {
    let maxb = sc.f32(S_MAX, 1);
    unsafe { launch_reduce_max_abs(y, n as i64, maxb, stream.0) };
    let mut mx = [0.0f32];
    unsafe { cuda_memcpy_d2h(mx.as_mut_ptr() as *mut CV, maxb as *mut CV, 4, stream.0) };
    stream.sync();
    let s = (mx[0] / 127.0).max(1e-12);
    let q = Arc::new(DevBuf::new(n));
    unsafe { launch_relu_quant(y, n as i64, s, q.as_i8_mut(), stream.0) };
    (q, s)
}

// ------------------------------------------------------------------ blocks --

pub struct SELayer {
    pub fc1: LinearF,
    pub fc2: LinearF,
}

impl SELayer {
    pub fn new(c: usize, reduction: usize) -> Self {
        Self { fc1: LinearF::new(c, c / reduction, false), fc2: LinearF::new(c / reduction, c, false) }
    }
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }
}

pub struct BasicBlock {
    pub conv1: ConvQat,
    pub conv2: ConvQat,
    pub se: SELayer,
    pub ds: Option<ConvQat>,
}

pub struct BlockTrace {
    pub c1: ConvTrace,
    pub c2: ConvTrace,
    pub ds: Option<ConvTrace>,
    pub q1: Arc<DevBuf>,
    pub s1: f32,
    pub y2: DevBuf,
    pub gate: Vec<f32>,
    pub pooled: Vec<f32>,
    pub h1: Vec<f32>,
    pub b: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub c: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub s_in: f32,
    pub q_in: Arc<DevBuf>,
}

impl BasicBlock {
    pub fn new(in_c: usize, out_c: usize, stride: usize, name: &str) -> Self {
        let ds = if stride != 1 || in_c != out_c {
            Some(ConvQat::new(&format!("{name}.downsample.0"), in_c, out_c, 1, 1, stride, stride, 0, 0))
        } else {
            None
        };
        Self {
            conv1: ConvQat::new(&format!("{name}.conv1"), in_c, out_c, 3, 3, stride, stride, 1, 1),
            conv2: ConvQat::new(&format!("{name}.conv2"), out_c, out_c, 3, 3, 1, 1, 1, 1),
            se: SELayer::new(out_c, 16),
            ds,
        }
    }

    pub fn zero_grad(&mut self) {
        self.conv1.zero_grad();
        self.conv2.zero_grad();
        self.se.zero_grad();
        if let Some(d) = &mut self.ds {
            d.zero_grad();
        }
    }

    pub fn forward(
        &mut self,
        q_in: &Arc<DevBuf>,
        s_in: f32,
        b: usize,
        h: usize,
        w: usize,
        train: bool,
        stream: &CudaStream,
        sc: &mut Scratch,
    ) -> (Arc<DevBuf>, f32, BlockTrace) {
        let (ho1, wo1) = self.conv1.out_hw(h, w);
        let c = self.conv2.out_c;
        let n_out = b * c * ho1 * wo1;

        let ds_fwd: Option<ConvFwd> = if let Some(ds) = &self.ds {
            Some(ds.forward(q_in, s_in, b, h, w, train, stream, sc))
        } else {
            None
        };

        let f1 = self.conv1.forward(q_in, s_in, b, h, w, train, stream, sc);
        let (h1o, w1o) = (f1.h_out, f1.w_out);
        let n1 = b * self.conv1.out_c * h1o * w1o;
        let (q1, s1) = relu_quant_owned(f1.y.as_f32(), n1, stream, sc);

        let f2 = self.conv2.forward(&q1, s1, b, h1o, w1o, train, stream, sc);
        let (h2, w2) = (f2.h_out, f2.w_out);

        // SE on the pre-ReLU BN output of conv2 (as in the reference)
        let pooled_buf = sc.f32(S_SMALL, b * c);
        unsafe {
            launch_avgpool_hw(f2.y.as_f32(), c as i32, (h2 * w2) as i32, b as i64, pooled_buf, stream.0);
        }
        let mut pooled = vec![0.0f32; b * c];
        unsafe { cuda_memcpy_d2h(pooled.as_mut_ptr() as *mut CV, pooled_buf as *mut CV, b * c * 4, stream.0) };
        stream.sync();
        let pre1 = self.se.fc1.forward(&pooled, b);
        let h1v: Vec<f32> = pre1.iter().map(|&v| v.max(0.0)).collect();
        let pre2 = self.se.fc2.forward(&h1v, b);
        let gate: Vec<f32> = pre2.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

        let gate_buf = sc.f32(S_SMALL, b * c);
        unsafe { cuda_memcpy_h2d(gate_buf as *mut CV, gate.as_ptr() as *const CV, b * c * 4, stream.0) };
        let yse = sc.f32(S_T3, n_out);
        unsafe {
            launch_mul_gate(f2.y.as_f32(), gate_buf as *const f32, c as i32, (h2 * w2) as i32, n_out as i64, yse, stream.0);
        }
        let z = sc.f32(S_T4, n_out);
        match &ds_fwd {
            Some(f) => unsafe { launch_add_f32(yse as *const f32, f.y.as_f32(), n_out as i64, z, stream.0) },
            None => {
                let id = sc.f32(S_T5, n_out);
                unsafe { launch_dequant_i8_f32(q_in.as_i8(), n_out as i64, s_in, id, stream.0) };
                unsafe { launch_add_f32(yse as *const f32, id as *const f32, n_out as i64, z, stream.0) };
            }
        }
        let (q_out, s_z) = relu_quant_owned(z as *const f32, n_out, stream, sc);

        let ds_trace = ds_fwd.map(|f| trace_of(f, b, h, w, s_in, q_in.clone()));

        let tr = BlockTrace {
            c1: {
                let x = f1.x; let mean = f1.mean; let var = f1.var;
                let mh = f1.mean_h.clone(); let vh = f1.var_h.clone();
                ConvTrace { b, h_in: h, w_in: w, h_out: h1o, w_out: w1o, s_in, x, mean, var, mean_h: mh, var_h: vh, q_in: q_in.clone() }
            },
            c2: {
                let x = f2.x; let mean = f2.mean; let var = f2.var;
                let mh = f2.mean_h.clone(); let vh = f2.var_h.clone();
                ConvTrace { b, h_in: h1o, w_in: w1o, h_out: h2, w_out: w2, s_in: s1, x, mean, var, mean_h: mh, var_h: vh, q_in: q1.clone() }
            },
            ds: ds_trace,
            q1,
            s1,
            y2: f2.y,
            gate,
            pooled,
            h1: h1v,
            b,
            h_out: h2,
            w_out: w2,
            c,
            h_in: h,
            w_in: w,
            s_in,
            q_in: q_in.clone(),
        };
        (q_out, s_z, tr)
    }

    /// `dq_out`/`s_dq`: dL/dq_int of the block output.  `q_out` needed for the
    /// ReLU mask.  Returns dL/d(input value) as device f32.
    pub fn backward(
        &mut self,
        dq_out: &DevBuf,
        s_dq: f32,
        q_out: &DevBuf,
        tr: &BlockTrace,
        stream: &CudaStream,
        sc: &mut Scratch,
    ) -> DevBuf {
        let b = tr.b;
        let (ho, wo) = (tr.h_out, tr.w_out);
        let n_out = b * tr.c * ho * wo;
        let hw = ho * wo;
        let n_in = b * self.conv1.in_c * tr.h_in * tr.w_in;

        // dz = dL/dq * s_dq, masked by the ReLU sign of the block output
        let gvz = sc.f32(S_T1, n_out);
        unsafe { launch_dequant_i8_f32(dq_out.as_i8(), n_out as i64, s_dq, gvz, stream.0) };
        let qzv = sc.f32(S_T5, n_out);
        unsafe { launch_dequant_i8_f32(q_out.as_i8(), n_out as i64, 1.0, qzv, stream.0) };
        let dz = DevBuf::new(n_out * 4);
        unsafe { launch_mask_relu_bwd(qzv as *const f32, gvz as *const f32, n_out as i64, dz.as_f32_mut(), stream.0) };

        // ---- SE backward ----
        let gate_buf = sc.f32(S_SMALL, b * tr.c);
        unsafe { cuda_memcpy_h2d(gate_buf as *mut CV, tr.gate.as_ptr() as *const CV, b * tr.c * 4, stream.0) };
        let dy2 = sc.f32(S_T2, n_out);
        unsafe {
            launch_mul_gate(dz.as_f32(), gate_buf as *const f32, tr.c as i32, hw as i32, n_out as i64, dy2, stream.0);
        }
        let prod = sc.f32(S_T3, n_out);
        unsafe { launch_mul_f32(dz.as_f32(), tr.y2.as_f32(), n_out as i64, prod, stream.0) };
        let poolmean = sc.f32(S_SMALL, b * tr.c);
        unsafe { launch_avgpool_hw(prod as *const f32, tr.c as i32, hw as i32, b as i64, poolmean, stream.0) };
        let mut dgate = vec![0.0f32; b * tr.c];
        unsafe { cuda_memcpy_d2h(dgate.as_mut_ptr() as *mut CV, poolmean as *mut CV, b * tr.c * 4, stream.0) };
        stream.sync();
        for g in dgate.iter_mut() {
            *g *= hw as f32; // mean -> sum
        }
        // sigmoid + fc2 + relu + fc1 on the host
        let mut dpre2 = vec![0.0f32; b * tr.c];
        for i in 0..b * tr.c {
            dpre2[i] = dgate[i] * tr.gate[i] * (1.0 - tr.gate[i]);
        }
        let dh1 = self.se.fc2.backward(&tr.h1, &dpre2, b);
        let dpre1: Vec<f32> = dh1.iter().zip(tr.h1.iter()).map(|(&d, &h)| d * if h > 0.0 { 1.0 } else { 0.0 }).collect();
        let dpooled = self.se.fc1.backward(&tr.pooled, &dpre1, b);
        let apb = sc.f32(S_T3, n_out);
        let dp_buf = sc.f32(S_SMALL, b * tr.c);
        unsafe { cuda_memcpy_h2d(dp_buf as *mut CV, dpooled.as_ptr() as *const CV, b * tr.c * 4, stream.0) };
        unsafe {
            launch_avgpool_hw_backward(dp_buf as *const f32, tr.c as i32, hw as i32, b as i64, apb, stream.0);
        }
        let dy2_full = sc.f32(S_T4, n_out);
        unsafe { launch_add_f32(dy2 as *const f32, apb as *const f32, n_out as i64, dy2_full, stream.0) };

        // ---- conv2 backward (no ReLU after its BN) ----
        let gv1 = self.conv2.backward(dy2_full as *const f32, &tr.c2, stream, sc); // dL/d v1 (value of q1)
        // mask ReLU of conv1: dy1 = gv1 * (q1 > 0)
        let q1v = sc.f32(S_T5, n1_len(tr));
        unsafe { launch_dequant_i8_f32(tr.q1.as_i8(), n1_len(tr) as i64, 1.0, q1v, stream.0) };
        let dy1 = DevBuf::new(n1_len(tr) * 4);
        unsafe { launch_mask_relu_bwd(q1v as *const f32, gv1.as_f32(), n1_len(tr) as i64, dy1.as_f32_mut(), stream.0) };

        // ---- conv1 backward ----
        let gi_in = self.conv1.backward(dy1.as_f32(), &tr.c1, stream, sc); // dL/d v_in

        // ---- skip branch ----
        let total = DevBuf::new(n_in * 4);
        match (&mut self.ds, &tr.ds) {
            (Some(ds), Some(dtr)) => {
                let dx_ds = ds.backward(dz.as_f32(), dtr, stream, sc); // dL/d v_in via skip
                unsafe { launch_add_f32(gi_in.as_f32(), dx_ds.as_f32(), n_in as i64, total.as_f32_mut(), stream.0) };
            }
            _ => {
                unsafe { launch_add_f32(gi_in.as_f32(), dz.as_f32(), n_in as i64, total.as_f32_mut(), stream.0) };
            }
        }
        total
    }
}

fn n1_len(tr: &BlockTrace) -> usize {
    tr.b * tr.c * tr.c1.h_out * tr.c1.w_out
}

fn trace_of(f: ConvFwd, b: usize, h_in: usize, w_in: usize, s_in: f32, q_in: Arc<DevBuf>) -> ConvTrace {
    ConvTrace {
        b,
        h_in,
        w_in,
        h_out: f.h_out,
        w_out: f.w_out,
        s_in,
        x: f.x,
        mean: f.mean,
        var: f.var,
        mean_h: f.mean_h,
        var_h: f.var_h,
        q_in,
    }
}

// ------------------------------------------------------------------- net --

pub struct StemTrace {
    pub c0: ConvTrace,
    pub c1: ConvTrace,
    pub q0: Arc<DevBuf>,
    pub s0: f32,
    pub q1: Arc<DevBuf>,
    pub s1: f32,
    pub pool_idx: Vec<u64>,
    pub b: usize,
    pub h_pool_in: usize,
    pub w_pool_in: usize,
}

pub struct OutTrace {
    pub c0: ConvTrace,
    pub c1: ConvTrace,
    pub q0: Arc<DevBuf>,
    pub s0: f32,
    pub b: usize,
    pub h_feat: usize, // H before meanpool (1 or 2)
    pub w_feat: usize,
}

pub struct NetTrace {
    pub stem: StemTrace,
    pub blocks: Vec<BlockTrace>,
    pub out: OutTrace,
    pub q_chain: Vec<(Arc<DevBuf>, f32)>, // quantised output + scale of every stage
    pub s_img: f32,
}

pub struct SEResNetQ {
    pub c0: ConvQat,
    pub c1: ConvQat,
    pub layer1: Vec<BasicBlock>,
    pub layer2: Vec<BasicBlock>,
    pub layer3: Vec<BasicBlock>,
    pub layer4: Vec<BasicBlock>,
    pub o0: ConvQat,
    pub o1: ConvQat,
}

impl SEResNetQ {
    pub fn new() -> Self {
        Self {
            c0: ConvQat::new("stem.0", 3, 64, 3, 3, 1, 1, 1, 1),
            c1: ConvQat::new("stem.3", 64, 128, 3, 3, 1, 1, 1, 1),
            layer1: vec![BasicBlock::new(128, 256, 2, "layer1.0")],
            layer2: vec![BasicBlock::new(256, 256, 1, "layer2.0"), BasicBlock::new(256, 256, 1, "layer2.1")],
            layer3: {
                let mut v = vec![BasicBlock::new(256, 512, 2, "layer3.0")];
                for i in 1..5 {
                    v.push(BasicBlock::new(512, 512, 1, &format!("layer3.{i}")));
                }
                v
            },
            layer4: (0..3).map(|i| BasicBlock::new(512, 512, 1, &format!("layer4.{i}"))).collect(),
            o0: ConvQat::new("conv_out.0", 512, 512, 2, 2, 2, 1, 0, 1),
            o1: ConvQat::new("conv_out.3", 512, 512, 2, 2, 1, 1, 0, 0),
        }
    }

    pub fn convs_mut(&mut self) -> Vec<&mut ConvQat> {
        let mut v = vec![&mut self.c0, &mut self.c1];
        for l in [&mut self.layer1, &mut self.layer2, &mut self.layer3, &mut self.layer4] {
            for b in l.iter_mut() {
                if let Some(d) = &mut b.ds {
                    v.push(d);
                }
                v.push(&mut b.conv1);
                v.push(&mut b.conv2);
            }
        }
        v.push(&mut self.o0);
        v.push(&mut self.o1);
        v
    }

    pub fn zero_grad(&mut self) {
        for c in self.convs_mut() {
            c.zero_grad();
        }
        for l in [&mut self.layer1, &mut self.layer2, &mut self.layer3, &mut self.layer4] {
            for b in l.iter_mut() {
                b.se.zero_grad();
            }
        }
    }

    pub fn requantize_all(&mut self) {
        for c in self.convs_mut() {
            c.requantize();
        }
    }

    /// `img` = int8 image activations with scale `s_img` (values in [-1, 1]).
    /// Returns encoder input features `[B, T, 512]` (device f32) + trace.
    pub fn forward(
        &mut self,
        img: &Arc<DevBuf>,
        s_img: f32,
        b: usize,
        train: bool,
        img_h: usize,
        img_w: usize,
        stream: &CudaStream,
        sc: &mut Scratch,
    ) -> (DevBuf, NetTrace) {
        let mut h = img_h;
        let mut w = img_w;
        let mut chain: Vec<(Arc<DevBuf>, f32)> = Vec::new();

        // stem
        let f0 = self.c0.forward(img, s_img, b, h, w, train, stream, sc);
        let n0 = b * 64 * h * w;
        let (q0, s0) = relu_quant_owned(f0.y.as_f32(), n0, stream, sc);
        chain.push((q0.clone(), s0));
        let f1 = self.c1.forward(&q0, s0, b, h, w, train, stream, sc);
        let n1 = b * 128 * h * w;
        let (q1, s1) = relu_quant_owned(f1.y.as_f32(), n1, stream, sc);
        chain.push((q1.clone(), s1));
        // maxpool2x2 on the host over int8 (indices needed for backward)
        let mut q1_h = vec![0i8; n1];
        q1.download_i8(&mut q1_h, stream);
        stream.sync();
        let (pooled_h, idx) = crate::conv2d::maxpool2x2_with_indices(&q1_h, b, 128, h, w);
        h /= 2;
        w /= 2;
        let q_pool = Arc::new(DevBuf::new(pooled_h.len()));
        q_pool.upload_i8(&pooled_h, stream);
        let s_pool = s1; // pure selection: same scale
        chain.push((q_pool.clone(), s_pool));

        let stem_tr = StemTrace {
            c0: trace_of(f0, b, img_h, img_w, s_img, img.clone()),
            c1: trace_of(f1, b, h * 2, w * 2, s0, q0.clone()),
            q0: q0.clone(),
            s0,
            q1: q1.clone(),
            s1,
            pool_idx: idx,
            b,
            h_pool_in: h * 2,
            w_pool_in: w * 2,
        };
        // NOTE: stem c0 trace q_in unused (image gradient is discarded);
        // replace with the real image buffer below via NetTrace.s_img.

        let mut q_cur = q_pool;
        let mut s_cur = s_pool;
        let mut blocks = Vec::new();
        for layer in [&mut self.layer1, &mut self.layer2, &mut self.layer3, &mut self.layer4] {
            for blk in layer.iter_mut() {
                let (q, s, tr) = blk.forward(&q_cur, s_cur, b, h, w, train, stream, sc);
                h = tr.h_out;
                w = tr.w_out;
                chain.push((q.clone(), s));
                q_cur = q;
                s_cur = s;
                blocks.push(tr);
            }
        }

        // conv_out
        let fo0 = self.o0.forward(&q_cur, s_cur, b, h, w, train, stream, sc);
        let no0 = b * 512 * fo0.h_out * fo0.w_out;
        let (qo0, so0) = relu_quant_owned(fo0.y.as_f32(), no0, stream, sc);
        chain.push((qo0.clone(), so0));
        let (ho0, wo0) = (fo0.h_out, fo0.w_out);
        let fo1 = self.o1.forward(&qo0, so0, b, ho0, wo0, train, stream, sc);
        let no1 = b * 512 * fo1.h_out * fo1.w_out;
        let (qo1, so1) = relu_quant_owned(fo1.y.as_f32(), no1, stream, sc);
        chain.push((qo1.clone(), so1));
        let (hf, wf) = (fo1.h_out, fo1.w_out);

        // features: dequant -> mean over H -> [B, W, C]
        let fv = sc.f32(S_T1, no1);
        unsafe { launch_dequant_i8_f32(qo1.as_i8(), no1 as i64, so1, fv, stream.0) };
        let feat = DevBuf::new(b * wf * 512 * 4);
        unsafe { launch_meanpool_h(fv as *const f32, 512, hf as i32, wf as i32, b as i64, feat.as_f32_mut(), stream.0) };

        let out_tr = OutTrace {
            c0: trace_of(fo0, b, h, w, s_cur, q_cur.clone()),
            c1: trace_of(fo1, b, ho0, wo0, so0, qo0.clone()),
            q0: qo0,
            s0: so0,
            b,
            h_feat: hf,
            w_feat: wf,
        };
        (feat, NetTrace { stem: stem_tr, blocks, out: out_tr, q_chain: chain, s_img })
    }

    /// Backprop a `[B, T, 512]` feature gradient through the whole CNN.
    pub fn backward(&mut self, d_feat: &DevBuf, tr: &NetTrace, stream: &CudaStream, sc: &mut Scratch) {
        let b = tr.out.b;
        let (hf, wf) = (tr.out.h_feat, tr.out.w_feat);
        let no1 = b * 512 * hf * wf;

        // meanpool backward -> dL/d(conv_out1 output value)
        let dv = sc.f32(S_T1, no1);
        unsafe { launch_meanpool_h_backward(d_feat.as_f32(), 512, hf as i32, wf as i32, b as i64, dv, stream.0) };
        // value -> quant int gradient, then ReLU mask via qo1 sign
        let qo1 = &tr.q_chain.last().unwrap().0;
        let so1 = tr.q_chain.last().unwrap().1;
        unsafe { cuda_scale_vector(dv, so1, no1 as i64, stream.0) };
        let qv = sc.f32(S_T3, no1);
        unsafe { launch_dequant_i8_f32(qo1.as_i8(), no1 as i64, 1.0, qv, stream.0) };
        let dy1 = DevBuf::new(no1 * 4);
        unsafe { launch_mask_relu_bwd(qv as *const f32, dv as *const f32, no1 as i64, dy1.as_f32_mut(), stream.0) };
        let gv_o0 = self.o1.backward(dy1.as_f32(), &tr.out.c1, stream, sc); // dL/d v(o0 out)
        // ReLU mask of o0 via qo0
        let no0 = b * 512 * tr.out.c0.h_out * tr.out.c0.w_out;
        let qv0 = sc.f32(S_T3, no0);
        unsafe { launch_dequant_i8_f32(tr.out.q0.as_i8(), no0 as i64, 1.0, qv0, stream.0) };
        let dy0 = DevBuf::new(no0 * 4);
        unsafe { launch_mask_relu_bwd(qv0 as *const f32, gv_o0.as_f32(), no0 as i64, dy0.as_f32_mut(), stream.0) };
        let mut gv = self.o0.backward(dy0.as_f32(), &tr.out.c0, stream, sc); // dL/d v(layer4 out)

        // blocks, reversed
        let mut layers: Vec<&mut BasicBlock> = Vec::new();
        layers.extend(self.layer4.iter_mut());
        layers.extend(self.layer3.iter_mut());
        layers.extend(self.layer2.iter_mut());
        layers.extend(self.layer1.iter_mut());
        // chain indices: [q0,s0],[q1,s1],[pool], then one per block, then o0, o1
        let n_blocks = tr.blocks.len();
        for (i, blk) in layers.iter_mut().enumerate().rev() {
            let bi = n_blocks - 1 - i; // index into tr.blocks
            let btr = &tr.blocks[bi];
            // dq for this block output = gv * s_of_this_block_output
            let s_out = tr.q_chain[3 + bi].1;
            let q_out = &tr.q_chain[3 + bi].0;
            let n_out = gv.len / 4;
            unsafe { cuda_scale_vector(gv.as_f32_mut(), s_out, n_out as i64, stream.0) };
            let dqi = DevBuf::new(n_out);
            // quantise gv (now dL/dq_int) for the block backward
            let maxb = sc.f32(S_MAX, 1);
            unsafe { launch_reduce_max_abs(gv.as_f32(), n_out as i64, maxb, stream.0) };
            let mut mx = [0.0f32];
            unsafe { cuda_memcpy_d2h(mx.as_mut_ptr() as *mut CV, maxb as *mut CV, 4, stream.0) };
            stream.sync();
            let s_dq = (mx[0] / 127.0).max(1e-12);
            unsafe { launch_quant_f32_i8(gv.as_f32(), n_out as i64, s_dq, dqi.as_i8_mut(), stream.0) };
            gv = blk.backward(&dqi, s_dq, q_out, btr, stream, sc);
        }

        // stem: gv is dL/d v(pool output) == dL/d v(q1 selected)
        let s_pool = tr.stem.s1;
        let n_pool = gv.len / 4;
        unsafe { cuda_scale_vector(gv.as_f32_mut(), s_pool, n_pool as i64, stream.0) };
        // scatter through maxpool
        let idx64: Vec<i64> = tr.stem.pool_idx.iter().map(|&x| x as i64).collect();
        let idx_dev = DevBuf::new(idx64.len() * 8);
        unsafe {
            cuda_memcpy_h2d(idx_dev.ptr, idx64.as_ptr() as *const core::ffi::c_void, idx64.len() * 8, stream.0);
        }
        let n1 = b * 128 * tr.stem.h_pool_in * tr.stem.w_pool_in;
        let dpre = DevBuf::new(n1 * 4);
        unsafe {
            launch_maxpool2x2_backward(gv.as_f32(), idx_dev.ptr as *const i64, dpre.as_f32_mut(), (n_pool) as i64, n1 as i64, stream.0);
        }
        // ReLU mask of c1 via q1
        let qv1 = sc.f32(S_T3, n1);
        unsafe { launch_dequant_i8_f32(tr.stem.q1.as_i8(), n1 as i64, 1.0, qv1, stream.0) };
        let dyc1 = DevBuf::new(n1 * 4);
        unsafe { launch_mask_relu_bwd(qv1 as *const f32, dpre.as_f32(), n1 as i64, dyc1.as_f32_mut(), stream.0) };
        let gvc0 = self.c1.backward(dyc1.as_f32(), &tr.stem.c1, stream, sc);
        let n0 = b * 64 * tr.stem.h_pool_in * tr.stem.w_pool_in;
        let qv0 = sc.f32(S_T3, n0);
        unsafe { launch_dequant_i8_f32(tr.stem.q0.as_i8(), n0 as i64, 1.0, qv0, stream.0) };
        let dyc0 = DevBuf::new(n0 * 4);
        unsafe { launch_mask_relu_bwd(qv0 as *const f32, gvc0.as_f32(), n0 as i64, dyc0.as_f32_mut(), stream.0) };
        let _ = self.c0.backward(dyc0.as_f32(), &tr.stem.c0, stream, sc);
    }

    /// Hook running-stat updates (call right after forward with the trace).
    pub fn update_running_stats(&mut self, tr: &NetTrace) {
        let order: Vec<(Vec<f32>, Vec<f32>)> = {
            let mut v = vec![(tr.stem.c0.mean_h.clone(), tr.stem.c0.var_h.clone()), (tr.stem.c1.mean_h.clone(), tr.stem.c1.var_h.clone())];
            for bt in &tr.blocks {
                if let Some(d) = &bt.ds {
                    v.push((d.mean_h.clone(), d.var_h.clone()));
                }
                v.push((bt.c1.mean_h.clone(), bt.c1.var_h.clone()));
                v.push((bt.c2.mean_h.clone(), bt.c2.var_h.clone()));
            }
            v.push((tr.out.c0.mean_h.clone(), tr.out.c0.var_h.clone()));
            v.push((tr.out.c1.mean_h.clone(), tr.out.c1.var_h.clone()));
            v
        };
        for (c, (m, vv)) in self.convs_mut().into_iter().zip(order.into_iter()) {
            c.bn.update_running(&m, &vv);
        }
    }
}
