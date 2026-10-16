//! SEResNet31-style backbone for OCR, quantisation-aware, INT8 forward.
//!
//! Stem(3->64->128) + 4 stages, 19 conv layers, output 512 channels at H/16,
//! W/16 — i.e. 4x16 for a 64x256 input, which is what the BiLSTM consumes.
//!
//! The forward pass keeps every conv *input* activation (host side, int8) plus
//! the maxpool argmax indices, so the backward pass can run without
//! recomputing anything.  At batch 256 that is ~350 MB of host RAM.

#![allow(dead_code, unused_mut, unused_assignments, unused_variables)]

use crate::backend::CudaStream;
use crate::conv2d::{maxpool2x2_with_indices, Conv2dInt8};

pub struct ConvActivation {
    pub input: Vec<i8>,
    pub h_in: usize,
    pub w_in: usize,
    pub batch: usize,
    pub layer_idx: usize,
}

pub struct MaxpoolActivation {
    pub h_in: usize,
    pub w_in: usize,
    pub batch: usize,
    pub channels: usize,
    /// For every output element: index of the maximum inside the *whole*
    /// input tensor, so the backward scatter needs no re-derivation.
    pub max_indices: Vec<u64>,
}

pub enum Activation {
    Conv(ConvActivation),
    Maxpool(MaxpoolActivation),
    /// Placeholder that keeps `activations[idx]` aligned with the graph while
    /// holding no data: the layer is frozen at the current stage, so its input
    /// (or the pool's argmax indices) will never be read by `backward_chain`.
    ///
    /// This is what makes batch 256 fit at all.  Recording every conv input
    /// costs ~17 GB of *host* memory at 64x256 (the stage-3 activations alone
    /// are 33.5 MB each at batch 256); recording only the trainable tail costs
    /// a few hundred MB while the network is mostly frozen.
    Skipped,
}

pub struct SEResNet31ForwardResult {
    pub output: Vec<i8>,
    pub out_h: usize,
    pub out_w: usize,
    pub activations: Vec<Activation>,
}

/// How many of the *last* conv layers each unfreezing stage trains.
/// Progressive unfreezing (arXiv:1912.12607) starts next to the loss and walks
/// towards the input, which is the opposite of what the previous version did —
/// it trained the stem while every gradient had to cross 12 frozen layers.
pub fn trainable_layer_count(stage: usize) -> usize {
    match stage {
        0 => 7,
        1 => 13,
        2 => 19,
        99 => 1,
        98 => 4,
        97 => 3,
        96 => 2,
        _ => 19,
    }
}

/// Host memory (bytes) needed to record the activations of a training step at
/// the given batch size and unfreezing stage.
///
/// Two things dominate: the int8 conv *inputs* of the trainable tail, and the
/// u64 argmax indices of every maxpool inside that tail.  At 64x256 a single
/// stem activation is 4.7 MB per image, so recording the whole graph would
/// cost ~17 GB of host RAM at batch 256 — recording only what the backward
/// pass can reach costs a few hundred MB while the network is mostly frozen.
pub fn estimate_activation_bytes(batch: usize, stage: usize) -> u64 {
    // Per image: (conv input bytes, conv output bytes) for the 19 convs, in
    // graph order.  64x256 input, five 2x2 maxpools -> 2x8 output.
    const CONV: [(usize, usize); 19] = [
        (3 * 64 * 256, 64 * 64 * 256),     //  0 conv0_0    64x256 -> 64x256
        (64 * 32 * 128, 128 * 32 * 128),   //  1 conv0_1    32x128 -> 32x128
        (128 * 16 * 64, 256 * 16 * 64),    //  2 layer1[0]  16x64  -> 16x64
        (256 * 16 * 64, 256 * 16 * 64),    //  3 layer1[1]
        (256 * 16 * 64, 256 * 16 * 64),    //  4 layer1[2]
        (256 * 8 * 32, 256 * 8 * 32),      //  5 layer2[0]   8x32  ->  8x32
        (256 * 8 * 32, 256 * 8 * 32),      //  6 layer2[1]
        (256 * 8 * 32, 256 * 8 * 32),      //  7 layer2[2]
        (256 * 8 * 32, 256 * 8 * 32),      //  8 layer2[3]
        (256 * 4 * 16, 512 * 4 * 16),      //  9 layer3[0]   4x16  ->  4x16
        (512 * 4 * 16, 512 * 4 * 16),      // 10 layer3[1]
        (512 * 4 * 16, 512 * 4 * 16),      // 11 layer3[2]
        (512 * 4 * 16, 512 * 4 * 16),      // 12 layer3[3]
        (512 * 4 * 16, 512 * 4 * 16),      // 13 layer3[4]
        (512 * 4 * 16, 512 * 4 * 16),      // 14 layer3[5]
        (512 * 2 * 8, 512 * 2 * 8),        // 15 layer4[0]   2x8   ->  2x8
        (512 * 2 * 8, 512 * 2 * 8),        // 16 layer4[1]
        (512 * 2 * 8, 512 * 2 * 8),        // 17 layer4[2]
        (512 * 2 * 8, 512 * 2 * 8),        // 18 conv_out
    ];
    // Conv whose *output* feeds a maxpool (the pool indices are recorded there).
    const POOLED: [usize; 5] = [0, 1, 4, 8, 14];

    let n = trainable_layer_count(stage).min(CONV.len());
    let first = CONV.len() - n;

    let mut bytes = 0u64;
    for (i, (in_bytes, out_bytes)) in CONV.iter().enumerate() {
        if i < first {
            continue;
        }
        bytes += (batch * in_bytes) as u64;
        if POOLED.contains(&i) {
            // one u64 argmax index per pool output element (= out/4)
            bytes += (batch * out_bytes / 4 * 8) as u64;
        }
    }
    bytes
}

pub struct SEResNet31 {
    pub conv0_0: Conv2dInt8,
    pub conv0_1: Conv2dInt8,
    pub layer1: Vec<Conv2dInt8>,
    pub layer2: Vec<Conv2dInt8>,
    pub layer3: Vec<Conv2dInt8>,
    pub layer4: Vec<Conv2dInt8>,
    pub conv_out: Conv2dInt8,
}

impl SEResNet31 {
    pub fn new() -> Self {
        Self {
            conv0_0: Conv2dInt8::new(3, 64, 3, 1, 1),
            conv0_1: Conv2dInt8::new(64, 128, 3, 1, 1),
            layer1: vec![
                Conv2dInt8::new(128, 256, 3, 1, 1),
                Conv2dInt8::new(256, 256, 3, 1, 1),
                Conv2dInt8::new(256, 256, 3, 1, 1),
            ],
            layer2: (0..4).map(|_| Conv2dInt8::new(256, 256, 3, 1, 1)).collect(),
            layer3: {
                let mut convs = vec![Conv2dInt8::new(256, 512, 3, 1, 1)];
                for _ in 0..5 {
                    convs.push(Conv2dInt8::new(512, 512, 3, 1, 1));
                }
                convs
            },
            layer4: (0..3).map(|_| Conv2dInt8::new(512, 512, 3, 1, 1)).collect(),
            conv_out: Conv2dInt8::new(512, 512, 3, 1, 1),
        }
    }

    pub fn trainable_layer_count(stage: usize) -> usize {
        trainable_layer_count(stage)
    }

    /// Inference forward: same graph as [`Self::forward_save`], activations
    /// dropped immediately so the host footprint stays at one layer.
    pub fn forward(&self, input: &[i8], batch: usize, stream: &CudaStream) -> Vec<i8> {
        use crate::conv2d::maxpool2x2_batched;

        let mut h = 64usize;
        let mut w = 256usize;

        macro_rules! conv {
            ($conv:expr, $inp:expr) => {{
                let (out, ho, wo) = $conv.forward_batched($inp, batch, h, w, stream);
                h = ho;
                w = wo;
                out
            }};
        }
        macro_rules! mp {
            ($inp:expr, $channels:expr) => {{
                let out = maxpool2x2_batched($inp, batch, $channels, h, w);
                h /= 2;
                w /= 2;
                out
            }};
        }

        let mut x = conv!(self.conv0_0, input);
        x = mp!(&x, 64);

        x = conv!(self.conv0_1, &x);
        x = mp!(&x, 128);

        for conv in &self.layer1 {
            x = conv!(conv, &x);
        }
        x = mp!(&x, 256);

        for blk in 0..2 {
            let identity = x.clone();
            let c1 = conv!(self.layer2[blk * 2], &x);
            let c2 = conv!(self.layer2[blk * 2 + 1], &c1);
            let n = batch * 256 * h * w;
            x = c2
                .iter()
                .zip(identity.iter())
                .take(n)
                .map(|(a, b)| (*a as i16 + *b as i16).clamp(-127, 127) as i8)
                .collect();
        }
        x = mp!(&x, 256);

        for conv in &self.layer3 {
            x = conv!(conv, &x);
        }
        x = mp!(&x, 512);

        for conv in &self.layer4 {
            x = conv!(conv, &x);
        }

        conv!(self.conv_out, &x)
    }

    /// Training forward that records *everything* (24 activations).
    ///
    /// Only useful for tests and diagnostics — a real training step should use
    /// [`Self::forward_save_stage`], which records just the trainable tail.
    #[allow(dead_code)]

    /// argmax is retained for the backward pass.
    pub fn forward_save(
        &self,
        input: &[i8],
        batch: usize,
        stream: &CudaStream,
    ) -> SEResNet31ForwardResult {
        self.forward_save_stage(input, batch, stream, 19)
    }

    /// Training forward: same graph, but only the activations the backward pass
    /// will actually touch are recorded.  `trainable_count` is
    /// [`trainable_layer_count`] for the current unfreezing stage.
    pub fn forward_save_stage(
        &self,
        input: &[i8],
        batch: usize,
        stream: &CudaStream,
        trainable_count: usize,
    ) -> SEResNet31ForwardResult {
        const N_CONV: usize = 19;
        const N_ACTS: usize = 24;
        let trainable_count = trainable_count.min(N_CONV);
        // Activation index of the n-th conv (0-based).  Five maxpools sit
        // after convs 0, 1, 4, 8 and 14, so:
        //   0,1, 2,3,4, 5,6,7,8, 9..14, 15,16,17, 18  ->  activation index
        //   conv 18 (conv_out) is activation 23, conv 15 is 20, conv 14 is 17.
        // `backward_chain` trains the *last* `trainable_count` convs, so the
        // first activation that has to be recorded is this one.
        let conv_act_idx = |n: usize| -> usize {
            n + [0usize, 1, 4, 8, 14].iter().filter(|&&p| n > p).count()
        };
        let first_trainable_act = if trainable_count == 0 {
            N_ACTS
        } else {
            conv_act_idx(N_CONV - trainable_count)
        };

        let mut activations: Vec<Activation> = Vec::with_capacity(N_ACTS);
        let mut layer_idx = 0usize;
        let mut h = 64usize;
        let mut w = 256usize;

        // Record the conv input, run it, advance the layer counter.
        macro_rules! conv_act {
            ($conv:expr, $inp:expr) => {{
                assert!(
                    !$inp.is_empty(),
                    "forward_save: empty conv input at layer_idx={layer_idx}"
                );
                assert!(
                    $inp.len() >= batch * $conv.in_channels * h * w,
                    "forward_save: conv input at layer_idx={} is {} bytes, need {}",
                    layer_idx,
                    $inp.len(),
                    batch * $conv.in_channels * h * w
                );
                let (out, ho, wo) = $conv.forward_batched($inp, batch, h, w, stream);
                if activations.len() >= first_trainable_act {
                    activations.push(Activation::Conv(ConvActivation {
                        input: $inp.to_vec(),
                        h_in: h,
                        w_in: w,
                        batch,
                        layer_idx,
                    }));
                } else {
                    activations.push(Activation::Skipped);
                }
                layer_idx += 1;
                h = ho;
                w = wo;
                out
            }};
        }

        macro_rules! mp_act {
            ($inp:expr, $channels:expr) => {{
                assert!($inp.len() >= batch * $channels * h * w, "maxpool input too small");
                if activations.len() >= first_trainable_act {
                    let (out, indices) = maxpool2x2_with_indices($inp, batch, $channels, h, w);
                    activations.push(Activation::Maxpool(MaxpoolActivation {
                        h_in: h,
                        w_in: w,
                        batch,
                        channels: $channels,
                        max_indices: indices,
                    }));
                    h /= 2;
                    w /= 2;
                    out
                } else {
                    // Frozen part of the graph: the plain pool keeps no indices.
                    let out = crate::conv2d::maxpool2x2_batched($inp, batch, $channels, h, w);
                    activations.push(Activation::Skipped);
                    h /= 2;
                    w /= 2;
                    out
                }
            }};
        }

        // Residual add of two int8 tensors, saturating.
        let add_tensors = |a: &[i8], b: &[i8], n: usize| -> Vec<i8> {
            let n = n.min(a.len()).min(b.len());
            let mut out = vec![0i8; n];
            for i in 0..n {
                out[i] = (a[i] as i16 + b[i] as i16).clamp(-127, 127) as i8;
            }
            out
        };

        // === STEM: 3 -> 64 -> 128, two 2x2 pools ===
        let mut x = conv_act!(self.conv0_0, input);
        x = mp_act!(&x, 64);

        x = conv_act!(self.conv0_1, &x);
        x = mp_act!(&x, 128);

        // === STAGE 1: 128 -> 256 (3 convs) + pool ===
        for conv in &self.layer1 {
            x = conv_act!(conv, &x);
        }
        x = mp_act!(&x, 256);

        // === STAGE 2: 256 -> 256 (2 residual blocks) + pool ===
        for blk in 0..2 {
            let identity = x.clone();
            let c1 = conv_act!(self.layer2[blk * 2], &x);
            let c2 = conv_act!(self.layer2[blk * 2 + 1], &c1);
            x = add_tensors(&c2, &identity, batch * 256 * h * w);
        }
        x = mp_act!(&x, 256);

        // === STAGE 3: 256 -> 512, then 5 x 512 -> 512, + pool ===
        x = conv_act!(self.layer3[0], &x);
        for conv in self.layer3.iter().skip(1) {
            x = conv_act!(conv, &x);
        }
        x = mp_act!(&x, 512);

        // === STAGE 4: 3 x 512 -> 512 ===
        for conv in &self.layer4 {
            x = conv_act!(conv, &x);
        }

        // === OUTPUT CONV ===
        let x_out = conv_act!(self.conv_out, &x);

        SEResNet31ForwardResult {
            output: x_out,
            out_h: h,
            out_w: w,
            activations,
        }
    }

    /// Backpropagate `grad_output` through the recorded activations, updating
    /// the last `trainable_layer_count(stage)` conv layers.
    ///
    /// Returns the gradient w.r.t. the earliest layer that was trained, so the
    /// caller can inspect it; gradients are *not* propagated through frozen
    /// layers (that is the point of freezing them).
    pub fn backward_chain(
        &mut self,
        forward_result: &SEResNet31ForwardResult,
        grad_output: &[f32],
        lr: f32,
        momentum: f32,
        stream: &CudaStream,
        stage: usize,
        diag: bool,
    ) -> Vec<f32> {
        let max_trainable = trainable_layer_count(stage);

        // Conv activations, from the last one backwards.
        let conv_acts: Vec<usize> = forward_result
            .activations
            .iter()
            .enumerate()
            .filter_map(|(i, a)| match a {
                Activation::Conv(_) => Some(i),
                _ => None,
            })
            .rev()
            .collect();

        let trainable: std::collections::HashSet<usize> =
            conv_acts.iter().take(max_trainable).copied().collect();

        if diag {
            eprintln!(
                "    [backward_chain] stage={} -> training {} of {} conv layers",
                stage,
                trainable.len(),
                conv_acts.len()
            );
        }

        let n_acts = forward_result.activations.len();
        let mut current_grad: Vec<f32> = grad_output.to_vec();
        let mut processed = 0usize;

        for act_idx in (0..n_acts).rev() {
            match &forward_result.activations[act_idx] {
                Activation::Conv(conv_act) => {
                    if !trainable.contains(&act_idx) {
                        if diag {
                            eprintln!(
                                "      frozen conv act_idx={} layer_idx={} - stopping here",
                                act_idx, conv_act.layer_idx
                            );
                        }
                        break;
                    }
                    if diag {
                        eprintln!(
                            "      conv act_idx={} layer_idx={} input={} B={} {}x{}",
                            act_idx,
                            conv_act.layer_idx,
                            conv_act.input.len(),
                            conv_act.batch,
                            conv_act.h_in,
                            conv_act.w_in
                        );
                    }
                    if let Some(layer_ptr) = self.get_layer_mut_by_idx(conv_act.layer_idx) {
                        current_grad = unsafe {
                            (*layer_ptr).backward_qat_gpu(
                                &conv_act.input,
                                &current_grad,
                                conv_act.batch,
                                conv_act.h_in,
                                conv_act.w_in,
                                lr,
                                momentum,
                                stream,
                                diag,
                            )
                        };
                        processed += 1;
                    } else {
                        eprintln!(
                            "    [backward_chain] WARNING: no layer for idx {}",
                            conv_act.layer_idx
                        );
                        break;
                    }
                }
                Activation::Skipped => {
                    if diag {
                        eprintln!(
                            "      act_idx={} not recorded (frozen at this stage) - stopping here",
                            act_idx
                        );
                    }
                    break;
                }
                Activation::Maxpool(pool_act) => {
                    let out_size = pool_act.batch
                        * pool_act.channels
                        * (pool_act.h_in / 2)
                        * (pool_act.w_in / 2);
                    let in_size =
                        pool_act.batch * pool_act.channels * pool_act.h_in * pool_act.w_in;

                    if current_grad.len() < out_size {
                        eprintln!(
                            "    [backward_chain] maxpool grad mismatch: have {}, need {} — stopping",
                            current_grad.len(),
                            out_size
                        );
                        break;
                    }
                    if diag {
                        eprintln!(
                            "      maxpool act_idx={} {}x{} -> {}x{}",
                            act_idx,
                            pool_act.h_in,
                            pool_act.w_in,
                            pool_act.h_in / 2,
                            pool_act.w_in / 2
                        );
                    }

                    // Both operands live in host memory here (the argmax
                    // indices come from the CPU maxpool), so this runs on the
                    // host.  Handing a Vec pointer to a CUDA kernel — which is
                    // what this used to do — is an instant illegal access.
                    let mut grad_input = vec![0.0f32; in_size];
                    unsafe {
                        crate::backend::maxpool2x2_backward_host(
                            current_grad.as_ptr(),
                            pool_act.max_indices.as_ptr(),
                            grad_input.as_mut_ptr(),
                            out_size as u64,
                            in_size as u64,
                        );
                    }
                    current_grad = grad_input;
                }
            }
        }

        if diag {
            eprintln!("    [backward_chain] updated {} layers", processed);
        }
        current_grad
    }

    pub fn get_all_layers(&self) -> Vec<&Conv2dInt8> {
        let mut layers = Vec::with_capacity(19);
        layers.push(&self.conv0_0);
        layers.push(&self.conv0_1);
        for l in &self.layer1 {
            layers.push(l);
        }
        for l in &self.layer2 {
            layers.push(l);
        }
        for l in &self.layer3 {
            layers.push(l);
        }
        for l in &self.layer4 {
            layers.push(l);
        }
        layers.push(&self.conv_out);
        layers
    }

    pub fn all_layers_mut(&mut self) -> Vec<*mut Conv2dInt8> {
        let mut layers = Vec::with_capacity(19);
        layers.push(&mut self.conv0_0 as *mut Conv2dInt8);
        layers.push(&mut self.conv0_1 as *mut Conv2dInt8);
        for l in &mut self.layer1 {
            layers.push(l as *mut Conv2dInt8);
        }
        for l in &mut self.layer2 {
            layers.push(l as *mut Conv2dInt8);
        }
        for l in &mut self.layer3 {
            layers.push(l as *mut Conv2dInt8);
        }
        for l in &mut self.layer4 {
            layers.push(l as *mut Conv2dInt8);
        }
        layers.push(&mut self.conv_out as *mut Conv2dInt8);
        layers
    }

    fn get_layer_mut_by_idx(&mut self, idx: usize) -> Option<*mut Conv2dInt8> {
        self.all_layers_mut().get(idx).copied()
    }

    pub fn param_count(&self) -> usize {
        self.get_all_layers().iter().map(|l| l.param_count()).sum()
    }
}

impl Default for SEResNet31 {
    fn default() -> Self {
        Self::new()
    }
}
