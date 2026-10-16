//! Static device-side copies of the quantised weights, plus the memory report
//! printed at start-up.
//!
//! The training kernels upload the weights they need per call (through the
//! shared arena in `backend.rs`); these blobs exist so the run can report an
//! honest device-memory budget and so an inference-only path has a single
//! contiguous copy of every weight to read from.

#![allow(dead_code)]

use std::ffi::c_void;

extern "C" {
    fn allocate_model_weights(cnn: usize, bilstm: usize, attn: usize, ctc: usize);
    fn upload_cnn_weights(data: *const c_void, size: usize);
    fn upload_bilstm_weights(data: *const c_void, size: usize);
    fn upload_attention_weights(data: *const c_void, size: usize);
    fn upload_ctc_weights(data: *const c_void, size: usize);
    fn free_model_weights();
    fn trba_workspace_size() -> usize;
}

pub struct ModelWeightsGpu {
    pub cnn_size: usize,
    pub bilstm_size: usize,
    pub attn_size: usize,
    pub ctc_size: usize,
}

impl ModelWeightsGpu {
    pub fn allocate_and_upload(
        cnn: &[i8],
        bilstm: &[i8],
        attn: &[i8],
        ctc: &[i8],
    ) -> Self {
        unsafe {
            allocate_model_weights(cnn.len(), bilstm.len(), attn.len(), ctc.len());
            if !cnn.is_empty() {
                upload_cnn_weights(cnn.as_ptr() as *const c_void, cnn.len());
            }
            if !bilstm.is_empty() {
                upload_bilstm_weights(bilstm.as_ptr() as *const c_void, bilstm.len());
            }
            if !attn.is_empty() {
                upload_attention_weights(attn.as_ptr() as *const c_void, attn.len());
            }
            if !ctc.is_empty() {
                upload_ctc_weights(ctc.as_ptr() as *const c_void, ctc.len());
            }
        }
        Self {
            cnn_size: cnn.len(),
            bilstm_size: bilstm.len(),
            attn_size: attn.len(),
            ctc_size: ctc.len(),
        }
    }

    pub fn total(&self) -> usize {
        self.cnn_size + self.bilstm_size + self.attn_size + self.ctc_size
    }

    pub fn print_memory_breakdown(&self) {
        let mb = |b: usize| b as f64 / 1_048_576.0;

        println!("\n📦 Выделение GPU памяти для весов:");
        println!("   SEResNet31:  {} bytes ({:.2} MB)", self.cnn_size, mb(self.cnn_size));
        println!("   BiLSTM:      {} bytes ({:.2} MB)", self.bilstm_size, mb(self.bilstm_size));
        println!("   Attention:   {} bytes ({:.2} MB)", self.attn_size, mb(self.attn_size));
        println!("   CTC Head:    {} bytes ({:.2} MB)", self.ctc_size, mb(self.ctc_size));
        println!(
            "   ИТОГО:       {} bytes ({:.2} MB)",
            self.total(),
            mb(self.total())
        );

        let (free, total) = crate::backend::memory_info();
        let used = total.saturating_sub(free);
        println!("\n📊 ИСПОЛЬЗОВАНИЕ ПАМЯТИ ({}):", crate::backend::backend_name());
        println!("   Устройство:         {}", crate::backend::device_name());
        println!("   Всего памяти:       {:.0} MB", mb(total));
        println!(
            "   Используется:       {:.0} MB ({:.1}%)",
            mb(used),
            if total > 0 { used as f64 / total as f64 * 100.0 } else { 0.0 }
        );
        println!(
            "   Свободно:           {:.0} MB ({:.1}%)",
            mb(free),
            if total > 0 { free as f64 / total as f64 * 100.0 } else { 0.0 }
        );

        println!("\n📦 Breakdown:");
        println!(
            "   Контекст/драйвер:   ~{:.0} MB",
            mb(used.saturating_sub(self.total()))
        );
        println!("   Веса модели:        {:.2} MB", mb(self.total()));
        println!(
            "   Рабочая область:    {:.2} MB (выделяется по требованию, растёт до пика)",
            mb(unsafe { trba_workspace_size() })
        );
    }

    /// Rough device-memory budget of one training step, in bytes, for the
    /// given batch size.  Used to refuse batch sizes that cannot fit instead
    /// of dying in cudaMalloc halfway through epoch 1.
    pub fn estimate_step_bytes(batch: usize) -> usize {
        // The backward pass never materialises an im2col matrix (that was the
        // 38.6 GB allocation that used to kill the run at batch 1); what is
        // left is a handful of [B, C, H, W] fp32 buffers.
        //
        // Worst case over the graph, per image:
        //   * grad_input of conv0_0 (3x64x256 fp32)         = 196 kB
        //   * grad_output of the same layer (64x64x256 fp32) = 4 MB
        // Only reached once the stem is unfrozen, but budget for it anyway.
        let grad_input = batch * 3 * 64 * 256 * 4;
        let grad_out = batch * 64 * 64 * 256 * 4;
        // Forward int8 col matrices: stem (Cin*kH*kW = 27) and conv0_1 (576).
        let col_stem = batch * 64 * 256 * 32;
        let col_after_stem = batch * 32 * 128 * 640;
        // The model weight blob itself plus arena slack.
        grad_input + grad_out + col_stem.max(col_after_stem) + (48 << 20)
    }
}

impl Drop for ModelWeightsGpu {
    fn drop(&mut self) {
        unsafe { free_model_weights() };
    }
}
