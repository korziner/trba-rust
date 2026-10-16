//! Bindings to the native backend (`cuda/kernels.cu`, or `cuda/cpu_backend.cpp`
//! when the project is built without CUDA).  Both implement the same
//! `extern "C"` surface, so the rest of the crate does not care which one got
//! linked in.
//!
//! Every pointer handed to a `launch_*` function is a *device* pointer
//! obtained from [`cuda_malloc`] / [`GpuWorkspace`] — never a `Vec` pointer.
//! (Passing host memory to a kernel was one of the crashes this port had.)

#![allow(dead_code)]

use std::ffi::{c_void, CStr};

extern "C" {
    pub fn cuda_malloc(ptr: *mut *mut c_void, size: usize);
    pub fn cuda_free(ptr: *mut c_void);
    pub fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void);
    pub fn cuda_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void);
    pub fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void);
    pub fn cuda_sync(stream: *mut c_void);
    pub fn cuda_stream_create(stream: *mut *mut c_void);
    pub fn cuda_stream_destroy(stream: *mut c_void);
    pub fn cuda_get_memory_info(free_mem: *mut usize, total_mem: *mut usize);

    pub fn launch_gemm_int8(a: *const i8, b: *const i8, c: *mut i32, m: i32, n: i32, k: i32, stream: *mut c_void);
    pub fn launch_gemm_fp16(a: *const c_void, b: *const c_void, c: *mut f32, m: i32, n: i32, k: i32, stream: *mut c_void);
    pub fn launch_gemm_fp32(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, stream: *mut c_void);
    pub fn launch_im2col_pad(input: *const i8, col: *mut i8, batch: i32, c: i32, h: i32, w: i32,
                             kh: i32, kw: i32, sh: i32, sw: i32, ph: i32, pw: i32,
                             ho: i32, wo: i32, col_ld: i32, stream: *mut c_void);
    pub fn launch_im2col_gpu(input: *const i8, col: *mut f32, batch: i32, c: i32, h: i32, w: i32,
                             kh: i32, kw: i32, sh: i32, sw: i32, ph: i32, pw: i32,
                             ho: i32, wo: i32, stream: *mut c_void);
    pub fn launch_conv_epilogue(acc: *const i32, bias: *const f32, out: *mut i8, acc_ld: i32,
                                acc_div: f32, out_mul: f32, b: i32, c: i32, ho: i32, wo: i32,
                                stream: *mut c_void);
    pub fn launch_conv_backward_grad_weight(input: *const i8, grad_out: *const f32, grad_w: *mut f32,
                                            b: i32, cin: i32, h: i32, w: i32, cout: i32,
                                            kh: i32, kw: i32, sh: i32, sw: i32, ph: i32, pw: i32,
                                            ho: i32, wo: i32, stream: *mut c_void);
    pub fn launch_conv_backward_grad_bias(grad_out: *const f32, grad_bias: *mut f32,
                                          b: i32, cout: i32, ho: i32, wo: i32, stream: *mut c_void);
    pub fn launch_conv_backward_grad_input(grad_out: *const f32, weight: *const f32,
                                           grad_input: *mut f32, b: i32, cin: i32, h: i32, w: i32,
                                           cout: i32, kh: i32, kw: i32, sh: i32, sw: i32,
                                           ph: i32, pw: i32, ho: i32, wo: i32, stream: *mut c_void);
    pub fn maxpool2x2_backward_host(grad_output: *const f32, max_indices: *const u64,
                                    grad_input: *mut f32, output_size: u64, input_size: u64);
    pub fn launch_maxpool2x2_backward(grad_output: *const f32, max_indices: *const i64,
                                      grad_input: *mut f32, output_size: i64, input_size: i64,
                                      stream: *mut c_void);
    pub fn cuda_float_to_half(src: *const f32, dst: *mut c_void, n: i64, stream: *mut c_void);
    pub fn cuda_scale_vector(data: *mut f32, scale: f32, n: i64, stream: *mut c_void);
    pub fn cuda_transpose_f32(src: *const f32, dst: *mut f32, rows: i32, cols: i32, stream: *mut c_void);

    pub fn launch_bn_stats(x: *const f32, b: i32, c: i32, hw: i32, mean: *mut f32, var: *mut f32, stream: *mut c_void);
    pub fn launch_bn_apply(x: *const f32, mean: *const f32, var: *const f32, gamma: *const f32, beta: *const f32, eps: f32, y: *mut f32, c: i32, hw: i32, n: i64, stream: *mut c_void);
    pub fn launch_bn_bwd_sums(dy: *const f32, x: *const f32, mean: *const f32, var: *const f32, eps: f32, c: i32, hw: i32, n: i64, sum_dy: *mut f32, sum_dy_xhat: *mut f32, stream: *mut c_void);
    pub fn launch_bn_bwd_dx(dy: *const f32, x: *const f32, mean: *const f32, var: *const f32, gamma: *const f32, sum_dy: *const f32, sum_dy_xhat: *const f32, eps: f32, c: i32, hw: i32, n: i64, count: i64, dx: *mut f32, stream: *mut c_void);
    pub fn launch_reduce_max_abs(x: *const f32, n: i64, out: *mut f32, stream: *mut c_void);
    pub fn launch_quant_f32_i8(x: *const f32, n: i64, scale: f32, q: *mut i8, stream: *mut c_void);
    pub fn launch_relu_quant(y: *const f32, n: i64, scale: f32, q: *mut i8, stream: *mut c_void);
    pub fn launch_dequant_i8_f32(q: *const i8, n: i64, scale: f32, x: *mut f32, stream: *mut c_void);
    pub fn launch_add_scaled_i8(a: *const i8, sa: f32, b: *const i8, sb: f32, n: i64, out: *mut f32, stream: *mut c_void);
    pub fn launch_epilogue_f32(acc: *const i32, acc_ld: i32, scale: f32, out: *mut f32, c: i32, ho: i32, wo: i32, total: i64, stream: *mut c_void);
    pub fn launch_avgpool_hw(x: *const f32, c: i32, hw: i32, b: i64, out: *mut f32, stream: *mut c_void);
    pub fn launch_avgpool_hw_backward(dy: *const f32, c: i32, hw: i32, b: i64, dx: *mut f32, stream: *mut c_void);
    pub fn launch_mul_gate(x: *const f32, gate: *const f32, c: i32, hw: i32, n: i64, out: *mut f32, stream: *mut c_void);
    pub fn launch_mask_relu_bwd(y: *const f32, dy: *const f32, n: i64, dx: *mut f32, stream: *mut c_void);
    pub fn launch_zero_f32(p: *mut f32, n: i64, stream: *mut c_void);
    pub fn launch_transpose_i8(src: *const i8, rows: i32, cols: i32, dst: *mut i8, stream: *mut c_void);
    pub fn launch_add_f32(a: *const f32, b: *const f32, n: i64, o: *mut f32, stream: *mut c_void);
    pub fn launch_mul_f32(a: *const f32, b: *const f32, n: i64, o: *mut f32, stream: *mut c_void);
    pub fn launch_meanpool_h(x: *const f32, c: i32, h: i32, w: i32, b: i64, out: *mut f32, stream: *mut c_void);
    pub fn launch_meanpool_h_backward(dy: *const f32, c: i32, h: i32, w: i32, b: i64, dx: *mut f32, stream: *mut c_void);
    pub fn trba_workspace_acquire(bytes: usize) -> *mut c_void;
    pub fn trba_workspace_size() -> usize;
    pub fn trba_workspace_release();
    pub fn trba_backend_name() -> *const std::os::raw::c_char;
    pub fn trba_device_name() -> *const std::os::raw::c_char;
    pub fn trba_device_cc_major() -> i32;
    pub fn trba_device_cc_minor() -> i32;
}

#[inline]
pub fn align16(x: usize) -> usize {
    (x + 15) & !15
}

pub fn backend_name() -> String {
    unsafe {
        let p = trba_backend_name();
        if p.is_null() { "unknown".into() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

pub fn device_name() -> String {
    unsafe {
        let p = trba_device_name();
        if p.is_null() { "unknown".into() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

pub fn device_cc() -> (i32, i32) {
    unsafe { (trba_device_cc_major(), trba_device_cc_minor()) }
}

pub fn memory_info() -> (usize, usize) {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { cuda_get_memory_info(&mut free, &mut total) };
    (free, total)
}

/// A CUDA stream.  On the CPU backend the handle is null and every call is a
/// no-op, which is exactly the semantics we want.
pub struct CudaStream(pub *mut c_void);

impl CudaStream {
    pub fn new() -> Self {
        let mut p: *mut c_void = std::ptr::null_mut();
        unsafe { cuda_stream_create(&mut p) };
        Self(p)
    }
    pub fn sync(&self) {
        unsafe { cuda_sync(self.0) };
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { cuda_stream_destroy(self.0) };
    }
}

// The raw pointer is only ever used from the thread that owns the model.
unsafe impl Send for CudaStream {}

/// Owns a device allocation.
pub struct DevBuf {
    pub ptr: *mut c_void,
    pub len: usize,
}

impl DevBuf {
    pub fn new(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { cuda_malloc(&mut ptr, len) };
        Self { ptr, len }
    }
    pub fn upload(&self, src: &[u8], stream: &CudaStream) {
        assert!(src.len() <= self.len, "DevBuf::upload overflow {} > {}", src.len(), self.len);
        unsafe { cuda_memcpy_h2d(self.ptr, src.as_ptr() as *const c_void, src.len(), stream.0) };
    }
    pub fn download(&self, dst: &mut [u8], stream: &CudaStream) {
        assert!(dst.len() <= self.len, "DevBuf::download overflow {} > {}", dst.len(), self.len);
        unsafe { cuda_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), stream.0) }
    }
    pub fn download_f32(&self, dst: &mut [f32], stream: &CudaStream) {
        self.download(as_bytes_mut_f32(dst), stream)
    }
    pub fn download_i8(&self, dst: &mut [i8], stream: &CudaStream) {
        self.download(as_bytes_mut_i8(dst), stream)
    }
    pub fn upload_f32(&self, src: &[f32], stream: &CudaStream) {
        self.upload(f32_as_bytes(src), stream)
    }
    pub fn upload_i8(&self, src: &[i8], stream: &CudaStream) {
        self.upload(as_bytes(src), stream)
    }
    pub fn as_i32(&self) -> *const i32 { self.ptr as *const i32 }
    pub fn as_i8(&self) -> *const i8 { self.ptr as *const i8 }
    pub fn as_i8_mut(&self) -> *mut i8 { self.ptr as *mut i8 }
    pub fn as_f32(&self) -> *const f32 { self.ptr as *const f32 }
    pub fn as_f32_mut(&self) -> *mut f32 { self.ptr as *mut f32 }
    pub fn as_i32_mut(&self) -> *mut i32 { self.ptr as *mut i32 }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_free(self.ptr) };
        }
    }
}

/// A named slice of the process-wide grow-only device arena.
///
/// The backward pass needs ~170 MB of scratch per layer.  Doing that with
/// `cudaMalloc`/`cudaFree` on every layer of every batch is slow and
/// fragments the device heap; one arena carved into regions is not.
pub struct GpuWorkspace {
    base: *mut c_void,
    regions: Vec<(usize, usize)>,
}

impl GpuWorkspace {
    /// `sizes` are bytes; every region starts on a 256-byte boundary.
    pub fn new(sizes: &[usize]) -> Self {
        let mut offs = Vec::with_capacity(sizes.len());
        let mut cur = 0usize;
        for &s in sizes {
            offs.push(cur);
            cur += (s + 255) & !255;
        }
        let base = unsafe { trba_workspace_acquire(cur) };
        assert!(!base.is_null() || cur == 0, "failed to acquire {} bytes of device workspace", cur);
        Self { base, regions: offs.into_iter().zip(sizes.iter().copied()).collect() }
    }

    pub fn total(&self) -> usize {
        self.regions.iter().map(|(_, s)| (s + 255) & !255).sum()
    }

    pub fn ptr(&self, i: usize) -> *mut c_void {
        let (off, _) = self.regions[i];
        unsafe { (self.base as *mut u8).add(off) as *mut c_void }
    }
    pub fn len(&self, i: usize) -> usize { self.regions[i].1 }

    pub fn as_i8(&self, i: usize) -> *const i8 { self.ptr(i) as *const i8 }
    pub fn as_i8_mut(&self, i: usize) -> *mut i8 { self.ptr(i) as *mut i8 }
    pub fn as_f32(&self, i: usize) -> *const f32 { self.ptr(i) as *const f32 }
    pub fn as_f32_mut(&self, i: usize) -> *mut f32 { self.ptr(i) as *mut f32 }
    pub fn as_i32_mut(&self, i: usize) -> *mut i32 { self.ptr(i) as *mut i32 }

    pub fn upload(&self, i: usize, src: &[u8], stream: &CudaStream) {
        assert!(src.len() <= self.len(i), "workspace region {} overflow: {} > {}", i, src.len(), self.len(i));
        unsafe { cuda_memcpy_h2d(self.ptr(i), src.as_ptr() as *const c_void, src.len(), stream.0) };
    }

    pub fn upload_i8(&self, i: usize, src: &[i8], stream: &CudaStream) {
        self.upload(i, as_bytes(src), stream);
    }

    pub fn upload_f32(&self, i: usize, src: &[f32], stream: &CudaStream) {
        self.upload(i, f32_as_bytes(src), stream);
    }

    pub fn download(&self, i: usize, dst: &mut [u8], stream: &CudaStream) {
        assert!(dst.len() <= self.len(i), "workspace region {} underflow: {} < {}", i, self.len(i), dst.len());
        unsafe { cuda_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, self.ptr(i), dst.len(), stream.0) };
    }

    pub fn download_i8(&self, i: usize, dst: &mut [i8], stream: &CudaStream) {
        self.download(i, as_bytes_mut_i8(dst), stream);
    }

    pub fn download_f32(&self, i: usize, dst: &mut [f32], stream: &CudaStream) {
        self.download(i, as_bytes_mut_f32(dst), stream);
    }
}

/// Reinterpret `&[i8]` as bytes (the workspace API is byte oriented).
pub fn as_bytes(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

/// Reinterpret `&mut [i8]` as bytes.
pub fn as_bytes_mut_i8(v: &mut [i8]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len()) }
}

/// Reinterpret `&mut [f32]` as bytes.
pub fn as_bytes_mut_f32(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}

/// Reinterpret `&[f32]` as bytes.
pub fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Reinterpret `&[i32]` as bytes.
pub fn i32_as_bytes(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Pad `data` ([rows, cols], row-major) into a [rows_p, cols_p] zero-filled
/// buffer.  The GEMM kernels require every operand to be padded to a multiple
/// of 16 in all three dimensions; the result matrix `C` is *not* padded.
pub fn pad_to_multiple_i8(data: &[i8], rows: usize, cols: usize, m: usize) -> (Vec<i8>, usize, usize) {
    let rp = ((rows + m - 1) / m) * m;
    let cp = ((cols + m - 1) / m) * m;
    let mut p = vec![0i8; rp * cp];
    for r in 0..rows {
        let src = &data[r * cols..(r + 1) * cols];
        p[r * cp..r * cp + cols].copy_from_slice(src);
    }
    (p, rp, cp)
}
