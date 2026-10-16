//! trba-rust library crate.
//!
//! Both binaries (`trba-rust`, `diagnose`) and the unit tests build on this,
//! so the kernels are exercised by `cargo test` on whatever backend build.rs
//! picked — which means the fused backward kernels are verified against a
//! naive reference even on a machine with no GPU.

#![allow(dead_code, unused_variables)]

pub mod attention;
pub mod backend;
pub mod bilstm;
pub mod conv2d;
pub mod gpu_memory;
pub mod linear;
pub mod qnet;
pub mod attn;
pub mod rnn;
pub mod opt;
pub mod seresnet;

#[cfg(test)]
mod tests;
pub mod ctc;
