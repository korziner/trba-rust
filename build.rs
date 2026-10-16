// build.rs — compiles the native backend and links it into the Rust binaries.
//
// Two backends, same `extern "C"` surface:
//   * cuda/kernels.cu      via nvcc      (default when a CUDA toolkit is found)
//   * cuda/cpu_backend.cpp via the C++ compiler (fallback / TRBA_FORCE_CPU=1)
//
// Everything is linked *statically*, so the old
// `LD_LIBRARY_PATH=cuda/ ./target/release/trba-rust` dance is gone.

use std::path::{Path, PathBuf};
use std::process::Command;

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).unwrap_or_default().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TRBA_NVCC") {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(p) = std::env::var("CUDA_PATH") {
        let cand = PathBuf::from(p).join("bin").join("nvcc");
        if cand.is_file() {
            return Some(cand);
        }
    }
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        let cand = Path::new(root).join("bin").join("nvcc");
        if cand.is_file() {
            return Some(cand);
        }
    }
    // Last resort: whatever is on PATH.
    let out = Command::new("sh")
        .args(["-c", "command -v nvcc"])
        .output()
        .ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(PathBuf::from(s));
        }
    }
    None
}

/// Compute capabilities to embed.  Overridable with TRBA_CUDA_ARCH="75;86".
fn target_archs() -> Vec<String> {
    if let Ok(spec) = std::env::var("TRBA_CUDA_ARCH") {
        let v: Vec<String> = spec
            .split(|c: char| c == ';' || c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    // Ask the driver when there is one — that is always the right answer for
    // the machine actually doing the training.
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            let first = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .replace('.', "");
            if !first.is_empty() && first.chars().all(|c| c.is_ascii_digit()) {
                return vec![first];
            }
        }
    }
    // Otherwise ship a fatbin covering Turing..Blackwell consumer parts.
    // (TRBA is trained on a single machine; a fatbin keeps the repo portable.)
    vec!["75".into(), "86".into(), "89".into()]
}

/// `ar crus` the object into a static library the linker can consume.
fn archive(out_dir: &Path, obj: &Path, lib: &Path) -> Result<(), String> {
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".into());
    // Run inside out_dir and pass bare file names: `ar` stores whatever path
    // it is given in the archive member header, and a long or absolute member
    // name makes rustc reject the archive ("Unsupported archive identifier").
    let obj_name = obj.file_name().ok_or("bad object path")?;
    let lib_name = lib.file_name().ok_or("bad library path")?;
    // `ar r` updates members in place; a stale or truncated archive from an
    // earlier failed build would otherwise poison the result.
    let _ = std::fs::remove_file(lib);
    let out = Command::new(&ar)
        .arg("crus")
        .arg(lib_name)
        .arg(obj_name)
        .current_dir(out_dir)
        .output()
        .map_err(|e| format!("failed to run {ar}: {e}"))?;
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        return Err(format!("{ar} failed to create {}", lib.display()));
    }
    Ok(())
}

fn build_cuda(out_dir: &Path, nvcc: &Path) -> Result<(), String> {
    let archs = target_archs();
    println!("cargo:warning=trba-rust: building CUDA backend for arch {}", archs.join(", "));

    let mut gencode = Vec::new();
    for a in &archs {
        gencode.push(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    // PTX for the newest target so the driver can JIT for GPUs we did not
    // know about at build time.
    if let Some(last) = archs.iter().max() {
        gencode.push(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }
    // PTX for the newest target so future drivers can JIT it.
    if let Some(last) = archs.last() {
        gencode.push(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }

    let obj = out_dir.join("kernels.o");
    let mut cmd = Command::new(nvcc);
    cmd.arg("-O3")
        .arg("--std=c++17")
        .arg("-Xcompiler")
        .arg("-fPIC,-O3,-Wall")
        .arg("--expt-relaxed-constexpr")
        .arg("-lineinfo")
        .arg("-c")
        .arg("cuda/kernels.cu")
        .arg("-o")
        .arg(&obj);
    for g in &gencode {
        cmd.arg(g);
    }

    let out = cmd.output().map_err(|e| format!("failed to run nvcc: {e}"))?;
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        return Err("nvcc failed to compile cuda/kernels.cu".into());
    }
    archive(out_dir, &obj, &out_dir.join("libtrba_cuda.a"))?;

    // cudart lives in the toolkit; prefer the static one so the binary is
    // self-contained.
    let toolkit_lib = nvcc
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("lib64"))
        .filter(|p| p.exists());
    if let Some(dir) = &toolkit_lib {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    if let Ok(extra) = std::env::var("TRBA_CUDA_LIB") {
        for d in extra.split(':') {
            if !d.is_empty() {
                println!("cargo:rustc-link-search=native={d}");
            }
        }
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=trba_cuda");
    println!(
        "cargo:rustc-link-lib={}",
        if env_flag("TRBA_CUDA_SHARED_RT") { "dylib=cudart" } else { "static=cudart" }
    );
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=pthread");
    Ok(())
}

fn build_cpu(out_dir: &Path) -> Result<(), String> {
    println!("cargo:warning=trba-rust: building CPU fallback backend (no CUDA)");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".into());
    let obj = out_dir.join("cpu_backend.o");
    let lib = out_dir.join("libtrba_cpu.a");

    let compile = |openmp: bool| -> Result<std::process::Output, String> {
        let mut cmd = Command::new(&cxx);
        cmd.arg("-O3")
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg("-Wall")
            .arg("-c")
            .arg("cuda/cpu_backend.cpp")
            .arg("-o")
            .arg(&obj);
        if openmp {
            cmd.arg("-fopenmp");
        }
        cmd.output().map_err(|e| format!("failed to run {cxx}: {e}"))
    };

    let mut openmp = true;
    let out = compile(true)?;
    if !out.status.success() {
        // Retry without OpenMP (clang without libomp, exotic toolchains, ...).
        let out2 = compile(false)?;
        if !out2.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            eprintln!("{}", String::from_utf8_lossy(&out2.stderr));
            return Err(format!("{cxx} failed to compile cuda/cpu_backend.cpp"));
        }
        openmp = false;
    }
    if openmp {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
    archive(out_dir, &obj, &lib)?;

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=trba_cpu");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/cpu_backend.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TRBA_FORCE_CPU");
    println!("cargo:rerun-if-env-changed=TRBA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=TRBA_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    let nvcc = if env_flag("TRBA_FORCE_CPU") { None } else { find_nvcc() };

    let result = match &nvcc {
        Some(nvcc) => build_cuda(&out_dir, nvcc),
        None => Err("nvcc not found".into()),
    };

    match result {
        Ok(()) => {}
        Err(why) => {
            if nvcc.is_some() {
                // nvcc exists but failed: tell the user loudly, then fall back
                // so `cargo build` still produces a runnable (CPU) binary.
                println!("cargo:warning=trba-rust: CUDA build failed ({why}); falling back to CPU backend");
                build_cpu(&out_dir).unwrap_or_else(|e| panic!("{e}"));
            } else {
                build_cpu(&out_dir).unwrap_or_else(|e| panic!("{e}"));
            }
        }
    }
}
